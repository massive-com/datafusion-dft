//! `vortex_ohlc.scaled`: integers or decimals stored as their values divided by a power of a base,
//! with an exponent per block of [`BLOCK`] values, or one for the whole array.
//!
//! Decimal columns keep the trailing zeros of their scale: prices at a scale of 4 that move in
//! cents, volumes in round lots. How many differs by ticker, and so by block: a chunk holds many
//! tickers, a block mostly one. Each block is stored as its values divided by `base^exponent`,
//! and multiplied back when read. A block's exponent is the one that packs it smallest: a few
//! values that aren't its multiples (a sub-cent print among cents) are kept whole, as patches.
//! A divisor that isn't a power, the greatest common divisor of a chunk's values, say, is a base
//! with the exponent 1.
//!
//! - child `values`: the quotients, in the array's dtype and with its validity, compressed like
//!   any integers or decimals (blocked FoR and bit-packing suit them). A null or patched row holds
//!   its block's lowest quotient, so that it doesn't widen the block's range.
//! - child `exponents`: one `u8` per block, unless every block has the same.
//! - metadata: the base, 2 or more; the exponent, where every block has the same (and there is no
//!   `exponents` child); and where the patches start.
//! - patches: the rows whose values aren't their block's multiples, with those values. Integers
//!   only.
//!
//! An array with one divisor for every row (one exponent, and no patches) is read through
//! its values: filters, slices, takes, masks and casts of nullability go through to them, and so
//! do comparisons with a constant, rewritten to the quotients, so the values' own kernels run.
//! Any other array is filtered through its values' kernels too, which then multiplies only the
//! rows kept, and is decoded for the rest.
//!
//! Its schemes so far are base 10's: [`DecimalPow10Scheme`] divides a chunk of decimals by the
//! trailing zeros all its values have, and [`Pow10Scheme`] each block of a decimal's integer part
//! by those its values have.

use std::fmt::{Display, Formatter};
use std::hash::Hasher;
use std::ops::Range;

use num_traits::{checked_pow, AsPrimitive, PrimInt, ToPrimitive};
use prost::Message;
use vortex::array::arrays::dict::{TakeReduce, TakeReduceAdaptor};
use vortex::array::arrays::filter::{FilterReduce, FilterReduceAdaptor};
use vortex::array::arrays::slice::{SliceKernel, SliceReduce, SliceReduceAdaptor};
use vortex::array::arrays::{ConstantArray, DecimalArray, PrimitiveArray};
use vortex::array::buffer::BufferHandle;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::{DType, DecimalType, NativeDecimalType, Nullability, PType};
use vortex::array::optimizer::rules::ParentRuleSet;
use vortex::array::patches::{PatchSlotIndices, Patches, PatchesData, PatchesMetadata};
use vortex::array::scalar::{DecimalValue, Scalar};
use vortex::array::scalar_fn::fns::binary::CompareKernel;
use vortex::array::scalar_fn::fns::cast::{CastReduce, CastReduceAdaptor};
use vortex::array::scalar_fn::fns::mask::{Mask as MaskExpr, MaskReduce, MaskReduceAdaptor};
use vortex::array::scalar_fn::fns::operators::{CompareOperator, Operator};
use vortex::array::serde::ArrayChildren;
use vortex::array::smallvec::smallvec;
use vortex::array::vtable::{OperationsVTable, VTable, ValidityChild, ValidityVTableFromChild};
use vortex::array::{
    array_slots, match_each_integer_ptype, Array, ArrayEq, ArrayHash, ArrayId, ArrayParts,
    ArrayRef, ArraySlots, ArrayView, EqMode, ExecutionCtx, ExecutionResult, IntoArray,
    TypedArrayRef, VortexSessionExecute,
};
use vortex::buffer::Buffer;
use vortex::error::{
    vortex_bail, vortex_ensure, vortex_ensure_eq, vortex_err, vortex_panic, VortexResult,
};
use vortex::mask::Mask;
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

mod decimal;
mod encode;
mod filter;
mod scheme;

pub use decimal::{chunk_exponent, DecimalPow10Scheme};
pub(crate) use decimal::{decoded, predecoded};
pub use encode::scaled_encode;
pub use scheme::Pow10Scheme;
pub(crate) use scheme::{failed_estimate, is_decimal_values};

/// Values per exponent: one FastLanes block, as Vortex's blocked FoR and bit-packing use.
pub const BLOCK: usize = 1024;

/// Register the encoding and its filter, compare and slice kernels on `session`, and its
/// [`Pow10Scheme`] for the compressor (once).
pub fn register(session: &VortexSession) {
    use vortex::array::arrays::filter::FilterExecuteAdaptor;
    use vortex::array::arrays::slice::SliceExecuteAdaptor;
    use vortex::array::arrays::{Filter, Slice};
    use vortex::array::optimizer::kernels::ArrayKernelsExt;
    use vortex::array::scalar_fn::fns::binary::{Binary, CompareExecuteAdaptor};
    use vortex::array::scalar_fn::ScalarFnVTable;
    use vortex::array::session::ArraySessionExt;
    session.arrays().register(Scaled);
    session.kernels().register_execute_parent_kernel(
        Filter.id(),
        Scaled,
        FilterExecuteAdaptor(Scaled),
    );
    session.kernels().register_execute_parent_kernel(
        Binary.id(),
        Scaled,
        CompareExecuteAdaptor(Scaled),
    );
    session.kernels().register_execute_parent_kernel(
        Slice.id(),
        Scaled,
        SliceExecuteAdaptor(Scaled),
    );
    crate::register_scheme(session, &Pow10Scheme);
}

/// A [`Scaled`]-encoded array.
pub type ScaledArray = Array<Scaled>;

#[derive(Clone, Debug)]
pub struct Scaled;

#[array_slots(Scaled)]
pub struct ScaledSlots {
    /// The values divided by their block's divisor, in the array's dtype and with its validity.
    #[slot(0)]
    pub values: ArrayRef,
    /// One exponent (`u8`) per block of [`BLOCK`] values, unless every block has the same.
    #[slot(1)]
    pub exponents: Option<ArrayRef>,
    /// The patched rows.
    #[slot(2)]
    pub patch_indices: Option<ArrayRef>,
    /// Their values.
    #[slot(3)]
    pub patch_values: Option<ArrayRef>,
    /// Where each chunk's patches start, if the patches have it.
    #[slot(4)]
    pub patch_chunk_offsets: Option<ArrayRef>,
}

const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: ScaledSlots::PATCH_INDICES,
    values: ScaledSlots::PATCH_VALUES,
    chunk_offsets: ScaledSlots::PATCH_CHUNK_OFFSETS,
};

/// What a [`ScaledArray`] keeps besides its slots: the base, the exponent every block has if
/// they all have the same, and where its patches start.
#[derive(Clone, Debug)]
pub struct ScaledData {
    base: u64,
    exponent: Option<u8>,
    patches: Option<PatchesData>,
}

impl Display for ScaledData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base: {}", self.base)?;
        if let Some(exponent) = self.exponent {
            write!(f, ", exponent: {exponent}")?;
        }
        match &self.patches {
            Some(patches) => write!(f, ", patch_offset: {}", patches.offset()),
            None => Ok(()),
        }
    }
}

impl ArrayHash for ScaledData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        std::hash::Hash::hash(&self.base, state);
        std::hash::Hash::hash(&self.exponent, state);
        std::hash::Hash::hash(&self.patches, state);
    }
}

impl ArrayEq for ScaledData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.base == other.base && self.exponent == other.exponent && self.patches == other.patches
    }
}

#[derive(Clone, prost::Message)]
struct ScaledMetadata {
    #[prost(message, optional, tag = "1")]
    patches: Option<PatchesMetadata>,
    #[prost(uint64, tag = "2")]
    base: u64,
    #[prost(uint32, optional, tag = "3")]
    exponent: Option<u32>,
}

fn exponents_dtype() -> DType {
    DType::Primitive(PType::U8, Nullability::NonNullable)
}

fn slots(values: ArrayRef, exponents: Option<ArrayRef>, patches: Option<&Patches>) -> ArraySlots {
    let mut slots: ArraySlots = smallvec![Some(values), exponents];
    PatchesData::push_slots(&mut slots, patches);
    slots
}

impl Scaled {
    /// The array of `values[i] × base^exponents[i / BLOCK]`, but where `patches` have a value.
    /// `values` gives the array its dtype, integers or decimals, and its validity; `exponents` has
    /// one non-null `u8` per block of [`BLOCK`] values; `base` is 2 or more. A constant
    /// `exponents` is kept as the one exponent, in the metadata.
    pub fn try_new(
        values: ArrayRef,
        exponents: ArrayRef,
        base: u64,
        patches: Option<Patches>,
    ) -> VortexResult<ScaledArray> {
        let exponent = (exponents.as_constant())
            .map(|exponent| {
                (exponent.as_primitive().typed_value::<u8>())
                    .ok_or_else(|| vortex_err!("Scaled: a null exponent"))
            })
            .transpose()?;
        let exponents = exponent.is_none().then_some(exponents);
        Self::from_parts(values, exponents, exponent, base, patches)
    }

    /// The array of `values × base^exponent`: one divisor for every row.
    pub fn try_new_uniform(values: ArrayRef, base: u64, exponent: u8) -> VortexResult<ScaledArray> {
        Self::from_parts(values, None, Some(exponent), base, None)
    }

    /// The array of its parts: the exponents per block, or the one `exponent`.
    fn from_parts(
        values: ArrayRef,
        exponents: Option<ArrayRef>,
        exponent: Option<u8>,
        base: u64,
        patches: Option<Patches>,
    ) -> VortexResult<ScaledArray> {
        let (dtype, len) = (values.dtype().clone(), values.len());
        let data = ScaledData {
            base,
            exponent,
            patches: patches.as_ref().map(PatchesData::from_patches),
        };
        let slots = slots(values, exponents, patches.as_ref());
        Array::try_from_parts(ArrayParts::new(Scaled, dtype, len, data, slots))
    }
}

pub trait ScaledArrayExt: ScaledArraySlotsExt {
    /// The base the exponents are of.
    fn base(&self) -> u64 {
        self.base
    }

    /// The rows whose values aren't their block's multiples, with those values.
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches.as_ref(),
            self.len(),
            self.slots(),
            PATCH_SLOTS,
        )
    }

    /// The exponent every block has, if they all have the same.
    fn exponent(&self) -> Option<u8> {
        self.exponent
    }

    /// The divisor every row has, if the blocks share an exponent and no row is patched.
    fn divisor(&self) -> Option<i128> {
        if self.patches.is_some() {
            return None;
        }
        checked_pow(i128::from(self.base), usize::from(self.exponent()?))
    }
}

impl<T: TypedArrayRef<Scaled>> ScaledArrayExt for T {}

/// Each block's divisor, `base^exponent`, in the values' type.
enum Divisors<T> {
    /// Every block's.
    All(T),
    /// Block by block.
    PerBlock(Vec<T>),
}

/// `array`'s divisors in `T`, which they must fit.
fn divisors<T, A>(array: &A, ctx: &mut ExecutionCtx) -> VortexResult<Divisors<T>>
where
    T: PrimInt,
    A: ScaledArrayExt,
{
    let base = array.base();
    let fits = || vortex_err!("Scaled: a base of {base} for {}", array.dtype());
    let base = <T as num_traits::NumCast>::from(base).ok_or_else(fits)?;
    let power = |exponent: u8| {
        checked_pow(base, usize::from(exponent))
            .ok_or_else(|| vortex_err!("Scaled: {}^{exponent} overflows", array.base()))
    };
    if let Some(exponent) = array.exponent() {
        return Ok(Divisors::All(power(exponent)?));
    }
    let exponents = (array.exponents())
        .ok_or_else(|| vortex_err!("Scaled: neither an exponent nor exponents"))?;
    let exponents = exponents.clone().execute::<PrimitiveArray>(ctx)?;
    let divisors = exponents.as_slice::<u8>().iter().map(|&e| power(e));
    Ok(Divisors::PerBlock(divisors.collect::<VortexResult<_>>()?))
}

/// A valid scalar of an integer or decimal dtype, as an integer (a decimal unscaled).
fn to_i128(scalar: &Scalar) -> Option<i128> {
    match scalar.dtype() {
        DType::Decimal(..) => scalar.as_decimal().decimal_value()?.cast::<i128>(),
        dtype if dtype.is_int() => match_each_integer_ptype!(dtype.as_ptype(), |T| {
            scalar.as_primitive().typed_value::<T>()?.to_i128()
        }),
        _ => None,
    }
}

/// `value` as a scalar of `dtype`, an integer's or a decimal's, if it fits.
fn from_i128(value: i128, dtype: &DType) -> Option<Scalar> {
    match dtype {
        DType::Decimal(decimal, nullability) => Some(Scalar::decimal(
            DecimalValue::I128(value),
            *decimal,
            *nullability,
        )),
        _ if dtype.is_int() => match_each_integer_ptype!(dtype.as_ptype(), |T| {
            let value = <T as num_traits::NumCast>::from(value)?;
            Some(Scalar::primitive(value, dtype.nullability()))
        }),
        _ => None,
    }
}

/// Whether `base` fits the values of `dtype`.
fn base_fits(base: u64, dtype: &DType) -> bool {
    match dtype {
        DType::Decimal(..) => true,
        _ => match_each_integer_ptype!(dtype.as_ptype(), |T| {
            <T as num_traits::NumCast>::from(base).is_some()
        }),
    }
}

impl VTable for Scaled {
    type TypedArrayData = ScaledData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex_ohlc.scaled");
        *ID
    }

    fn validate(
        &self,
        data: &ScaledData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = ScaledSlotsView::from_slots(slots);
        vortex_ensure!(
            dtype.is_int() || dtype.is_decimal(),
            "Scaled arrays are integers or decimals, not {dtype}"
        );
        if let Some(patches) =
            PatchesData::patches_from_slots(data.patches.as_ref(), len, slots, PATCH_SLOTS)
        {
            vortex_ensure!(dtype.is_int(), "Scaled: patches for {dtype}, not integers");
            vortex_ensure_eq!(patches.array_len(), len, "Scaled patches' length");
            vortex_ensure_eq!(patches.dtype(), dtype, "Scaled patches' dtype");
        }
        vortex_ensure_eq!(view.values.dtype(), dtype, "Scaled values' dtype");
        vortex_ensure_eq!(view.values.len(), len, "Scaled values' length");
        match (data.exponent, view.exponents) {
            (Some(_), None) => {}
            (None, Some(exponents)) => {
                vortex_ensure_eq!(exponents.dtype(), &exponents_dtype(), "Scaled exponents");
                vortex_ensure_eq!(
                    exponents.len(),
                    len.div_ceil(BLOCK),
                    "Scaled needs an exponent per block"
                );
            }
            _ => vortex_bail!("Scaled has either one exponent or an exponents child"),
        }
        vortex_ensure!(
            data.base >= 2 && base_fits(data.base, dtype),
            "Scaled: a base of {} for {dtype}",
            data.base
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Scaled has no buffer {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex::array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let patches = array
            .patches()
            .map(|patches| patches.to_metadata(array.len(), array.dtype()))
            .transpose()?;
        let (base, exponent) = (array.base, array.exponent.map(u32::from));
        let metadata = ScaledMetadata {
            patches,
            base,
            exponent,
        };
        Ok(Some(metadata.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ScaledMetadata::decode(metadata)?;
        let exponent = (metadata.exponent)
            .map(|e| u8::try_from(e).map_err(|_| vortex_err!("Scaled: an exponent of {e}")))
            .transpose()?;
        let values = children.get(0, dtype, len)?;
        // The children are the slots that are there: the exponents only without one exponent.
        let exponents = match exponent {
            Some(_) => None,
            None => Some(children.get(1, &exponents_dtype(), len.div_ceil(BLOCK))?),
        };
        let first_patch = 1 + usize::from(exponents.is_some());
        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(first_patch, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(first_patch + 1, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| {
                        let len = usize::try_from(p.chunk_offsets_len())?;
                        children.get(first_patch + 2, &dtype, len)
                    })
                    .transpose()?;
                let mut ctx = session.create_execution_ctx();
                Patches::from_metadata(len, &p, indices, values, chunk_offsets, &mut ctx)
            })
            .transpose()?;
        let data = ScaledData {
            base: metadata.base,
            exponent,
            patches: patches.as_ref().map(PatchesData::from_patches),
        };
        let slots = slots(values, exponents, patches.as_ref());
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ScaledSlots::NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if let DType::Decimal(decimal, _) = array.dtype() {
            let values = array.values().clone().execute::<DecimalArray>(ctx)?;
            let multiplied = multiplied_decimals(&values, divisors::<i128, _>(&array, ctx)?)?;
            let decoded = DecimalArray::new(multiplied, *decimal, values.validity()?);
            return Ok(ExecutionResult::done(decoded.into_array()));
        }
        let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = values.validity()?;
        let decoded = match_each_integer_ptype!(values.ptype(), |T| {
            let divisors = divisors::<T, _>(&array, ctx)?;
            let mut values = values.into_buffer_mut::<T>();
            match divisors {
                Divisors::All(divisor) => {
                    (values.as_mut_slice().iter_mut()).for_each(|v| *v = v.wrapping_mul(divisor))
                }
                Divisors::PerBlock(divisors) => {
                    let blocks = values.as_mut_slice().chunks_mut(BLOCK);
                    for (block, &divisor) in blocks.zip(&divisors) {
                        if divisor != 1 {
                            block.iter_mut().for_each(|v| *v = v.wrapping_mul(divisor));
                        }
                    }
                }
            }
            PrimitiveArray::new(values.freeze(), validity)
        });
        let decoded = match array.patches() {
            Some(patches) => decoded.patch(&patches, ctx)?,
            None => decoded,
        };
        Ok(ExecutionResult::done(decoded.into_array()))
    }
}

/// A decimal array's unscaled values times their blocks' `divisors`, as i128s.
///
/// At the widest width a decimal of up to 38 digits needs, whatever the values' width.
/// *A future Vortex*: vortex#10701 is to give every decimal leaf its precision's one width (i64 up
/// to 18 digits). Vortex 0.91 has the `Widen` array but still accepts any width; once a release
/// enforces it, this i128 buffer is invalid for a narrower precision, so build that width, or a
/// `Widen` of the values.
fn multiplied_decimals(
    values: &DecimalArray,
    divisors: Divisors<i128>,
) -> VortexResult<Buffer<i128>> {
    fn multiply<T: NativeDecimalType + AsPrimitive<i128>>(
        values: &[T],
        divisors: &Divisors<i128>,
    ) -> Buffer<i128> {
        match divisors {
            Divisors::All(divisor) => Buffer::from_trusted_len_iter(
                values.iter().map(|&v| v.as_().wrapping_mul(*divisor)),
            ),
            Divisors::PerBlock(divisors) => Buffer::from_trusted_len_iter(
                (values.iter().enumerate())
                    .map(|(i, &v)| v.as_().wrapping_mul(divisors[i / BLOCK])),
            ),
        }
    }
    Ok(match values.values_type() {
        DecimalType::I8 => multiply(&values.buffer::<i8>(), &divisors),
        DecimalType::I16 => multiply(&values.buffer::<i16>(), &divisors),
        DecimalType::I32 => multiply(&values.buffer::<i32>(), &divisors),
        DecimalType::I64 => multiply(&values.buffer::<i64>(), &divisors),
        DecimalType::I128 => multiply(&values.buffer::<i128>(), &divisors),
        other => vortex_bail!("Scaled: decimals stored as {other:?}"),
    })
}

impl OperationsVTable<Scaled> for Scaled {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Scaled>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = array.values().execute_scalar(index, ctx)?;
        if value.is_null() {
            return Ok(value);
        }
        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index, ctx)? {
                return patch.cast(array.dtype());
            }
        }
        let exponent = match array.exponent() {
            Some(exponent) => exponent,
            None => {
                let exponents = (array.exponents())
                    .ok_or_else(|| vortex_err!("Scaled: neither an exponent nor exponents"))?;
                let exponent = exponents.execute_scalar(index / BLOCK, ctx)?;
                (exponent.as_primitive().typed_value::<u8>())
                    .ok_or_else(|| vortex_err!("Scaled: a null exponent"))?
            }
        };
        let divisor = checked_pow(i128::from(array.base), usize::from(exponent));
        let product = (to_i128(&value).zip(divisor)).and_then(|(v, d)| v.checked_mul(d));
        (product.and_then(|product| from_i128(product, array.dtype()))).ok_or_else(|| {
            vortex_err!(
                "Scaled: {value} × {}^{exponent} is no {}",
                array.base,
                array.dtype()
            )
        })
    }
}

impl ValidityChild<Scaled> for Scaled {
    fn validity_child(array: ArrayView<'_, Scaled>) -> ArrayRef {
        array.values().clone()
    }
}

const PARENT_RULES: ParentRuleSet<Scaled> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Scaled)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Scaled)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Scaled)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Scaled)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Scaled)),
]);

/// `array` with `f` applied to its values, which moves rows (a filter, a slice, a take): only
/// where every row has the same divisor, so the blocks needn't be kept apart.
fn map_uniform(
    array: ArrayView<'_, Scaled>,
    f: impl FnOnce(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<Option<ArrayRef>> {
    let Some(exponent) = array.exponent().filter(|_| array.patches.is_none()) else {
        return Ok(None);
    };
    let values = f(array.values())?;
    Ok(Some(
        Scaled::try_new_uniform(values, array.base, exponent)?.into_array(),
    ))
}

/// `array` with `f` applied to its values, which keeps every row where it is (a mask, a cast):
/// any array without patches, whose values would otherwise disagree with theirs.
fn map_in_place(
    array: ArrayView<'_, Scaled>,
    f: impl FnOnce(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<Option<ArrayRef>> {
    if array.patches.is_some() {
        return Ok(None);
    }
    let values = f(array.values())?;
    let exponents = array.exponents().cloned();
    let scaled = Scaled::from_parts(values, exponents, array.exponent, array.base, None)?;
    Ok(Some(scaled.into_array()))
}

impl FilterReduce for Scaled {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        map_uniform(array, |values| values.filter(mask.clone()))
    }
}

impl SliceReduce for Scaled {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        if let Some(sliced) = map_uniform(array, |values| values.slice(range.clone()))? {
            return Ok(Some(sliced));
        }
        slice_blocks(array, range, None)
    }
}

impl SliceKernel for Scaled {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice_blocks(array, range, Some(ctx))
    }
}

/// `array` cut to the blocks that hold `range`, then sliced within them: an array whose blocks
/// have their own divisors, or with patches, so slicing it decodes those blocks and not the rest
/// (a needle's rows are often one range of a chunk). `None` if the blocks are all of `array`, or
/// if it has patches and there is no `ctx` to slice them with.
fn slice_blocks(
    array: ArrayView<'_, Scaled>,
    range: Range<usize>,
    ctx: Option<&mut ExecutionCtx>,
) -> VortexResult<Option<ArrayRef>> {
    let start = range.start / BLOCK * BLOCK;
    let end = (range.end.div_ceil(BLOCK) * BLOCK).min(array.len());
    if start == 0 && end == array.len() {
        return Ok(None);
    }
    let values = array.values().slice(start..end)?;
    let exponents = (array.exponents())
        .map(|exponents| exponents.slice(start / BLOCK..end.div_ceil(BLOCK)))
        .transpose()?;
    let patches = match (array.patches(), ctx) {
        (None, _) => None,
        (Some(patches), Some(ctx)) => patches.slice(start..end, ctx)?,
        (Some(_), None) => return Ok(None),
    };
    let blocks = Scaled::from_parts(values, exponents, array.exponent, array.base, patches)?;
    blocks
        .into_array()
        .slice(range.start - start..range.end - start)
        .map(Some)
}

impl TakeReduce for Scaled {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        map_uniform(array, |values| values.take(indices.clone()))
    }
}

impl MaskReduce for Scaled {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        map_in_place(array, |values| {
            Ok(MaskExpr::try_new(values.clone(), mask.clone())?.into_array())
        })
    }
}

impl CastReduce for Scaled {
    /// Only a change of nullability: any other cast decodes.
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.eq_ignore_nullability(array.dtype()) {
            return Ok(None);
        }
        map_in_place(array, |values| {
            values.cast(values.dtype().with_nullability(dtype.nullability()))
        })
    }
}

impl CompareKernel for Scaled {
    /// Where every row has the divisor `d`: `values × d` against a constant `c` of the array's
    /// dtype is `values` against `c / d` when `c` is a multiple of `d`. When it isn't,
    /// `values × d` is above `c` exactly when `values` is above the multiple below it, `⌊c / d⌋`,
    /// and never equal to it (which this leaves to the decoded values).
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(divisor) = lhs.divisor() else {
            return Ok(None);
        };
        if !rhs.dtype().eq_ignore_nullability(lhs.dtype()) {
            return Ok(None);
        }
        let Some(value) = rhs.as_constant().and_then(to_i128) else {
            return Ok(None);
        };
        let operator = match (operator, value.rem_euclid(divisor) == 0) {
            (operator, true) => operator,
            (CompareOperator::Gt | CompareOperator::Gte, false) => CompareOperator::Gt,
            (CompareOperator::Lt | CompareOperator::Lte, false) => CompareOperator::Lte,
            (CompareOperator::Eq | CompareOperator::NotEq, false) => return Ok(None),
        };
        let Some(quotient) = from_i128(value.div_euclid(divisor), rhs.dtype()) else {
            return Ok(None);
        };
        let quotient = ConstantArray::new(quotient, rhs.len()).into_array();
        lhs.values()
            .binary(quotient, Operator::from(operator))
            .map(Some)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use vortex::array::arrays::Slice;
    use vortex::array::validity::Validity;
    use vortex::array::{Canonical, VortexSessionExecute};
    use vortex::dtype::DecimalDType;

    use super::*;
    use crate::testing::session;

    /// What a comparison is on numbers.
    pub(crate) type Holds = fn(i128, i128) -> bool;

    /// The six comparisons, and what each is on numbers.
    pub(crate) const OPERATORS: [(Operator, Holds); 6] = [
        (Operator::Gt, |a, b| a > b),
        (Operator::Gte, |a, b| a >= b),
        (Operator::Lt, |a, b| a < b),
        (Operator::Lte, |a, b| a <= b),
        (Operator::Eq, |a, b| a == b),
        (Operator::NotEq, |a, b| a != b),
    ];

    /// `array` decoded, as integers (decimals unscaled): the same read whole and a row at a time.
    fn decoded(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Vec<i128> {
        let canonical = array
            .clone()
            .execute::<Canonical>(ctx)
            .unwrap()
            .into_array();
        let mut read = |array: &ArrayRef| -> Vec<i128> {
            (0..array.len())
                .map(|i| to_i128(&array.execute_scalar(i, ctx).unwrap()).unwrap())
                .collect()
        };
        let (whole, by_row) = (read(&canonical), read(array));
        assert_eq!(whole, by_row, "decoded whole and a row at a time");
        whole
    }

    /// `array`, a Scaled array, against `constant` through Scaled's compare kernel, called
    /// directly: where it holds, or `None` if the kernel leaves it to the decoded values.
    pub(crate) fn kernel_compare(
        array: &ArrayRef,
        constant: i128,
        operator: Operator,
        ctx: &mut ExecutionCtx,
    ) -> Option<Vec<bool>> {
        let literal = from_i128(constant, &array.dtype().as_nonnullable()).unwrap();
        let literal = ConstantArray::new(literal, array.len()).into_array();
        let operator = CompareOperator::try_from(operator).unwrap();
        let scaled = array.as_opt::<Scaled>().expect("a Scaled array");
        let result =
            <Scaled as CompareKernel>::compare(scaled, &literal, operator, ctx).unwrap()?;
        let result = result.execute::<Mask>(ctx).unwrap();
        Some((0..array.len()).map(|i| result.value(i)).collect())
    }

    /// Compared with each of `constants`, filtered, sliced and taken, `array` (six values) gives
    /// what its decoded values do. With one divisor for every row, the kernel answers every
    /// comparison with a multiple of it, and all but = and != with another.
    fn operations_match_decoded(array: &ArrayRef, constants: &[i128], ctx: &mut ExecutionCtx) {
        let divisor = array.as_opt::<Scaled>().unwrap().divisor();
        let values = decoded(array, ctx);
        for &constant in constants {
            let literal = from_i128(constant, &array.dtype().as_nonnullable()).unwrap();
            for (operator, expected) in OPERATORS {
                let literal = ConstantArray::new(literal.clone(), array.len()).into_array();
                let result = array.binary(literal, operator).unwrap();
                let result = result.execute::<Mask>(ctx).unwrap();
                let expected: Vec<bool> = values.iter().map(|&v| expected(v, constant)).collect();
                let result: Vec<bool> = (0..array.len()).map(|i| result.value(i)).collect();
                assert_eq!(result, expected, "{operator:?} {constant}");
                let answerable = divisor.is_some_and(|divisor| {
                    constant % divisor == 0 || !matches!(operator, Operator::Eq | Operator::NotEq)
                });
                match kernel_compare(array, constant, operator, ctx) {
                    Some(kernel) => assert_eq!(kernel, expected, "kernel: {operator:?} {constant}"),
                    None => assert!(!answerable, "the kernel left {operator:?} {constant}"),
                }
            }
        }
        let rows = |indices: &[usize]| indices.iter().map(|&i| values[i]).collect::<Vec<_>>();
        let mask = Mask::from_iter([true, false, true, true, false, true]);
        let filtered = array.filter(mask).unwrap();
        assert_eq!(decoded(&filtered, ctx), rows(&[0, 2, 3, 5]));
        let sliced = array.slice(1..4).unwrap();
        assert_eq!(decoded(&sliced, ctx), rows(&[1, 2, 3]));
        let indices: Buffer<u32> = [5u32, 0].into_iter().collect();
        let taken = array.take(indices.into_array()).unwrap();
        assert_eq!(decoded(&taken, ctx), rows(&[5, 0]));
    }

    /// Integers with one divisor that is no power of ten: 60, as the base with the exponent 1.
    #[test]
    fn one_divisor_for_integers() {
        let mut ctx = session().create_execution_ctx();
        let quotients = PrimitiveArray::from_iter([-3i32, 0, 1, 2, 5, 9]);
        let scaled = Scaled::try_new_uniform(quotients.into_array(), 60, 1)
            .unwrap()
            .into_array();
        assert_eq!(decoded(&scaled, &mut ctx), [-180, 0, 60, 120, 300, 540]);
        let constants = [-181, -180, -179, 0, 90, 120, 539, 540, 541];
        operations_match_decoded(&scaled, &constants, &mut ctx);
    }

    /// Decimals with one divisor, `10^2`: quotients of `Decimal(10, 2)` read back at its scale.
    #[test]
    fn one_divisor_for_decimals() {
        let mut ctx = session().create_execution_ctx();
        let dtype = DecimalDType::new(10, 2);
        let quotients: Buffer<i64> = [-3i64, 0, 1, 2, 5, 9].into_iter().collect();
        let values = DecimalArray::new(quotients, dtype, Validity::NonNullable);
        let scaled = Scaled::try_new_uniform(values.into_array(), 10, 2)
            .unwrap()
            .into_array();
        assert_eq!(
            scaled.dtype(),
            &DType::Decimal(dtype, Nullability::NonNullable)
        );
        assert_eq!(decoded(&scaled, &mut ctx), [-300, 0, 100, 200, 500, 900]);
        let constants = [-301, -300, -299, 0, 150, 200, 899, 900, 901];
        operations_match_decoded(&scaled, &constants, &mut ctx);
    }

    /// An exponent per block, in base 2: decoded and filtered block by block, and compared by
    /// decoding, as no one divisor holds for every row.
    #[test]
    fn exponents_per_block_in_any_base() {
        let mut ctx = session().create_execution_ctx();
        let len = 2 * BLOCK + 6;
        let quotients = PrimitiveArray::from_iter((0..len as i64).map(|i| i % 7 - 3));
        let exponents = Buffer::from(vec![3u8, 0, 5]).into_array();
        let scaled = Scaled::try_new(quotients.into_array(), exponents, 2, None)
            .unwrap()
            .into_array();
        let want: Vec<i128> = (0..len as i128)
            .map(|i| (i % 7 - 3) * [8, 1, 32][i as usize / BLOCK])
            .collect();
        assert_eq!(decoded(&scaled, &mut ctx), want);
        let mask = Mask::from_indices(len, [0, 1500, 2050]);
        let filtered = scaled.filter(mask).unwrap();
        assert_eq!(
            decoded(&filtered, &mut ctx),
            [want[0], want[1500], want[2050]]
        );
        let literal = Scalar::primitive(0i64, Nullability::NonNullable);
        let literal = ConstantArray::new(literal, len).into_array();
        let positive = scaled.binary(literal, Operator::Gt).unwrap();
        let positive = positive.execute::<Mask>(&mut ctx).unwrap();
        assert_eq!(
            positive.true_count(),
            want.iter().filter(|&&v| v > 0).count()
        );
    }

    /// A slice of an array with an exponent per block and patches reads only the blocks that hold
    /// it, whether it starts or ends mid-block, spans blocks, or is all of the array, and gives
    /// what slicing the decoded values does.
    #[test]
    fn slices_read_their_blocks() {
        let mut ctx = session().create_execution_ctx();
        let len = 3 * BLOCK + 6;
        let quotients = PrimitiveArray::from_iter((0..len as i64).map(|i| i % 7 - 3));
        let exponents = Buffer::from(vec![3u8, 0, 5, 1]).into_array();
        // Rows whose values aren't their block's multiples, in three of the blocks.
        let rows: Buffer<u16> = [5u16, 1030, 2050, 2051, 3075].into_iter().collect();
        let patched: Buffer<i64> = [-1i64, 7, 9, -9, 11].into_iter().collect();
        let (rows, patched) = (rows.into_array(), patched.into_array());
        let patches = Patches::new(len, 0, rows, patched, None, &mut ctx).unwrap();
        let scaled = Scaled::try_new(quotients.into_array(), exponents, 2, Some(patches))
            .unwrap()
            .into_array();
        let mut want: Vec<i128> = (0..len as i128)
            .map(|i| (i % 7 - 3) * [8, 1, 32, 2][i as usize / BLOCK])
            .collect();
        for (row, value) in [(5, -1), (1030, 7), (2050, 9), (2051, -9), (3075, 11)] {
            want[row] = value;
        }
        assert_eq!(decoded(&scaled, &mut ctx), want);
        for (range, blocks) in [
            (1500..2060, 2 * BLOCK),
            (5..10, BLOCK),
            (1030..1040, BLOCK),
            (2 * BLOCK..3 * BLOCK, BLOCK),
            (3 * BLOCK + 2..len, 6),
            (3..len - 1, len),
        ] {
            // Slicing the patches reads them, so the slice kernel cuts the blocks; the reduce
            // rule leaves a Slice of the whole array.
            let lazy = scaled.slice(range.clone()).unwrap();
            let view = scaled.as_opt::<Scaled>().unwrap();
            let sliced = <Scaled as SliceKernel>::slice(view, range.clone(), &mut ctx)
                .unwrap()
                .unwrap_or_else(|| lazy.clone());
            // The Scaled array the slice reads: itself, or what a Slice is of.
            let read = match sliced.as_opt::<Slice>() {
                Some(_) => sliced.children()[0].clone(),
                None => sliced.clone(),
            };
            assert!(read.as_opt::<Scaled>().is_some(), "{range:?}");
            assert_eq!(read.len(), blocks, "rows read for {range:?}");
            assert_eq!(decoded(&sliced, &mut ctx), want[range.clone()]);
            assert_eq!(decoded(&lazy, &mut ctx), want[range]);
        }
    }

    /// Exponents that are a constant are kept as the one exponent, without an exponents child.
    #[test]
    fn a_constant_exponent_is_metadata() {
        let mut ctx = session().create_execution_ctx();
        let len = BLOCK + 6;
        let quotients = PrimitiveArray::from_iter((0..len as i64).map(|i| i - 3));
        let exponents = ConstantArray::new(Scalar::primitive(2u8, Nullability::NonNullable), 2);
        let scaled =
            Scaled::try_new(quotients.into_array(), exponents.into_array(), 10, None).unwrap();
        assert_eq!(scaled.exponent(), Some(2));
        assert!(scaled.exponents().is_none());
        assert_eq!(scaled.divisor(), Some(100));
        let want: Vec<i128> = (0..len as i128).map(|i| (i - 3) * 100).collect();
        assert_eq!(decoded(&scaled.into_array(), &mut ctx), want);
    }

    /// A base below 2, or one wider than the values' type, is refused.
    #[test]
    fn bases_that_fit() {
        let values = || PrimitiveArray::from_iter([1i8, 2]).into_array();
        assert!(Scaled::try_new_uniform(values(), 1, 1).is_err());
        assert!(Scaled::try_new_uniform(values(), 300, 1).is_err());
        assert!(Scaled::try_new_uniform(values(), 100, 1).is_ok());
    }
}
