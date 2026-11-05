//! `vortex_ohlc.scaled`: integers stored a block of [`BLOCK`] values at a time, divided by a power
//! of ten the block's values share.
//!
//! Decimal columns keep the trailing zeros of their scale: prices at a scale of 4 that move in
//! cents, volumes in round lots. How many differs by ticker, and so by block: a chunk holds many
//! tickers, a block mostly one. Each block is stored as its values divided by `10^exponent`, and
//! multiplied back when read. A block's exponent is the one that packs it smallest: a few values
//! that aren't its multiples (a sub-cent print among cents) are kept whole, as patches.
//!
//! - child `encoded`: the quotients, in the array's type and with its validity, compressed like
//!   any integers (blocked FoR and bit-packing suit them). A null or patched row holds its
//!   block's lowest quotient, so that it doesn't widen the block's range.
//! - child `exponents`: one `u8` per block.
//! - patches: the rows whose values aren't their block's multiples, with those values.
//!
//! [`Pow10Scheme`] lets Vortex's compressor pick it for a decimal's integer part.

use std::fmt::{Display, Formatter};
use std::hash::Hasher;

use num_traits::{PrimInt, WrappingMul};
use prost::Message;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::dtype::{DType, Nullability, PType};
use vortex::array::patches::{PatchSlotIndices, Patches, PatchesData, PatchesMetadata};
use vortex::array::scalar::Scalar;
use vortex::array::serde::ArrayChildren;
use vortex::array::smallvec::smallvec;
use vortex::array::vtable::{OperationsVTable, VTable, ValidityChild, ValidityVTableFromChild};
use vortex::array::{
    array_slots, match_each_integer_ptype, Array, ArrayEq, ArrayHash, ArrayId, ArrayParts,
    ArrayRef, ArraySlots, ArrayView, EqMode, ExecutionCtx, ExecutionResult, IntoArray,
    TypedArrayRef,
};
use vortex::error::{vortex_ensure, vortex_ensure_eq, vortex_panic, VortexResult};
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

mod encode;
mod filter;
mod scheme;

pub use encode::scaled_encode;
pub(crate) use scheme::is_decimal_values;
pub use scheme::Pow10Scheme;

/// Values per exponent: one FastLanes block, as Vortex's blocked FoR and bit-packing use.
pub const BLOCK: usize = 1024;

/// Register the encoding and its filter kernel on `session`, and its [`Pow10Scheme`] for the
/// compressor (once).
pub fn register(session: &VortexSession) {
    use vortex::array::arrays::filter::FilterExecuteAdaptor;
    use vortex::array::arrays::Filter;
    use vortex::array::optimizer::kernels::ArrayKernelsExt;
    use vortex::array::session::ArraySessionExt;
    session.arrays().register(Scaled);
    session.kernels().register_execute_parent_kernel(
        Filter.id(),
        Scaled,
        FilterExecuteAdaptor(Scaled),
    );
    crate::register_scheme(session, &Pow10Scheme);
}

/// A [`Scaled`]-encoded array.
pub type ScaledArray = Array<Scaled>;

#[derive(Clone, Debug)]
pub struct Scaled;

#[array_slots(Scaled)]
pub struct ScaledSlots {
    /// The values divided by their block's power of ten, with the array's validity.
    #[slot(0)]
    pub encoded: ArrayRef,
    /// One exponent (`u8`) per block of [`BLOCK`] values.
    #[slot(1)]
    pub exponents: ArrayRef,
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

/// What a [`ScaledArray`] keeps besides its slots: where its patches start.
#[derive(Clone, Debug)]
pub struct ScaledData {
    patches: Option<PatchesData>,
}

impl Display for ScaledData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.patches {
            Some(patches) => write!(f, "patch_offset: {}", patches.offset()),
            None => Ok(()),
        }
    }
}

impl ArrayHash for ScaledData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        std::hash::Hash::hash(&self.patches, state);
    }
}

impl ArrayEq for ScaledData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.patches == other.patches
    }
}

#[derive(Clone, prost::Message)]
struct ScaledMetadata {
    #[prost(message, optional, tag = "1")]
    patches: Option<PatchesMetadata>,
}

fn exponents_dtype() -> DType {
    DType::Primitive(PType::U8, Nullability::NonNullable)
}

fn slots(encoded: ArrayRef, exponents: ArrayRef, patches: Option<&Patches>) -> ArraySlots {
    let mut slots: ArraySlots = smallvec![Some(encoded), Some(exponents)];
    PatchesData::push_slots(&mut slots, patches);
    slots
}

impl Scaled {
    /// The array of `encoded[i] × 10^exponents[i / BLOCK]`, but where `patches` have a value.
    /// `encoded` gives the array its dtype and validity; `exponents` has one non-null `u8` per
    /// block of [`BLOCK`] values.
    pub fn try_new(
        encoded: ArrayRef,
        exponents: ArrayRef,
        patches: Option<Patches>,
    ) -> VortexResult<ScaledArray> {
        let (dtype, len) = (encoded.dtype().clone(), encoded.len());
        let data = ScaledData {
            patches: patches.as_ref().map(PatchesData::from_patches),
        };
        let slots = slots(encoded, exponents, patches.as_ref());
        Array::try_from_parts(ArrayParts::new(Scaled, dtype, len, data, slots))
    }
}

pub trait ScaledArrayExt: ScaledArraySlotsExt {
    /// The rows whose values aren't their block's multiples, with those values.
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches.as_ref(),
            self.len(),
            self.slots(),
            PATCH_SLOTS,
        )
    }
}

impl<T: TypedArrayRef<Scaled>> ScaledArrayExt for T {}

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
        if let Some(patches) =
            PatchesData::patches_from_slots(data.patches.as_ref(), len, slots, PATCH_SLOTS)
        {
            vortex_ensure_eq!(patches.array_len(), len, "Scaled patches' length");
            vortex_ensure_eq!(patches.dtype(), dtype, "Scaled patches' dtype");
        }
        vortex_ensure!(dtype.is_int(), "Scaled needs an integer dtype, got {dtype}");
        vortex_ensure_eq!(view.encoded.dtype(), dtype, "Scaled encoded dtype");
        vortex_ensure_eq!(view.encoded.len(), len, "Scaled encoded length");
        vortex_ensure_eq!(
            view.exponents.dtype(),
            &exponents_dtype(),
            "Scaled exponents"
        );
        vortex_ensure_eq!(
            view.exponents.len(),
            len.div_ceil(BLOCK),
            "Scaled needs an exponent per block"
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
        Ok(Some(ScaledMetadata { patches }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ScaledMetadata::decode(metadata)?;
        let encoded = children.get(0, dtype, len)?;
        let exponents = children.get(1, &exponents_dtype(), len.div_ceil(BLOCK))?;
        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(2, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(3, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(4, &dtype, usize::try_from(p.chunk_offsets_len())?))
                    .transpose()?;
                Patches::new(len, p.offset()?, indices, values, chunk_offsets)
            })
            .transpose()?;
        let data = ScaledData {
            patches: patches.as_ref().map(PatchesData::from_patches),
        };
        let slots = slots(encoded, exponents, patches.as_ref());
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

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let patches = array.patches();
        let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
        let exponents = array.exponents().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = encoded.validity()?;
        let decoded = match_each_integer_ptype!(encoded.ptype(), |T| {
            let mut values = encoded.into_buffer_mut::<T>();
            let blocks = values.as_mut_slice().chunks_mut(BLOCK);
            for (block, &exponent) in blocks.zip(exponents.as_slice::<u8>()) {
                if exponent != 0 {
                    let factor = pow10::<T>(exponent);
                    block.iter_mut().for_each(|v| *v = v.wrapping_mul(factor));
                }
            }
            PrimitiveArray::new(values.freeze(), validity)
        });
        let decoded = match patches {
            Some(patches) => decoded.patch(&patches, ctx)?,
            None => decoded,
        };
        Ok(ExecutionResult::done(decoded.into_array()))
    }
}

impl OperationsVTable<Scaled> for Scaled {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Scaled>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let encoded = array.encoded().execute_scalar(index, ctx)?;
        if encoded.is_null() {
            return Ok(encoded);
        }
        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index, ctx)? {
                return patch.cast(array.dtype());
            }
        }
        let exponent = array.exponents().execute_scalar(index / BLOCK, ctx)?;
        let exponent = exponent.as_primitive().typed_value::<u8>().unwrap_or(0);
        let nullability = array.dtype().nullability();
        Ok(match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            let value = encoded
                .as_primitive()
                .typed_value::<T>()
                .unwrap_or_default();
            Scalar::primitive(value.wrapping_mul(pow10::<T>(exponent)), nullability)
        }))
    }
}

impl ValidityChild<Scaled> for Scaled {
    fn validity_child(array: ArrayView<'_, Scaled>) -> ArrayRef {
        array.encoded().clone()
    }
}

/// `10^exponent` in `T`, wrapping where it doesn't fit (no encoder picks such an exponent).
pub(crate) fn pow10<T: PrimInt + WrappingMul>(exponent: u8) -> T {
    let ten = T::from(10).unwrap_or_else(T::one);
    (0..exponent).fold(T::one(), |acc, _| acc.wrapping_mul(&ten))
}
