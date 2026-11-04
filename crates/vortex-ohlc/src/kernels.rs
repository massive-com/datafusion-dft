//! Filter kernels for two of Vortex's blocked encodings, which `vortex-fastlanes` 0.89 can only
//! filter by decoding all their values: bit-packing with a bit width per 1024-value block
//! ([`BlockedBitPackedFilter`]), and frame of reference with a reference per block
//! ([`BlockedForFilter`]). A filter that keeps a few rows of a chunk, as a needle does, then
//! unpacks only the blocks that hold them.
//!
//! A session tries its kernels for a pair of encodings in the order they were registered until
//! one answers (vortex-array 0.89 `executor.rs:661`). Vortex's own answer for a single width or
//! reference and give up on blocks, so these, registered after them, take over only there.

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use num_traits::{AsPrimitive, WrappingAdd};
use vortex::array::arrays::filter::{filter_chunked, ChunkDecoder, FILTER_CHUNK_LEN};
use vortex::array::arrays::{Filter, PrimitiveArray};
use vortex::array::kernel::ExecuteParentKernel;
use vortex::array::matcher::Matcher;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, ArrayRef, ArrayVTable, ArrayView,
    ExecutionCtx, IntoArray,
};
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::{
    BitPacked, BitPackedArrayExt, BitWidthsView, FoR, FoRArrayExt, FoRArraySlotsExt, FoRReferences,
    FL_CHUNK_SIZE,
};
use vortex::error::{vortex_ensure, VortexResult};
use vortex::mask::{Mask, MaskValues};
use vortex::session::VortexSession;

const _: () = assert!(FL_CHUNK_SIZE == FILTER_CHUNK_LEN);

/// How often each kernel filtered an array, so tests can tell they ran.
#[cfg(test)]
static FILTERED: [std::sync::atomic::AtomicUsize; 2] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

#[cfg(test)]
fn count(kernel: usize) {
    FILTERED[kernel].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Register the kernels on `session`, after Vortex's own.
pub fn register(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Filter.id(), BitPacked, BlockedBitPackedFilter);
    kernels.register_execute_parent_kernel(Filter.id(), FoR, BlockedForFilter);
}

/// Filters a bit-packed array with a bit width per block, unpacking only the blocks that hold
/// selected values.
#[derive(Debug)]
pub struct BlockedBitPackedFilter;

impl ExecuteParentKernel<BitPacked> for BlockedBitPackedFilter {
    type Parent = Filter;

    fn execute_parent(
        &self,
        array: ArrayView<'_, BitPacked>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Blocked(offsets) = array.bit_widths() else {
            return Ok(None);
        };
        let mask = parent.filter_mask();
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        // FastLanes only unpacks unsigned types, so filter as unsigned and reinterpret the
        // buffer with the array's (possibly signed) type.
        let ptype = array.dtype().as_ptype();
        let offsets = block_offsets(array, offsets, ctx)?;
        let buffer = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            let blocks = PackedBlocks {
                packed: array.data().packed_slice::<U>(),
                offsets: &offsets,
            };
            filter_chunked(&blocks, usize::from(array.offset()), values).into_byte_buffer()
        });
        let validity = array.validity()?.filter(mask)?;
        let primitive = PrimitiveArray::from_byte_buffer(buffer, ptype, validity);
        #[cfg(test)]
        count(0);
        let patches = (array.patches())
            .map(|patches| patches.filter(mask, ctx))
            .transpose()?
            .flatten();
        Ok(Some(match patches {
            Some(patches) => primitive.patch(&patches, ctx)?.into_array(),
            None => primitive.into_array(),
        }))
    }
}

/// The byte boundaries of `array`'s blocks, as `u64`s from the first, checked as Vortex's own
/// decoder checks them (`vortex-fastlanes` 0.89 `validate_primitive_offsets`): within the packed
/// buffer, in order, and each block a whole number of 128-byte rows, one per bit of its width,
/// at most the type's width.
fn block_offsets(
    array: ArrayView<'_, BitPacked>,
    offsets: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<u64>> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let blocks = (array.len() + usize::from(array.offset())).div_ceil(FL_CHUNK_SIZE);
    vortex_ensure!(
        offsets.len() == blocks + 1,
        "expected {} block boundaries, got {}",
        blocks + 1,
        offsets.len()
    );
    let offsets: Vec<u64> = match_each_unsigned_integer_ptype!(offsets.ptype(), |I| {
        (offsets.as_slice::<I>().iter())
            .map(|&offset| AsPrimitive::<u64>::as_(offset))
            .collect()
    });
    let max_bit_width = array.dtype().as_ptype().bit_width() as u64;
    let packed_len = array.data().packed().len() as u64;
    let base = offsets[0];
    for pair in offsets.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        vortex_ensure!(
            base <= start && start <= end && end - base <= packed_len,
            "block boundaries {start} and {end} are outside the packed buffer"
        );
        let size = end - start;
        vortex_ensure!(
            (start - base).is_multiple_of(128)
                && size.is_multiple_of(128)
                && size / 128 <= max_bit_width,
            "block boundaries {start} and {end} hold no supported bit width"
        );
    }
    vortex_ensure!(
        offsets[blocks] - base == packed_len,
        "block boundaries don't span the packed buffer"
    );
    Ok(offsets)
}

/// The packed blocks of a bit-packed array, each at its own bit width, between byte `offsets`
/// checked by [`block_offsets`].
struct PackedBlocks<'a, T> {
    packed: &'a [T],
    offsets: &'a [u64],
}

impl<T> PackedBlocks<'_, T> {
    /// The bit width and packed values of block `block`.
    fn block(&self, block: usize) -> (usize, &[T]) {
        // `block_offsets` bounds these by the packed buffer's length.
        let start = (self.offsets[block] - self.offsets[0]) as usize;
        let end = (self.offsets[block + 1] - self.offsets[0]) as usize;
        let bit_width = (end - start) / 128;
        let packed = &self.packed[start / size_of::<T>()..end / size_of::<T>()];
        (bit_width, packed)
    }
}

// SAFETY: the FastLanes unpack kernels initialize every value of `dst`.
unsafe impl<T: BitPacking> ChunkDecoder<T> for PackedBlocks<'_, T> {
    fn decode_chunk(&self, block: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        let (bit_width, packed) = self.block(block);
        // SAFETY: the unpack only writes to `dst`, and the checked block holds `FL_CHUNK_SIZE`
        // values of `bit_width` bits.
        unsafe { BitPacking::unchecked_unpack(bit_width, packed, dst.assume_init_mut()) }
    }

    fn decode_indices(&self, block: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        let (bit_width, packed) = self.block(block);
        // SAFETY: every index is within the block, and `dst` holds one value for each.
        unsafe { BitPacking::unchecked_unpack_indices(bit_width, packed, indices, dst) }
    }
}

/// Filters a frame-of-reference array with a reference per block: filters the encoded values
/// (bit-packed ones through [`BlockedBitPackedFilter`] or Vortex's own kernel), then adds each
/// selected value's block reference.
#[derive(Debug)]
pub struct BlockedForFilter;

impl ExecuteParentKernel<FoR> for BlockedForFilter {
    type Parent = Filter;

    fn execute_parent(
        &self,
        array: ArrayView<'_, FoR>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let FoRReferences::Blocked(references) = array.references() else {
            return Ok(None);
        };
        let mask = parent.filter_mask();
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        let encoded = (array.encoded().filter(mask.clone())?).execute::<PrimitiveArray>(ctx)?;
        let references = references.clone().execute::<PrimitiveArray>(ctx)?;
        let offset = usize::from(array.offset());
        let validity = encoded.validity()?;
        let filtered = match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            let buffer = add_references::<T>(
                encoded.as_slice::<T>(),
                references.as_slice::<T>(),
                offset,
                values,
            );
            PrimitiveArray::new(buffer, validity)
        });
        #[cfg(test)]
        count(1);
        Ok(Some(filtered.into_array()))
    }
}

/// `encoded`, the values `mask` selects, each plus the reference of its block; the array's first
/// value is `offset` values into the first block.
fn add_references<T: NativePType + WrappingAdd>(
    encoded: &[T],
    references: &[T],
    offset: usize,
    mask: &MaskValues,
) -> Buffer<T> {
    (mask.indices().iter())
        .zip(encoded)
        .map(|(&index, value)| value.wrapping_add(&references[(offset + index) / FL_CHUNK_SIZE]))
        .collect()
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::array::{ArrayRef, IntoArray, VortexSessionExecute};
    use vortex::buffer::Buffer;
    use vortex::encodings::fastlanes::bitpack_compress::bitpack_blocked_to_best_bit_widths;
    use vortex::encodings::fastlanes::{BitPackedArrayExt, FoR, FoRArrayExt, FoRArraySlotsExt};
    use vortex::mask::Mask;
    use vortex::session::VortexSession;
    use vortex::VortexSessionDefault;

    /// Values in blocks of very different ranges, with a few outliers that become patches.
    fn values(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| {
                let block = (i / 1024) as u32;
                match i % 997 {
                    0 => 1 << 30,
                    _ => (i as u32 * 7919) % (1 << (2 + 3 * (block % 6))),
                }
            })
            .collect()
    }

    /// Masks a filter can get: a few rows, every other row, a run, and rows past the end of the
    /// first block only.
    fn masks(len: usize) -> Vec<Mask> {
        let sparse = (0..len).filter(|i| i % 211 == 3).collect::<Vec<_>>();
        let dense = (0..len).filter(|i| i % 2 == 0).collect::<Vec<_>>();
        let late = (0..len)
            .filter(|i| *i > 1500 && i % 5 == 0)
            .collect::<Vec<_>>();
        vec![
            Mask::from_indices(len, sparse),
            Mask::from_indices(len, dense),
            Mask::from_slices(len, vec![(100, 2100)]),
            Mask::from_indices(len, late),
        ]
    }

    /// What every mask keeps of `array` through the session's kernels, against decoding first.
    fn check(session: &VortexSession, array: &ArrayRef) {
        let mut ctx = session.create_execution_ctx();
        let decoded = (array.clone().execute::<PrimitiveArray>(&mut ctx)).unwrap();
        for mask in masks(array.len()) {
            let expected = decoded.clone().into_array().filter(mask.clone()).unwrap();
            let expected = expected.execute::<PrimitiveArray>(&mut ctx).unwrap();
            let actual = array.filter(mask).unwrap();
            let actual = actual.execute::<PrimitiveArray>(&mut ctx).unwrap();
            assert_eq!(actual.as_slice::<u32>(), expected.as_slice::<u32>());
            let validity = |array: &PrimitiveArray, ctx: &mut _| {
                let validity = array.validity().unwrap();
                validity.execute_mask(array.len(), ctx).unwrap()
            };
            assert_eq!(validity(&actual, &mut ctx), validity(&expected, &mut ctx));
        }
    }

    fn filtered(kernel: usize) -> usize {
        super::FILTERED[kernel].load(std::sync::atomic::Ordering::Relaxed)
    }

    fn session() -> VortexSession {
        let session = VortexSession::default();
        super::register(&session);
        session
    }

    #[test]
    fn blocked_bit_packing_filters_as_decoded() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let len = 5000;
        let validity = Validity::from_iter((0..len).map(|i| i % 13 != 0));
        let array = PrimitiveArray::new(Buffer::from_iter(values(len)), validity);
        let packed = bitpack_blocked_to_best_bit_widths(&array, &mut ctx).unwrap();
        assert!(!packed.bit_widths().is_global() && packed.patches().is_some());
        let packed = packed.into_array();
        let before = filtered(0);
        check(&session, &packed);
        // Sliced off a block boundary, so the array starts within its first block.
        check(&session, &packed.slice(700..4321).unwrap());
        assert!(filtered(0) > before, "the kernel never ran");
    }

    #[test]
    fn blocked_references_filter_as_decoded() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let len = 5000;
        let array = PrimitiveArray::new(Buffer::from_iter(values(len)), Validity::NonNullable);
        let chunked = FoR::encode_chunked(array, &mut ctx).unwrap();
        assert!(!chunked.references().is_global());
        // Its values bit-packed with a width per block too, as files hold them.
        let values = chunked
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx);
        let values = bitpack_blocked_to_best_bit_widths(&values.unwrap(), &mut ctx).unwrap();
        let references = chunked.blocked_references().unwrap().clone();
        let packed = FoR::try_new_chunked(values.into_array(), references, 0).unwrap();
        let before = [filtered(0), filtered(1)];
        for array in [chunked.into_array(), packed.into_array()] {
            check(&session, &array);
            check(&session, &array.slice(700..4321).unwrap());
        }
        assert!(filtered(1) > before[1], "the FoR kernel never ran");
        assert!(
            filtered(0) > before[0],
            "the bit-packing kernel never ran below it"
        );
    }
}
