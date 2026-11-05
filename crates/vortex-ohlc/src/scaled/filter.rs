//! Filtering a [`Scaled`] array without decoding all of it: its quotients are filtered by their
//! own encoding's kernels (blocked FoR and bit-packing unpack only the blocks that hold kept rows,
//! see [`crate::kernels`]), then each kept value is multiplied by its block's power of ten, and
//! the patches it keeps are applied.

use vortex::array::arrays::filter::FilterKernel;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::{match_each_integer_ptype, ArrayRef, ArrayView, ExecutionCtx, IntoArray};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use super::{pow10, Scaled, ScaledArrayExt, ScaledArraySlotsExt, BLOCK};

/// Rows of a filter that ran this kernel, so tests can tell it did.
#[cfg(test)]
pub(super) static FILTERED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl FilterKernel for Scaled {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Mask::Values(selected) = mask else {
            return Ok(None);
        };
        let quotients = array.encoded().filter(mask.clone())?;
        let quotients = quotients.execute::<PrimitiveArray>(ctx)?;
        let exponents = array.exponents().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = quotients.validity()?;
        let filtered = match_each_integer_ptype!(quotients.ptype(), |T| {
            let factors: Vec<T> = (exponents.as_slice::<u8>().iter())
                .map(|&exponent| pow10::<T>(exponent))
                .collect();
            let mut values = quotients.into_buffer_mut::<T>();
            for (value, &row) in values.as_mut_slice().iter_mut().zip(selected.indices()) {
                *value = value.wrapping_mul(factors[row / BLOCK]);
            }
            PrimitiveArray::new(values.freeze(), validity)
        });
        #[cfg(test)]
        FILTERED.fetch_add(selected.true_count(), std::sync::atomic::Ordering::Relaxed);
        let patches = (array.patches())
            .map(|patches| patches.filter(mask, ctx))
            .transpose()?
            .flatten();
        Ok(Some(match patches {
            Some(patches) => filtered.patch(&patches, ctx)?.into_array(),
            None => filtered.into_array(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use vortex::array::arrays::DecimalArray;
    use vortex::array::validity::Validity;
    use vortex::array::{IntoArray, VortexSessionExecute};
    use vortex::buffer::Buffer;
    use vortex::compressor::BtrBlocksCompressor;
    use vortex::dtype::DecimalDType;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;
    use vortex::VortexSessionDefault;

    use super::FILTERED;

    /// Prices in cents at a scale of 4, block by block around very different levels, with a
    /// sub-cent price now and then (Scaled's patches) and a few far outliers (the bit-packing's).
    fn prices(len: usize) -> Vec<i64> {
        (0..len as i64)
            .map(|i| {
                let level = [1_000_000, 50_000, 4_000_000_000, 700][(i / 1024 % 4) as usize];
                match i % 101 {
                    0 => level + i * 13 + 7,
                    50 if i % 3 == 0 => level * 1000,
                    _ => level + 100 * (i * 7919 % 300),
                }
            })
            .collect()
    }

    /// A filter on a compressed decimal column reaches Scaled's kernel through its integer part,
    /// and keeps what the values do: a few rows of some blocks, whole blocks, and the short last
    /// block.
    #[test]
    fn filter_matches_values() {
        let session = VortexSession::default();
        crate::edition::enable(&session).unwrap();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let values = prices(5000);
        let decimal = DecimalArray::new(
            Buffer::from(values.clone()),
            DecimalDType::new(18, 4),
            Validity::NonNullable,
        );
        let compressed = BtrBlocksCompressor::from_session(&session)
            .compress(&decimal.into_array(), &mut ctx)
            .unwrap();
        let rows: Vec<usize> = [0, 1, 50, 101, 1100]
            .into_iter()
            .chain(2048..3072)
            .chain((3072..4096).step_by(7))
            .chain([4999])
            .collect();
        let mask = Mask::from_indices(values.len(), rows.iter().copied());
        let before = FILTERED.load(Ordering::Relaxed);
        // As a scan filters: optimized, so that DecimalByteParts hands it to its integer part.
        let filtered = compressed
            .filter(mask)
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();
        assert!(FILTERED.load(Ordering::Relaxed) > before, "the kernel ran");
        let want: Vec<i64> = rows.iter().map(|&row| values[row]).collect();
        assert_eq!(filtered.buffer::<i64>().to_vec(), want);
    }
}
