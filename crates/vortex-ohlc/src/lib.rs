//! Market aggregates (OHLC bars) in Vortex, stored with nothing but Vortex's own encodings and
//! layouts and a few of ours. A file has the source's dtype; how its columns are stored is up to
//! the components registered on a session ([`register`]):
//!
//! - [`bars`]: the layout at the root of a table of bars, which stores the four prices as parts
//!   and reads them back as prices ([`price_fn`]).
//! - [`ohlc`]: a bar's parts from its prices, and back.
//! - [`product_residual`]: a column from its residual against the product of two others, which
//!   the layout stores in its place for a bar's dollar volume, against volume × open.
//! - [`write`](mod@write): the options to write a table with.
//! - [`edition`]: the edition that lets a writer emit what follows, and Vortex's blocked FoR and
//!   bit-packing.
//! - [`kernels`]: filters on those blocked encodings that unpack only the blocks they keep.
//! - [`scaled`]: integers and decimals divided by powers of a base, a block or a whole array at a
//!   time: a chunk of decimals by the trailing zeros all its values have, at a smaller scale, and
//!   each block of its integer part by those the block's values have.
//! - [`blocked_for`]: a reference and a bit width per block for decimals, where that pays.
//! - [`gcd`]: timestamps' seconds into the day divided by the greatest common divisor of a
//!   chunk's (minute bars' are multiples of 60), as [`scaled`] arrays.
//!
//! Reading the files from DataFusion is datafusion-vortex-ohlc's, and pruning scans on the zone
//! maps before they start (the prices on their parts') is datafusion-vortex-prune's.

use vortex::session::VortexSession;

pub mod bars;
pub mod blocked_for;
pub mod datetime;
mod decimal;
pub mod edition;
pub mod gcd;
pub mod kernels;
pub mod ohlc;
pub mod price_fn;
pub mod product_residual;
pub mod scaled;
pub mod write;

#[cfg(test)]
mod testing;

/// Register what a session needs to read vortex-ohlc's files, and to read them well: the
/// [`bars`] layout and its price function, the [`scaled`] encoding, the filter [`kernels`] and
/// timestamps' [`datetime`] kernel, and
/// for writing, the compression schemes that pick [`scaled`] (with [`gcd`]'s) and
/// [`blocked_for`].
/// [`DecimalPow10Scheme`](scaled::DecimalPow10Scheme) takes the place of Vortex's decimal scheme,
/// so it is [`write::write_options`]' alone.
pub fn register(session: &VortexSession) {
    bars::register(session);
    kernels::register(session);
    datetime::register(session);
    scaled::register(session);
    register_scheme(session, &gcd::GcdScheme);
    register_scheme(session, &blocked_for::BlockedForScheme);
}

/// Let compressors built from `session` use `scheme`, unless they already may.
fn register_scheme(session: &VortexSession, scheme: &'static dyn vortex::compressor::Scheme) {
    use vortex::compressor::CompressionSessionExt;
    use vortex_btrblocks::SchemeExt;
    let registered = session
        .compression()
        .schemes()
        .iter()
        .any(|s| s.id() == scheme.id());
    if !registered {
        session.register_scheme(scheme);
    }
}
