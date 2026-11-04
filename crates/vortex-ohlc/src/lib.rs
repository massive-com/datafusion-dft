//! Market aggregates (OHLC bars) in Vortex, stored with Vortex's own encodings under a layout of
//! ours. A file has the source's dtype; how its columns are stored is up to the components
//! registered on a session ([`register`]):
//!
//! - [`bars`]: the layout at the root of a table of bars, which stores the four prices as parts
//!   and reads them back as prices ([`price_fn`]).
//! - [`ohlc`]: a bar's parts from its prices, and back.
//! - [`write`](mod@write): the options to write a table with.
//! - [`edition`]: the edition that lets a writer emit Vortex's blocked FoR and bit-packing.
//! - [`kernels`]: filters on those blocked encodings that unpack only the blocks they keep.
//! - [`pow10`]: integers divided, a block at a time, by a power of ten the block shares.
//!
//! Reading the files from DataFusion is datafusion-vortex-ohlc's, and pruning scans on the zone
//! maps before they start (the prices on their parts') is datafusion-vortex-prune's.

use vortex::session::VortexSession;

pub mod bars;
pub mod edition;
pub mod kernels;
pub mod ohlc;
pub mod pow10;
pub mod price_fn;
pub mod write;

/// Register what a session needs to read vortex-ohlc's files, and to read them well: the
/// [`bars`] layout and its price function, the [`pow10`] encoding, the filter [`kernels`], and
/// for writing, the compression scheme that picks [`pow10`].
pub fn register(session: &VortexSession) {
    bars::register(session);
    kernels::register(session);
    pow10::register(session);
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
