//! Market aggregates (OHLC bars) in Vortex, stored with nothing but Vortex's own encodings and
//! layouts. What is ours is how the columns are laid out before writing, which of Vortex's
//! encodings may compress them ([`edition`]), and how they are read back:
//!
//! - [`plan`]: the transforms applied to columns before writing (decimals rescaled to drop the
//!   trailing zeros every value has, and the four prices stored as one struct of parts).
//! - [`profile`]: measuring a source's decimal columns, and the plan that follows from that.
//! - [`ohlc`]: a bar's parts from its prices, and back.
//! - [`prices`]: the prices as Vortex expressions over the parts, to rebuild or filter on them
//!   within a scan.
//! - [`write`](mod@write): writing a table with a plan.
//! - [`edition`]: the edition that lets a writer emit Vortex's blocked FoR and bit-packing.
//!
//! Reading the files from DataFusion, the prices rebuilt, is datafusion-vortex-ohlc's, and
//! pruning scans on the zone maps (the prices on their parts') is datafusion-vortex-prune's.

pub mod edition;
pub mod ohlc;
pub mod plan;
pub mod prices;
pub mod profile;
pub mod write;
