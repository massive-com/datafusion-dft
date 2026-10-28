//! Market aggregates (OHLC bars) in Vortex, stored with nothing but Vortex's own encodings and
//! layouts. What is ours is how the columns are laid out before writing, how they are
//! compressed, and how a scan prunes on the prices:
//!
//! - [`plan`]: the transforms applied to columns before writing (decimals rescaled to drop the
//!   trailing zeros every value has, and the four prices stored as one struct of parts), and the
//!   record of them in the file, so a reader can undo them.
//! - [`profile`]: measuring a source's decimal columns, and the plan that follows from that.
//! - [`ohlc`]: a bar's parts from its prices, and back.
//! - [`prices`]: the prices as Vortex expressions over the parts, to rebuild or filter on them
//!   within a scan.
//! - [`compress`]: FoR + FastLanes for decimal and integer chunks, under DecimalByteParts.
//! - [`strategy`]: the write strategy that sends each column through it.
//! - [`datafusion`]: reading the files from DataFusion with the source's schema: the planned
//!   columns rebuilt, filters on them pushed down into Vortex, and scans pruned on the zone maps,
//!   the prices on their parts'.
//! - [`edition`]: the edition that lets a writer emit Vortex's blocked FoR and bit-packing.

pub mod compress;
pub mod datafusion;
pub mod edition;
pub mod ohlc;
pub mod plan;
pub mod prices;
pub mod profile;
pub mod strategy;
