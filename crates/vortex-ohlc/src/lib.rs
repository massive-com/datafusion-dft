//! Market aggregates (OHLC bars) in Vortex, stored with nothing but Vortex's own encodings and
//! layouts. What is ours is how the columns are laid out before writing, and how they are
//! compressed:
//!
//! - [`plan`]: the transforms applied to columns before writing (decimals rescaled to drop the
//!   trailing zeros every value has, and the four prices stored as one struct of parts), and the
//!   record of them in the file, so a reader can undo them.
//! - [`profile`]: measuring a source's decimal columns, and the plan that follows from that.
//! - [`ohlc`]: a bar's parts from its prices, and back.
//! - [`compress`]: FoR + FastLanes for decimal and integer chunks, under DecimalByteParts.
//! - [`strategy`]: the write strategy that sends each column through it.
//! - [`edition`]: the edition that lets a writer emit Vortex's blocked FoR and bit-packing.

pub mod compress;
pub mod edition;
pub mod ohlc;
pub mod plan;
pub mod profile;
pub mod strategy;
