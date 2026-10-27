//! The edition a writer needs for what [`crate::strategy`] writes, beyond Vortex's `core`
//! editions:
//!
//! - our layout at the root of a table of bars, [`crate::layout`];
//! - three of Vortex's own wire formats, new in Vortex 0.89 and in no Vortex edition yet:
//!   blocked FoR and bit-packing (a reference and a bit width per 1024 values), and
//!   DecimalByteParts with lower parts (for chunks wider than an i64).
//!
//! A writer only emits them when an edition it has enabled lists them. Enabling
//! [`VORTEX_OHLC_2026_10`] (see [`enable`]) keeps Vortex's edition checks for everything else,
//! rather than turning them off.

use vortex::editions::{
    Edition, EditionDeclaration, EditionFamily, EditionId, EditionMember, EditionSessionExt,
};
use vortex::error::VortexResult;
use vortex::session::VortexSession;

pub static FAMILY: EditionFamily = EditionFamily {
    name: "vortex_ohlc",
    origin: "vortex-ohlc",
    doc: "The components vortex-ohlc writes OHLC bars with that no Vortex edition lists: its bars \
layout, and Vortex's own components that are in no edition yet. A draft: nothing is frozen.",
};

/// The first edition, a draft.
pub const VORTEX_OHLC_2026_10: EditionId = EditionId::new("vortex_ohlc", 2026, 10, 0);

pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: VORTEX_OHLC_2026_10,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"fastlanes.for.v2"),
        EditionMember::array(&"fastlanes.bitpacked.v2"),
        EditionMember::array(&"vortex.decimal_byte_parts.v2"),
        EditionMember::layout(&"vortex_ohlc.bars"),
    ],
};

/// Declare [`FAMILY`] and [`VORTEX_OHLC_2026_10`] on `session`, and enable it for writing next to
/// whichever `core` edition is enabled.
pub fn enable(session: &VortexSession) -> VortexResult<()> {
    session.editions().declare_family(&FAMILY)?;
    session.register_edition(&DECLARATION)?;
    session.enable_edition(VORTEX_OHLC_2026_10)
}
