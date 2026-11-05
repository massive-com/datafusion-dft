//! What the crate's tests share.

use vortex::array::ArrayRef;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

/// A session that reads and writes as [`crate::write::write_options`]' does: the edition
/// enabled, and everything of ours registered.
pub(crate) fn session() -> VortexSession {
    let session = VortexSession::default();
    crate::edition::enable(&session).unwrap();
    crate::register(&session);
    session
}

/// The encoding IDs of `array` and everything under it.
pub(crate) fn encoding_ids(array: &ArrayRef) -> Vec<String> {
    let mut ids = vec![array.encoding_id().to_string()];
    for child in array.children() {
        ids.extend(encoding_ids(&child));
    }
    ids
}
