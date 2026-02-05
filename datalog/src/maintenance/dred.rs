use crate::reasoning::Reasoner;

use shared::triple::Triple;
use std::collections::HashSet;

/// A struct for incremental view maintenance (IVM) using the Delete and Rederive approach.
///
/// Contains the [`Reasoner`] to perform IVM on, as well as the explicit facts.
pub struct DRedMaintenance {
    /// The [`Reasoner`] containing the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) to perform IVM on.
    pub reasoner: Reasoner,
    /// The explicit facts.
    pub explicit: HashSet<Triple>,
}

impl DRedMaintenance {
    /// Creates a new [`DRedMaintenance`] with no explicit facts from an existing [`Reasoner`].
    pub fn new() -> DRedMaintenance {
        DRedMaintenance { reasoner: Reasoner::new(), explicit: HashSet::new() }
    }

    /// Add an ABox triple (instance-level information).
    pub fn add_abox_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let s = self.reasoner.dictionary.encode(subject);
        let p = self.reasoner.dictionary.encode(predicate);
        let o = self.reasoner.dictionary.encode(object);

        let triple = Triple { subject: s, predicate: p, object: o };

        self.reasoner.index_manager.insert(&triple);
        self.explicit.insert(triple);
    }
}









#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let dred = DRedMaintenance::new();
        let facts = dred.reasoner.index_manager.query(None, None, None);

        assert!(facts.is_empty());
    }
}
