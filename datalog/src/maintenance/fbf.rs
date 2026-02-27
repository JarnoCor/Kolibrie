use crate::reasoning::{Reasoner, construct_triple, evaluate_filters};

use shared::{rule::Rule, triple::Triple};
use std::collections::{HashMap, HashSet};

/// A struct for incremental view maintenance (IVM) using the Forward/Backward/Forward approach.
///
/// Contains the [`Reasoner`] to perform IVM on, as well as the explicit facts.
pub struct FBFMaintenance {
    /// The [`Reasoner`] containing the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) to perform IVM on.
    pub reasoner: Reasoner,
    /// The explicit facts.
    pub explicit: HashSet<Triple>,
}
