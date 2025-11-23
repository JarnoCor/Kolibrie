use crate::knowledge_graph::KnowledgeGraph;
use shared::rule::Rule;
use shared::terms::Term;
use shared::triple::Triple;
use std::collections::{BTreeMap, HashMap, HashSet};

/// A wrapper struct to keep track of the trace.
/// Contains methods for counting incremental view maintenance.
pub struct CountingMaintenanceGraph {
    /// The knowledge graph to perform incremental view maintenance on
    pub kg: KnowledgeGraph,
    /// The trace
    pub counts: HashMap<Triple, u32>,
}
