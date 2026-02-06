use crate::reasoning::{Reasoner, construct_triple, evaluate_filters};

use shared::{rule::Rule, triple::Triple};
use std::collections::{BTreeMap, HashMap, HashSet};

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

    pub fn dred_maintenance(&mut self, to_add: Vec<Triple>, to_remove: Vec<Triple>) {
        // convert the input to sets
        let to_add: HashSet<Triple> = to_add.into_iter().collect();
        // let to_remove: HashSet<Triple> = to_remove.into_iter().collect();

        let all_facts: Vec<Triple> = self.reasoner.index_manager.query(None, None, None);

        let deleted: HashSet<Triple> = self.overdelete(&to_remove, &all_facts);
        let rederived: HashSet<Triple> = self.rederive(&deleted, to_remove, &all_facts);
    }

    // Returns which facts should be deleted
    fn overdelete(&mut self, to_remove: &Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple> {
        // initialize n_d as the intersection between the explicitly removed facts and the facts in the index manager
        let mut n_d: HashSet<Triple> = to_remove.iter()
                .filter(|triple| all_facts.contains(*triple))
                .cloned()
                .collect();

        let mut delta: Vec<Triple>;
        let mut deleted: HashSet<Triple> = HashSet::new();

        loop {
            // calculate the unprocessed changes
            delta = n_d.iter()
                    .filter(|triple| !deleted.contains(triple))
                    .cloned()
                    .collect();

            // if there are no more changes, stop
            if delta.is_empty() {
                break;
            }

            let mut bindings: Vec<HashMap<String, u32>>;
            let mut inferred: Triple;

            // clear the contents of the previous iteration
            n_d.clear();

            for rule in self.reasoner.rules.iter() {
                bindings = self.reasoner.evaluate_rule_with_delta_improved(rule, all_facts, &delta);

                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary) {

                        for conclusion in &rule.conclusion {
                            inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary);
                            n_d.insert(inferred);
                        }
                    }
                }
            }

            deleted.extend(delta);
        }

        // return the deleted triples
        deleted
    }

    fn rederive(&mut self, deleted: &HashSet<Triple>, to_remove: Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple>{
        // if an explicitly deleted fact is still in the explicit facts, it needs to be rederived if it was deleted
        let mut rederived: HashSet<Triple> = self.explicit.iter()
                .filter(|triple| !to_remove.contains(*triple))
                .filter(|triple| deleted.contains(*triple))
                .cloned()
                .collect();

        // calculate the difference between all_facts and deleted
        let difference: Vec<Triple> = all_facts.iter()
                .filter(|triple| !deleted.contains(*triple))
                .cloned()
                .collect();

        let mut bindings: Vec<HashMap<String, u32>>;

        // evaluate the rules with the remaining facts
        for rule in self.reasoner.rules.clone().iter() {
            bindings = self.evaluate_rule(rule, &difference);

            // construct triples from these bindings, but only keep those that were deleted by the overdeletion step
            let constructed_triples: HashSet<Triple> = self.construct_triple_from_bindings(rule, bindings).into_iter()
                    .filter(|triple| deleted.contains(triple))
                    .collect();
            rederived.extend(constructed_triples);
        }

        // return the facts that should be rederived
        rederived
    }

    fn insert(&mut self) {todo!()}

    fn evaluate_rule(&self, rule: &Rule, all_facts: &Vec<Triple>) -> Vec<HashMap<String, u32>> {
        let n = rule.premise.len();
        let mut results = Vec::new();

        for i in 0..n {
            let mut current_bindings = vec![BTreeMap::new()];

            current_bindings = self.reasoner.join_premise_with_hash_join(&rule.premise[i], all_facts, current_bindings);

            // Convert and add results
            for binding in current_bindings {
                let u32_binding = self.reasoner.convert_string_binding_to_u32(&binding);
                results.push(u32_binding);
            }
        }

        results
    }

    fn construct_triple_from_bindings(&mut self, rule: &Rule, bindings: Vec<HashMap<String, u32>>) -> HashSet<Triple> {
        let mut result: HashSet<Triple> = HashSet::new();
        let mut inferred: Triple;

        for binding in bindings {
                // check if the binding adheres to the filters of the rule
                if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary) {

                    for conclusion in &rule.conclusion {
                        inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary);
                        result.insert(inferred);
                    }
                }
            }

        result
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
