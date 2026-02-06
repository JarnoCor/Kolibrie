use crate::reasoning::{Reasoner, construct_triple, evaluate_filters};

use shared::{rule::Rule, triple::Triple};
use std::collections::{HashMap, HashSet};

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

    pub fn dred_maintenance(&mut self, to_add: Vec<Triple>, to_delete: Vec<Triple>) {
        let all_facts: Vec<Triple> = self.reasoner.index_manager.query(None, None, None);

        let deleted: HashSet<Triple> = self.overdelete(&to_delete, &all_facts);
        let rederived: HashSet<Triple> = self.rederive(&deleted, &to_delete, &all_facts);
        let added: HashSet<Triple> = self.insert(&deleted, &rederived, &to_add, &all_facts);

        // update the explicit facts
        self.explicit = self.explicit.iter()
                .filter(|triple| !to_delete.contains(*triple))
                .cloned()
                .collect();
        
        self.explicit.extend(to_add.clone());

        // update the index manager
        for triple in deleted {
            if !added.contains(&triple) {
                self.reasoner.index_manager.delete(&triple);
            }
        }
        for triple in added {
            self.reasoner.index_manager.insert(&triple);
        }
    }

    /// Returns the explicitly deleted facts and the facts derived by them.
    fn overdelete(&mut self, to_remove: &Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple> {
        // initialize n_d as the intersection between the explicitly removed facts and the facts in the index manager
        let mut n_d: HashSet<Triple> = to_remove.iter()
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

            // clear the contents of the previous iteration
            n_d.clear();

            // calculate the difference between all_facts and deleted
            let difference: Vec<Triple> = all_facts.iter()
                    .filter(|triple| !deleted.contains(*triple))
                    .cloned()
                    .collect();

            let mut bindings: Vec<HashMap<String, u32>>;
            for rule in self.reasoner.rules.clone().iter() {
                bindings = self.reasoner.evaluate_rule_with_delta_improved(rule, &difference, &delta);
                n_d.extend(self.construct_triple_from_bindings(rule, bindings));
            }

            deleted.extend(delta);
        }

        // return the deleted triples
        deleted
    }

    /// Returns the facts that should not have been deleted.
    pub fn rederive(&mut self, deleted: &HashSet<Triple>, to_delete: &Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple>{
        // if an explicitly deleted fact is still in the explicit facts, it needs to be rederived if it was deleted
        let mut rederived: HashSet<Triple> = self.explicit.iter()
                .filter(|triple| !to_delete.contains(*triple))
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
            bindings = self.reasoner.evaluate_rule_with_optimized_join(rule, &difference);

            // construct triples from these bindings, but only keep those that were deleted by the overdeletion step
            let constructed_triples: HashSet<Triple> = self.construct_triple_from_bindings(rule, bindings).into_iter()
                    .filter(|triple| deleted.contains(triple))
                    .collect();
            rederived.extend(constructed_triples);
        }

        // return the facts that should be rederived
        rederived
    }

    /// Returns the facts that should be (re)inserted.
    fn insert(&mut self, deleted: &HashSet<Triple>, rederived: &HashSet<Triple>, to_add: &Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple> {
        // initialize n_a as the intersection between the explicitly added facts and the facts in the index manager
        let mut n_a: HashSet<Triple> = to_add.iter()
                .cloned()
                .collect();

        n_a.extend(rederived.clone());

        let mut delta: Vec<Triple>;
        let mut added: HashSet<Triple> = HashSet::new();

        loop {
            // calculate the difference between all_facts and deleted, and do the union of added
            let difference: Vec<Triple> = all_facts.iter()
            .filter(|triple| !deleted.contains(*triple))
            .chain(added.iter())
            .cloned()
            .collect();

            // calculate the unprocessed changes
            delta = n_a.iter()
                    .filter(|triple| !difference.contains(*triple))
                    .cloned()
                    .collect();

            // if there are no more changes, stop
            if delta.is_empty() {
                break;
            }


            // clear the contents of the previous iteration
            n_a.clear();

            let mut bindings: Vec<HashMap<String, u32>>;
            for rule in self.reasoner.rules.clone().iter() {
                bindings = self.reasoner.evaluate_rule_with_delta_improved(rule, &difference, &delta);
                n_a.extend(self.construct_triple_from_bindings(rule, bindings));
            }

            // add the delta to added
            added.extend(delta);
        }

        // return the added triples
        added
    }

    // fn evaluate_rule(&self, rule: &Rule, all_facts: &Vec<Triple>) -> Vec<HashMap<String, u32>> {
    //     let n = rule.premise.len();
    //     let mut results = Vec::new();

    //     for i in 0..n {
    //         let mut current_bindings = vec![BTreeMap::new()];

    //         current_bindings = self.reasoner.join_premise_with_hash_join(&rule.premise[i], all_facts, current_bindings);

    //         // Convert and add results
    //         for binding in current_bindings {
    //             let u32_binding = self.reasoner.convert_string_binding_to_u32(&binding);
    //             results.push(u32_binding);
    //         }
    //     }

    //     results
    // }

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

    pub fn decode_triple(&self, triple: &Triple) -> String {
        let subject = self.reasoner.dictionary.decode(triple.subject).unwrap_or("unknown");
        let predicate = self.reasoner.dictionary.decode(triple.predicate).unwrap_or("unknown");
        let object = self.reasoner.dictionary.decode(triple.object).unwrap_or("unknown");

        format!("{} {} {}", subject, predicate, object)
    }
}









#[cfg(test)]
mod tests {
    use super::*;
    use shared::terms::Term;
    use shared::rule::Rule;
    use std::time::Instant;

    fn vec_equal<T: PartialEq>(vec1: &Vec<T>, vec2: &Vec<T>) -> bool {
        if vec1.len() != vec2.len() {
            return false;
        }

        for item in vec1 {
            if !vec2.contains(item) {
                return false;
            }
        }

        for item in vec2 {
            if !vec1.contains(item) {
                return false;
            }
        }

        true
    }

    #[test]
    fn vec_equal_test() {
        let vec1 = vec![
            Triple {
                subject: 1,
                predicate: 2,
                object: 3,
            },
            Triple {
                subject: 5,
                predicate: 7,
                object: 9,
            },
        ];

        let vec2 = vec![
            Triple {
                subject: 5,
                predicate: 7,
                object: 9,
            },
            Triple {
                subject: 1,
                predicate: 2,
                object: 3,
            },
        ];

        assert!(vec_equal(&vec1, &vec2));
    }

    #[test]
    fn vec_not_equal_test() {
        let vec1 = vec![
            Triple {
                subject: 1,
                predicate: 2,
                object: 3,
            },
            Triple {
                subject: 5,
                predicate: 7,
                object: 1,
            },
        ];

        let vec2 = vec![
            Triple {
                subject: 5,
                predicate: 7,
                object: 9,
            },
            Triple {
                subject: 1,
                predicate: 2,
                object: 3,
            },
        ];

        assert!(!vec_equal(&vec1, &vec2));
    }

    #[test]
    fn new_test() {
        let dred = DRedMaintenance::new();
        let facts = dred.reasoner.index_manager.query(None, None, None);

        assert!(facts.is_empty());
    }

    #[test]
    fn maintenance_counting_graph_test() {
        let mut dred = DRedMaintenance::new();

        dred.add_abox_triple("a", "edge", "b");
        dred.add_abox_triple("b", "edge", "c");
        dred.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(dred.reasoner.dictionary.encode("edge"));
        let reachable = Term::Constant(dred.reasoner.dictionary.encode("reachable"));

        dred.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                edge,
                Term::Variable("y".to_string()),
            ),
            (
                Term::Variable("y".to_string()),
                reachable.clone(),
                Term::Variable("z".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                reachable,
                Term::Variable("z".to_string()),
            )],
            filters: vec![],
        });

        dred.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(dred.reasoner.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(dred.reasoner.clone().dictionary.encode("reachable")),
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("c"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("e"),
            },
        ];

        dred.reasoner.infer_new_facts_semi_naive();

        let start = Instant::now();
        dred.dred_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = dred.reasoner.index_manager.query(None, None, None);

        let test_materialisation = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("e"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("e"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("e"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("e"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("reachable"),
                object: dred.reasoner.dictionary.encode("e"),
            },
        ];

        let test_explicit = HashSet::from([
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("e"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("edge"),
                object: dred.reasoner.dictionary.encode("e"),
            },
        ]);

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, dred.explicit);
    }

    #[test]
    fn dred_maintenance_small_test() {
        let mut dred = DRedMaintenance::new();

        dred.add_abox_triple("a", "t", "b");
        dred.add_abox_triple("f", "t", "b");
        dred.add_abox_triple("c", "t", "d");

        let encoded_r = Term::Constant(dred.reasoner.dictionary.encode("r"));
        let encoded_t = Term::Constant(dred.reasoner.dictionary.encode("t"));

        dred.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_t,
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("y".to_string()),
                encoded_r,
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("f"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("g"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("d"),
            }
        ];

        dred.reasoner.infer_new_facts_semi_naive();

        let start = Instant::now();
        dred.dred_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = dred.reasoner.index_manager.query(None, None, None);

        let test_materialisation = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("c"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("g"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("b"),
                predicate: dred.reasoner.dictionary.encode("r"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("d"),
                predicate: dred.reasoner.dictionary.encode("r"),
                object: dred.reasoner.dictionary.encode("d"),
            },
        ];

        let test_explicit = HashSet::from([
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("c"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("d"),
            },
            Triple {
                subject: dred.reasoner.dictionary.encode("g"),
                predicate: dred.reasoner.dictionary.encode("t"),
                object: dred.reasoner.dictionary.encode("d"),
            },
        ]);

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, dred.explicit);
    }

    #[test]
    fn dred_maintenance_reflexive_test() {
        let mut dred = DRedMaintenance::new();

        dred.add_abox_triple("a", "r", "b");

        let encoded_r = Term::Constant(dred.reasoner.dictionary.encode("r"));

        dred.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_r.clone(),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("y".to_string()),
                encoded_r,
                Term::Variable("x".to_string()),
            )],
            filters: vec![],
        });

        dred.reasoner.infer_new_facts_semi_naive();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dred.reasoner.dictionary.encode("a"),
                predicate: dred.reasoner.dictionary.encode("r"),
                object: dred.reasoner.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![];


        let start = Instant::now();
        dred.dred_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = dred.reasoner.index_manager.query(None, None, None);
        let test_materialisation = vec![];

        let test_explicit = HashSet::new();

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, dred.explicit);
    }
}