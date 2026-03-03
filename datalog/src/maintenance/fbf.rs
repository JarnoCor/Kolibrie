use crate::reasoning::{Reasoner, construct_triple, evaluate_filters};

use shared::{rule::{FilterCondition, Rule}, terms::TriplePattern, triple::Triple};
use std::collections::{HashMap, HashSet};

/// A struct for incremental view maintenance (IVM) using the Forward/Backward/Forward approach.
///
/// Contains the [`Reasoner`] to perform IVM on, as well as the explicit facts.
pub struct FBFMaintenance {
    /// The [`Reasoner`] containing the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) to perform IVM on.
    pub reasoner: Reasoner,
    /// The explicit facts.
    pub explicit: HashSet<Triple>,
    /// The maximum amount of backward steps.
    steps: u32,

}

impl FBFMaintenance {
    /// Creates a new [`FBFMaintenance`] with no explicit facts and an empty [`Reasoner`], and the specified amount of maximum backward steps
    pub fn new(steps: u32) -> FBFMaintenance {
        FBFMaintenance { reasoner: Reasoner::new(), explicit: HashSet::new(), steps: steps }
    }

    /// Add an ABox triple (instance-level information).
    pub fn add_abox_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let mut dictionary = self.reasoner.dictionary.write().unwrap();

        let s = dictionary.encode(subject);
        let p = dictionary.encode(predicate);
        let o = dictionary.encode(object);

        let triple = Triple { subject: s, predicate: p, object: o };

        self.reasoner.index_manager.insert(&triple);
        self.explicit.insert(triple);
    }

    /// Performs IVM to update the facts in the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) of the [`Reasoner`] by adding and removing explicit and implicit facts using the Forward/Backward/Forward approach.
    ///
    /// # Examples
    ///
    /// ```
    /// use shared::{rule::Rule, terms::Term, triple::Triple};
    /// use datalog::maintenance::FBFMaintenance;
    ///
    /// let mut fbf = FBFMaintenance::new(3);
    /// fbf.reasoner.add_abox_triple("a", "knows", "b");
    ///
    /// let dictionary = &fbf.reasoner.dictionary.clone();
    /// let mut dictionary = dictionary.write().unwrap();
    ///
    /// fbf.reasoner.add_rule(Rule {
    ///     premise: vec![
    ///     (
    ///         Term::Variable("x".to_string()),
    ///         Term::Constant(dictionary.encode("knows")),
    ///         Term::Variable("y".to_string()),
    ///     ),
    ///     ],
    ///     conclusion: vec![(
    ///         Term::Variable("y".to_string()),
    ///         Term::Constant(dictionary.encode("knows")),
    ///         Term::Variable("x".to_string()),
    ///     )],
    ///     filters: vec![],
    /// });
    ///
    /// let to_add: Vec<Triple> = Vec::new();
    /// let to_remove: Vec<Triple> = vec![
    ///     Triple {
    ///         subject: dictionary.encode("a"),
    ///         predicate: dictionary.encode("knows"),
    ///         object: dictionary.encode("b"),
    ///     }
    /// ];
    ///
    /// drop(dictionary);
    ///
    /// fbf.fbf_maintenance(to_add, to_remove);
    ///
    /// let all_facts: Vec<Triple> = fbf.reasoner.index_manager.query(None, None, None);
    ///
    /// assert!(all_facts.is_empty());
    /// ```
    pub fn fbf_maintenance(&mut self, to_add: Vec<Triple>, to_delete: Vec<Triple>) {
        let all_facts: Vec<Triple> = self.reasoner.index_manager.query(None, None, None);

        let mut blocked_facts: HashSet<Triple> = HashSet::new();
        let mut checked_facts: HashSet<Triple> = HashSet::new();
        let mut proved_facts: HashSet<Triple> = HashSet::new();
        let mut delayed_facts: HashSet<Triple> = HashSet::new();

        let deleted: HashSet<Triple> = self.delete_unproved(&to_delete, &all_facts, &mut blocked_facts, &mut checked_facts, &mut proved_facts, &mut delayed_facts);
        let mut added: HashSet<Triple> = blocked_facts.intersection(&deleted).filter(|&triple| proved_facts.contains(triple)).cloned().collect();
        let rederived: HashSet<Triple> = self.rederive(&deleted, &added, &to_delete, &all_facts, &blocked_facts, &proved_facts, &delayed_facts);
        added.extend(self.insert(&deleted, &rederived, &to_add, &all_facts));

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

    fn delete_unproved(&mut self, to_remove: &Vec<Triple>,
        all_facts: &Vec<Triple>,
        blocked_facts: &mut HashSet<Triple>,
        checked_facts: &mut HashSet<Triple>,
        proved_facts: &mut HashSet<Triple>,
        delayed_facts: &mut HashSet<Triple>
    ) -> HashSet<Triple> {
        // initialize n_d as the intersection between the explicitly removed facts and the facts in the index manager
        let mut n_d: HashSet<Triple> = to_remove.iter()
                .cloned()
                .collect();

        let mut delta: Vec<Triple>;
        let mut deleted: HashSet<Triple> = HashSet::new();
        let mut nd_without_deleted: HashSet<Triple>;

        loop {
            // calculate the unprocessed changes
            // delta = n_d.iter()
            //         .filter(|triple| !deleted.contains(triple))
            //         .cloned()
            //         .collect();

            delta = Vec::new();

            nd_without_deleted = n_d.iter()
                    .filter(|triple| !deleted.contains(triple))
                    .cloned()
                    .collect();

            // try to prove each fact that is to be deleted
            for triple in nd_without_deleted.into_iter() {
                self.check(triple.clone(), &deleted, &delta, all_facts, to_remove, blocked_facts, checked_facts, proved_facts, delayed_facts, self.steps);
                if !proved_facts.contains(&triple) {
                    delta.push(triple);
                }
            }

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
                n_d.extend(self.construct_triple_from_bindings(rule, &bindings));
            }

            deleted.extend(delta);
        }

        // return the deleted triples
        deleted
    }

    /// Try to prove a fact still holds.
    fn check(&mut self, triple: Triple, deleted: &HashSet<Triple>, delta: &Vec<Triple>, all_facts: &Vec<Triple>, to_remove: &Vec<Triple>, blocked_facts: &mut HashSet<Triple>, checked_facts: &mut HashSet<Triple>, proved_facts: &mut HashSet<Triple>, delayed_facts: &mut HashSet<Triple>, remaining_steps: u32) {
        // only proceed if the fact has not been checked before
        if !checked_facts.contains(&triple) {
            if remaining_steps <= 0 {
                // we have reached the maximum amount of backchaining, so the fact can not be proved right now
                blocked_facts.insert(triple);
            } else if self.saturate(triple.clone(), to_remove, checked_facts, proved_facts, delayed_facts) {

                // get all rule bindings so that the current checked fact is the conclusion of the bound rule
                let mut union = delta.iter().chain(deleted);
                let difference = all_facts.iter()
                        .filter(|&triple| union.any(|other_triple| other_triple == triple) == false)
                        .cloned()
                        .collect();

                let mut bindings: Vec<HashMap<String, u32>>;
                for rule in self.reasoner.rules.clone().iter() {
                    bindings = self.reasoner.evaluate_rule_with_delta_improved(rule, &proved_facts.iter().cloned().collect::<Vec<Triple>>(), &difference);
                    let conclusion_facts = self.construct_triple_from_bindings(rule, &bindings);

                    if conclusion_facts.contains(&triple) {
                        let premise_facts = self.construct_triple_from_patterns(&rule.premise, &rule.filters, &bindings);

                        for premise_fact in premise_facts {
                            self.check(premise_fact.clone(), deleted, delta, all_facts, to_remove, blocked_facts, checked_facts, proved_facts, delayed_facts, remaining_steps - 1);

                            if blocked_facts.difference(proved_facts).any(|item| item == &premise_fact) {
                                blocked_facts.insert(triple.clone());
                            }

                            if proved_facts.contains(&triple) {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Try to prove a triple, and return true/false to indicate success/failure.
    fn saturate(&mut self, triple: Triple, to_remove: &Vec<Triple>, checked_facts: &mut HashSet<Triple>, proved_facts: &mut HashSet<Triple>, delayed_facts: &mut HashSet<Triple>) -> bool {
        // the current fact is now being checked
        checked_facts.insert(triple.clone());
        let remaining_explicit: HashSet<Triple> = self.explicit.iter()
                .filter(|triple| !to_remove.contains(*triple))
                .cloned()
                .collect();

        // try to prove the fact using the delayed facts or the remaining explicit facts
        if delayed_facts.contains(&triple) || remaining_explicit.contains(&triple) {
            let mut n_p: HashSet<Triple> = HashSet::from([triple]);
            let mut delta_p: Vec<Triple>;

            loop {
                delta_p = n_p.iter().filter(|triple| checked_facts.contains(*triple) && !proved_facts.contains(*triple)).cloned().collect();

                // if a fact has not been checked, add it to the delayed facts so we don't need to derive them later
                delayed_facts.extend(n_p.difference(checked_facts).cloned());

                if delta_p.is_empty() {
                    return true;
                }

                // everything in delta_p has been proven
                proved_facts.extend(delta_p.clone());

                n_p.clear();

                // try to derive facts from the proven facts (these are thus also proven)
                let mut bindings: Vec<HashMap<String, u32>>;
                for rule in self.reasoner.rules.clone().iter() {
                    bindings = self.reasoner.evaluate_rule_with_delta_improved(rule, &proved_facts.iter().cloned().collect::<Vec<Triple>>(), &delta_p);
                    n_p.extend(self.construct_triple_from_bindings(rule, &bindings));
                }
            }
        } else {
            // the current fact could not be proven
            false
        }
    }

    /// Returns the facts that should not have been deleted.
    pub fn rederive(&mut self, deleted: &HashSet<Triple>,
        added: &HashSet<Triple>,
        to_delete: &Vec<Triple>,
        all_facts: &Vec<Triple>,
        blocked_facts: &HashSet<Triple>,
        proved_facts: &HashSet<Triple>,
        delayed_facts: &HashSet<Triple>
    ) -> HashSet<Triple> {

        // let mut rederived: HashSet<Triple> = self.explicit.iter()
        // .filter(|triple| !to_delete.contains(*triple))
        // .filter(|triple| deleted.contains(*triple))
        // .cloned()
        // .collect();

        // if an explicitly deleted fact is still in the explicit facts, or is a delayed fact, it needs to be rederived if it was blocked and deleted
        let mut deleted_not_proved = deleted.difference(proved_facts);
        let rederivable: HashSet<Triple> = blocked_facts.iter().filter(|&triple| deleted_not_proved.any(|other_triple| other_triple == triple)).cloned().collect();

        let mut explicit_or_delayed: HashSet<Triple> = self.explicit.iter().filter(|&triple| to_delete.contains(triple)).cloned().collect();
        explicit_or_delayed.extend(delayed_facts.into_iter().cloned());

        let mut rederived: HashSet<Triple> = rederivable.iter().filter(|&triple| explicit_or_delayed.contains(triple)).cloned().collect();


        // calculate the difference between all_facts and deleted
        let mut deleted_difference = deleted.difference(added);
        let difference: Vec<Triple> = all_facts.iter()
                .filter(|&triple| deleted_difference.any(|other_triple| other_triple == triple) == false)
                .cloned()
                .collect();

        let mut bindings: Vec<HashMap<String, u32>>;

        // evaluate the rules with the remaining facts
        for rule in self.reasoner.rules.clone().iter() {
            bindings = self.reasoner.evaluate_rule_with_optimized_join(rule, &difference);

            // construct triples from these bindings, but only keep those that were deleted by the overdeletion step
            let constructed_triples: HashSet<Triple> = self.construct_triple_from_bindings(rule, &bindings).into_iter()
                    .filter(|triple| rederivable.contains(triple))
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
                n_a.extend(self.construct_triple_from_bindings(rule, &bindings));
            }

            // add the delta to added
            added.extend(delta);
        }

        // return the added triples
        added
    }

    fn construct_triple_from_bindings(&mut self, rule: &Rule, bindings: &Vec<HashMap<String, u32>>) -> HashSet<Triple> {
        let mut result: HashSet<Triple> = HashSet::new();
        let mut inferred: Triple;

        for binding in bindings {
            // check if the binding adheres to the filters of the rule
            if evaluate_filters(binding, &rule.filters, &self.reasoner.dictionary.write().unwrap()) {

                for conclusion in &rule.conclusion {
                    inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary.write().unwrap());
                    result.insert(inferred);
                }
            }
        }

        result
    }

    fn construct_triple_from_patterns(&mut self, patterns: &Vec<TriplePattern>, filters: &Vec<FilterCondition>, bindings: &Vec<HashMap<String, u32>>) -> HashSet<Triple> {
        let mut result: HashSet<Triple> = HashSet::new();
        let mut inferred: Triple;

        for binding in bindings {
            // check if the binding adheres to the filters of the rule
            if evaluate_filters(binding, filters, &self.reasoner.dictionary.write().unwrap()) {

                for pattern in patterns {
                    inferred = construct_triple(pattern, &binding, &mut self.reasoner.dictionary.write().unwrap());
                    result.insert(inferred);
                }
            }
        }

        result
    }

    pub fn decode_triple(&self, triple: &Triple) -> String {
        let dictionary = self.reasoner.dictionary.write().unwrap();

        let subject = dictionary.decode(triple.subject).unwrap_or("unknown");
        let predicate = dictionary.decode(triple.predicate).unwrap_or("unknown");
        let object = dictionary.decode(triple.object).unwrap_or("unknown");

        format!("{} {} {}", subject, predicate, object)
    }
}









#[cfg(test)]
mod tests {
    use super::*;
    use shared::terms::Term;
    use shared::rule::Rule;
    use std::hash::Hash;
    use std::time::Instant;

    fn vec_equal<T: Eq+Hash>(vec1: &Vec<T>, vec2: &Vec<T>) -> bool {
        // TODO: is dit correct?
        let mut counts: HashMap<&T, u32> = HashMap::new();

        for element in vec1 {
            let value = counts.entry(element).or_insert(0);
            *value += 1;
        }

        for element in vec2 {
            match counts.get_mut(element) {
                Some(count) => {
                    if *count > 0 {
                        *count -= 1;
                    }
                },
                None => return false,
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
        let fbf = FBFMaintenance::new(3);
        let facts = fbf.reasoner.index_manager.query(None, None, None);

        assert!(fbf.explicit.is_empty());
        assert!(facts.is_empty());
    }

    #[test]
    fn maintenance_counting_graph_test() {
        let mut fbf = FBFMaintenance::new(3);

        fbf.add_abox_triple("a", "edge", "b");
        fbf.add_abox_triple("b", "edge", "c");
        fbf.add_abox_triple("e", "edge", "d");

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let edge = Term::Constant(dictionary.encode("edge"));
        let reachable = Term::Constant(dictionary.encode("reachable"));

        drop(dictionary);

        fbf.reasoner.add_rule(Rule {
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

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let encoded_edge = Term::Constant(dictionary.encode("edge"));
        let encoded_reachable = Term::Constant(dictionary.encode("reachable"));

        drop(dictionary);

        fbf.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_edge,
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                encoded_reachable,
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("c"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("e"),
            },
        ];

        drop(dictionary);

        fbf.reasoner.infer_new_facts_semi_naive();

        let start = Instant::now();
        fbf.fbf_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = fbf.reasoner.index_manager.query(None, None, None);

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let test_materialisation = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("e"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("e"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("e"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("e"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("e"),
            },
        ];

        let test_explicit = HashSet::from([
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("e"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("e"),
            },
        ]);

        drop(dictionary);

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, fbf.explicit);
    }

    #[test]
    fn fbf_maintenance_small_test() {
        let mut fbf = FBFMaintenance::new(3);

        fbf.add_abox_triple("a", "t", "b");
        fbf.add_abox_triple("f", "t", "b");
        fbf.add_abox_triple("c", "t", "d");

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let encoded_r = Term::Constant(dictionary.encode("r"));
        let encoded_t = Term::Constant(dictionary.encode("t"));

        drop(dictionary);

        fbf.reasoner.add_rule(Rule {
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

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dictionary.encode("f"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: dictionary.encode("g"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            }
        ];

        drop(dictionary);

        fbf.reasoner.infer_new_facts_semi_naive();

        let start = Instant::now();
        fbf.fbf_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = fbf.reasoner.index_manager.query(None, None, None);

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let test_materialisation = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("c"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("g"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("d"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("d"),
            },
        ];

        let test_explicit = HashSet::from([
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("c"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("g"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
        ]);

        drop(dictionary);

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, fbf.explicit);
    }

    #[test]
    fn fbf_maintenance_reflexive_test() {
        let mut fbf = FBFMaintenance::new(3);

        fbf.add_abox_triple("a", "r", "b");

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let encoded_r = Term::Constant(dictionary.encode("r"));

        drop(dictionary);

        fbf.reasoner.add_rule(Rule {
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

        fbf.reasoner.infer_new_facts_semi_naive();

        let dictionary = &fbf.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("b"),
            },
        ];

        drop(dictionary);

        let added: Vec<Triple> = vec![];

        let start = Instant::now();
        fbf.fbf_maintenance(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = fbf.reasoner.index_manager.query(None, None, None);
        let test_materialisation = vec![];

        let test_explicit = HashSet::new();

        assert!(vec_equal(&test_materialisation, &materialisation));
        assert_eq!(test_explicit, fbf.explicit);
    }
}
