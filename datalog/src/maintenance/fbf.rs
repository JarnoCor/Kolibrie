use crate::maintenance::construct_triple;
use crate::reasoning::Reasoner;
use crate::reasoning::materialisation::replace_variables_with_bound_values;
use crate::reasoning::rules::{join_rule, evaluate_filters};

use shared::dictionary::Dictionary;
use shared::{rule::{FilterCondition, Rule}, terms::TriplePattern, triple::Triple};

use super::find_premise_solutions;

use std::collections::{HashMap, HashSet};

/// A struct for incremental view maintenance (IVM) using the Forward/Backward/Forward approach.
///
/// Contains the [`Reasoner`] to perform IVM on, as well as the explicit facts.
#[derive(Clone)]
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

        let mut added: HashSet<Triple> = HashSet::new();

        for triple in blocked_facts.iter() {
            if deleted.contains(triple) && proved_facts.contains(triple) {
                added.insert(*triple);
            }
        }

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
        let mut n_d: HashSet<Triple> = to_remove.iter().copied().collect();

        let mut delta: HashSet<Triple>;
        let mut deleted: HashSet<Triple> = HashSet::new();

        let dictionary = self.reasoner.dictionary.clone();
        let mut dictionary = dictionary.write().unwrap();

        loop {
            delta = HashSet::new();

            // try to prove each fact that is to be deleted
            for triple in n_d.difference(&deleted) {
                self.check(*triple, &deleted, &delta, all_facts, to_remove, blocked_facts, checked_facts, proved_facts, delayed_facts, self.steps, &mut dictionary);
                if !proved_facts.contains(triple) {
                    delta.insert(*triple);
                }
            }

            // if there are no more changes, stop
            if delta.is_empty() {
                break;
            }

            // clear the contents of the previous iteration
            n_d.clear();

            // calculate the difference between all_facts and deleted
            let difference: HashSet<Triple> = all_facts.iter()
                    .filter(|triple| !deleted.contains(*triple))
                    .copied()
                    .collect();

            let mut bindings: Vec<HashMap<String, u32>>;
            for rule in self.reasoner.rules.clone().iter() {
                bindings = join_rule(rule, &difference, &delta);
                n_d.extend(self.construct_triple_from_bindings(rule, bindings, &mut dictionary));
            }

            deleted.extend(delta);
        }

        drop(dictionary);

        // return the deleted triples
        deleted
    }

    /// Try to prove a fact still holds.
    fn check(&mut self, fact: Triple, deleted: &HashSet<Triple>, delta: &HashSet<Triple>, all_facts: &Vec<Triple>, to_remove: &Vec<Triple>, blocked_facts: &mut HashSet<Triple>, checked_facts: &mut HashSet<Triple>, proved_facts: &mut HashSet<Triple>, delayed_facts: &mut HashSet<Triple>, remaining_steps: u32, dict: &mut Dictionary) {
        // only proceed if the fact has not been checked before
        if !checked_facts.contains(&fact) {
            if remaining_steps <= 0 {
                // we have reached the maximum amount of backchaining, so the fact can not be proved right now
                blocked_facts.insert(fact);
            } else if !self.saturate(fact, to_remove, all_facts, deleted, checked_facts, proved_facts, delayed_facts, dict) {

                // get all rule bindings so that the current checked fact is the conclusion of the bound rule
                let mut difference: Vec<Triple> = Vec::new();

                for triple in all_facts.iter() {
                    if !deleted.contains(triple) && !delta.contains(triple) && !blocked_facts.contains(triple) {
                        difference.push(*triple);
                    }
                }

                let mut bindings: Vec<HashMap<String, u32>>;
                for rule in self.reasoner.rules.clone().iter() {
                    bindings = find_premise_solutions(dict, rule, &difference);
                    let conclusion_facts = self.construct_triple_from_bindings(rule, bindings.clone(), dict);

                    if conclusion_facts.contains(&fact) {
                        let premise_facts = self.construct_triple_from_patterns(&rule.premise, &rule.filters, &bindings, dict);

                        for premise_fact in premise_facts {
                            self.check(premise_fact, deleted, delta, all_facts, to_remove, blocked_facts, checked_facts, proved_facts, delayed_facts, remaining_steps - 1, dict);

                            if blocked_facts.difference(proved_facts).any(|item| *item == premise_fact) {
                                blocked_facts.insert(fact);
                            }

                            if proved_facts.contains(&fact) {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Try to prove a triple, and return true/false to indicate success/failure.
    fn saturate(&mut self, fact: Triple, to_remove: &Vec<Triple>, all_facts: &Vec<Triple>, deleted: &HashSet<Triple>, checked_facts: &mut HashSet<Triple>, proved_facts: &mut HashSet<Triple>, delayed_facts: &mut HashSet<Triple>, dict: &mut Dictionary) -> bool {
        // the current fact is now being checked
        checked_facts.insert(fact);

        let mut difference: HashSet<Triple> = HashSet::new();

        for triple in all_facts.iter() {
            if !deleted.contains(triple) {
                difference.insert(*triple);
            }
        }

        // try to prove the fact using the delayed facts or the remaining explicit facts
        if delayed_facts.contains(&fact) || (self.explicit.contains(&fact) && !to_remove.contains(&fact))/*remaining_explicit.contains(&triple)*/ {
            let mut n_p: HashSet<Triple> = HashSet::from([fact]);
            let mut delta_p: HashSet<Triple> = HashSet::new();

            loop {
                delta_p.clear();

                for triple in n_p.intersection(checked_facts) {
                    if !proved_facts.contains(triple) {
                        delta_p.insert(*triple);
                    }
                }

                // delta_p = n_p.iter().filter(|triple| checked_facts.contains(*triple) && !proved_facts.contains(*triple)).cloned().collect();

                // if a fact has not been checked, add it to the delayed facts so we don't need to derive them later
                delayed_facts.extend(n_p.difference(checked_facts));

                if delta_p.is_empty() {
                    return true;
                }

                // everything in delta_p has been proven
                proved_facts.extend(delta_p.iter());


                let set: HashSet<Triple> = proved_facts.union(&difference).copied().collect();
                n_p.clear();

                // try to derive facts from the proven facts (these are thus also proven)
                let mut bindings: Vec<HashMap<String, u32>>;
                for rule in self.reasoner.rules.clone().iter() {
                    bindings = join_rule(rule, &set, &delta_p);
                    n_p.extend(self.construct_triple_from_bindings(rule, bindings, dict));
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

        // if an explicitly deleted fact is still in the explicit facts, or is a delayed fact, it needs to be rederived if it was blocked and deleted
        let mut rederivable: HashSet<Triple> = HashSet::new();
        for triple in deleted.difference(proved_facts) {
            if blocked_facts.contains(triple) {
                rederivable.insert(*triple);
            }
        }

        let mut rederived: HashSet<Triple> = HashSet::new();

        let temp: HashSet<Triple> = deleted.difference(added).copied().collect();
        let difference: Vec<Triple> = all_facts.iter()
                .filter(|triple| !temp.contains(*triple))
                .copied()
                .collect();

        let mut bindings: Vec<HashMap<String, u32>>;

        let dictionary = self.reasoner.dictionary.clone();
        let mut dictionary = dictionary.write().unwrap();

        let mut heads: HashSet<Triple> = HashSet::new();

        // evaluate the rules with the remaining facts
        for rule in self.reasoner.rules.clone().iter() {
            bindings = find_premise_solutions(&dictionary, rule, &difference);

            // construct triples from these bindings, but only keep those that were deleted by the overdeletion step
            let constructed_triples: HashSet<Triple> = self.construct_triple_from_bindings(rule, bindings, &mut dictionary);
            heads.extend(constructed_triples);
        }

        drop(dictionary);

        for triple in rederivable.iter() {
            if (self.explicit.contains(triple) && !to_delete.contains(triple)) || delayed_facts.contains(triple) {
                rederived.insert(*triple);
            } else if heads.contains(triple) {
                rederived.insert(*triple);
            }
        }

        // return the facts that should be rederived
        rederived
    }

    /// Returns the facts that should be (re)inserted.
    fn insert(&mut self, deleted: &HashSet<Triple>, rederived: &HashSet<Triple>, to_add: &Vec<Triple>, all_facts: &Vec<Triple>) -> HashSet<Triple> {

        let mut n_a: Vec<Triple> = to_add.iter().copied().collect();

        n_a.extend(rederived.iter().copied());

        let mut delta: HashSet<Triple>;
        let mut added: HashSet<Triple> = HashSet::new();

        let difference: HashSet<Triple> = all_facts.iter()
                .filter(|triple| !deleted.contains(*triple))
                .copied()
                .collect();

        let dictionary = self.reasoner.dictionary.clone();
        let mut dictionary = dictionary.write().unwrap();

        loop {

            // calculate the unprocessed changes
            delta = n_a.iter()
                    .filter(|triple| !difference.contains(*triple) && !added.contains(*triple))
                    .cloned()
                    .collect();

            // if there are no more changes, stop
            if delta.is_empty() {
                break;
            }

            // add the delta to added
            added.extend(delta.iter().copied());

            // clear the contents of the previous iteration
            n_a.clear();

            let facts: HashSet<Triple> = difference.iter().copied().chain(added.iter().copied()).collect();

            let mut bindings: Vec<HashMap<String, u32>>;
            for rule in self.reasoner.rules.clone().iter() {
                bindings = join_rule(rule, &facts, &delta);
                n_a.extend(self.construct_triple_from_bindings(rule, bindings, &mut dictionary));
            }
        }

        drop(dictionary);

        // return the added triples
        added
    }

    fn construct_triple_from_bindings(&mut self, rule: &Rule, bindings: Vec<HashMap<String, u32>>, dict: &mut Dictionary) -> HashSet<Triple> {
        let mut result: HashSet<Triple> = HashSet::new();
        let mut inferred: Triple;

        for binding in bindings {
            // check if the binding adheres to the filters of the rule
            if evaluate_filters(&binding, &rule.filters, dict) {

                for conclusion in &rule.conclusion {
                    inferred = construct_triple(conclusion, &binding, dict);
                    result.insert(inferred);
                }
            }
        }

        result
    }

    fn construct_triple_from_patterns(&mut self, patterns: &Vec<TriplePattern>, filters: &Vec<FilterCondition>, bindings: &Vec<HashMap<String, u32>>, dict: &mut Dictionary) -> HashSet<Triple> {
        let mut result: HashSet<Triple> = HashSet::new();
        let mut inferred: Triple;

        for binding in bindings {
            // check if the binding adheres to the filters of the rule
            if evaluate_filters(binding, filters, dict) {

                for pattern in patterns {
                    inferred = replace_variables_with_bound_values(pattern, &binding, dict);
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
            negative_premise: vec![],
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
            negative_premise: vec![],
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
            negative_premise: vec![],
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
            negative_premise: vec![],
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
