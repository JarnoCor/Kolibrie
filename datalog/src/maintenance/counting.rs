use crate::reasoning::{Reasoner, construct_triple};
use crate::reasoning::{evaluate_filters};

use shared::triple::Triple;
use std::collections::{HashMap, HashSet};

/// Alias for a [`HashMap`] containing pairs of [`Triple`] and [`u32`].
/// 
/// This makes types and function signatures more readable.
pub type Multiset = HashMap<Triple, u32>;

/// A struct for incremental view maintenance (IVM) using the counting-based approach.
///
/// Contains the [`Reasoner`] to perform IVM on, as well as the trace. The first multiset of the trace contains the explicit facts.
pub struct CountingMaintenance {
    /// The [`Reasoner`] containing the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) to perform IVM on.
    pub reasoner: Reasoner,
    /// The trace containing a multiset of derived facts for each iteration.
    pub trace: Vec<Multiset>,
}

impl From<Reasoner> for CountingMaintenance {
    /// Creates a new [`CountingMaintenance`] with an empty trace from an existing [`Reasoner`].
    fn from(reasoner: Reasoner) -> CountingMaintenance {
        CountingMaintenance {
            reasoner,
            trace: Vec::new()
        }
    }
}

impl CountingMaintenance {
    /// Creates a new [`CountingMaintenance`] instance with a new [`Reasoner`] and an empty trace.
    pub fn new() -> CountingMaintenance {
        CountingMaintenance { reasoner:Reasoner::new(), trace: Vec::new() }
    }

    /// The counting version of the semi-naive materialisation algorithm.
    ///
    /// Returns a [`Vec`] containing the implicit facts.
    ///
    /// # Examples
    ///
    /// ```
    /// use shared::{rule::Rule, terms::Term, triple::Triple};
    /// use datalog::maintenance::CountingMaintenance;
    ///
    /// let mut counting_maintenance = CountingMaintenance::new();
    /// counting_maintenance.reasoner.add_abox_triple("a", "knows", "b");
    ///
    /// counting_maintenance.reasoner.add_rule(Rule {
    ///     premise: vec![
    ///     (
    ///         Term::Variable("x".to_string()),
    ///         Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("knows")),
    ///         Term::Variable("y".to_string()),
    ///     ),
    ///     ],
    ///     conclusion: vec![(
    ///         Term::Variable("y".to_string()),
    ///         Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("knows")),
    ///         Term::Variable("x".to_string()),
    ///     )],
    ///     filters: vec![],
    /// });
    ///
    /// let inferred: Vec<Triple> = counting_maintenance.semi_naive_counting();
    ///
    /// assert_eq!(counting_maintenance.decode_triple(&inferred[0]), String::from("b knows a"));
    /// ```
    pub fn semi_naive_counting(&mut self) -> Vec<Triple> {
        // get all facts currently in the index manager
        let mut all_facts: HashSet<Triple> = self.reasoner.index_manager.query(None, None, None).into_iter().collect();

        // this is a multiset containing which facts were derived how many times at each iteration
        let mut counts: Multiset = HashMap::new();
        counts.extend(all_facts.iter().cloned().map(|k| (k, 1)));

        let mut inferred_so_far: Vec<Triple> = Vec::new();
        let mut dataset_so_far: HashSet<Triple> = HashSet::new();
        let mut delta: HashSet<Triple>;

        loop {
            // calculate changes from the previous iterations
            delta = counts.keys().cloned().collect();
            delta = delta.difference(&dataset_so_far).cloned().collect();

            // add the multiset of iteration i to the trace
            self.trace.push(counts);

            // if there are no changes, stop
            if delta.is_empty() {
                break;
            }

            // update the dataset seen so far
            dataset_so_far = dataset_so_far.union(&delta).cloned().collect();

            counts = HashMap::new();

            let mut bindings: Vec<HashMap<String, u32>>;
            let mut inferred: Triple;

            for rule in self.reasoner.rules.clone() {
                // evaluate the rule using the facts in delta
                bindings = self.reasoner.evaluate_rule_with_delta_improved(&rule, &all_facts.iter().cloned().collect(), &Vec::from_iter(delta.clone()));

                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary) {

                        for conclusion in &rule.conclusion {
                            inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary);

                            // update the count of the inferred triple, or insert the triple with multiplicity 1 if it was not present
                            counts.entry(inferred.clone()).and_modify(|count| *count += 1).or_insert(1);

                            // if the inferred triple is a new fact, add it to all facts and to the the index manager
                            if !all_facts.contains(&inferred) {
                                all_facts.insert(inferred.clone());
                                inferred_so_far.push(inferred.clone());

                                // add the inferred triple to the index manager
                                self.reasoner.index_manager.insert(&inferred);
                            }
                        }
                    }
                }
            }
        }

        // return the facts that were implicitly added to the index manager
        inferred_so_far
    }

    /// Performs IVM to update the facts in the [`UnifiedIndex`](shared::index_manager::UnifiedIndex) of the [`Reasoner`] by adding and removing explicit and implicit facts using a counting based approach.
    ///
    /// # Examples
    ///
    /// ```
    /// use shared::{rule::Rule, terms::Term, triple::Triple};
    /// use datalog::maintenance::CountingMaintenance;
    ///
    /// let mut counting_maintenance = CountingMaintenance::new();
    /// counting_maintenance.reasoner.add_abox_triple("a", "knows", "b");
    ///
    /// counting_maintenance.reasoner.add_rule(Rule {
    ///     premise: vec![
    ///     (
    ///         Term::Variable("x".to_string()),
    ///         Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("knows")),
    ///         Term::Variable("y".to_string()),
    ///     ),
    ///     ],
    ///     conclusion: vec![(
    ///         Term::Variable("y".to_string()),
    ///         Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("knows")),
    ///         Term::Variable("x".to_string()),
    ///     )],
    ///     filters: vec![],
    /// });
    ///
    /// let _ = counting_maintenance.semi_naive_counting();
    /// 
    /// let to_add: Vec<Triple> = Vec::new();
    /// let to_remove: Vec<Triple> = vec![
    ///     Triple {
    ///         subject: counting_maintenance.reasoner.dictionary.encode("a"),
    ///         predicate: counting_maintenance.reasoner.dictionary.encode("knows"),
    ///         object: counting_maintenance.reasoner.dictionary.encode("b"),
    ///     }
    /// ];
    ///
    /// counting_maintenance.maintenance_counting(to_add, to_remove);
    ///
    /// let all_facts: Vec<Triple> = counting_maintenance.reasoner.index_manager.query(None, None, None);
    ///
    /// assert!(all_facts.is_empty());
    /// ```
    pub fn maintenance_counting(&mut self, added_facts: Vec<Triple>, removed_facts: Vec<Triple>) {
        let mut i_o: HashSet<Triple> = HashSet::new(); // the old dataset, updated per iteration
        let mut i_n: HashSet<Triple> = HashSet::new(); // the new dataset, updated per iteration
        let mut i_no: HashSet<Triple> = HashSet::new(); // facts that are in the new but not in the old dataset
        let mut i_on: HashSet<Triple> = HashSet::new(); // facts that are in the old but not in the new dataset

        // tracks the current iteration
        let mut i = 0;

        // contains multiset i of the old trace
        let mut n_o: Multiset = self.trace[i].clone();

        // update the i-th multiset of the trace
        multiset_add(&mut self.trace[i], added_facts);
        multiset_subtract(&mut self.trace[i], removed_facts);

        // contains multiset i of the new trace
        let mut n_n = &self.trace[i];

        // the deltas track new changes throughout iterations, for the old and new trace respectively
        let mut delta_old: HashSet<Triple>;
        let mut delta_new: HashSet<Triple>;

        loop {
            // calculate the delta using the old trace
            delta_old = n_o.keys().cloned().collect();
            delta_old = delta_old.difference(&i_o).cloned().collect();

            // calculate the delta using the new trace
            delta_new = n_n.keys().cloned().collect();
            delta_new = delta_new.difference(&i_n).cloned().collect();

            // if there is no change in both the old and new trace, stop
            if delta_old.is_empty() && delta_new.is_empty() {
                break;
            }

            // update the new and old datasets
            i_o = i_o.union(&delta_old).cloned().collect();
            i_n = i_n.union(&delta_new).cloned().collect();

            // helper variables to make calculation with multiple steps more readable
            let mut diff1: HashSet<Triple>;
            let mut diff2: HashSet<Triple>;

            // calculate which facts no longer hold
            diff1 = i_on.difference(&delta_new).cloned().collect();
            diff2 = delta_old.difference(&i_n).cloned().collect();

            i_on = diff1.union(&diff2).cloned().collect();

            // calculate which facts did not hold previously
            diff1 = i_no.difference(&delta_old).cloned().collect();
            diff2 = delta_new.difference(&i_o).cloned().collect();

            i_no = diff1.union(&diff2).cloned().collect();

            // if the old and new deltas are equal and nothing changes between the old and new trace, we can stop early
            if delta_old.difference(&delta_new).next() == None && i_on.is_empty() && i_no.is_empty() {
                break;
            }

            i += 1;

            // check if the trace contains a multiset for the next iteration, and add it if necessary
            if let None = self.trace.get(i) {
                self.trace.push(HashMap::new());
            }

            // contains multiset i of the old trace
            n_o = self.trace[i].clone();

            /*
            now we calculate which facts no longer hold and thus should be deleted
            there are two parts in this formula:
                1. facts that are derived from applying rules to the old dataset, where at least one body atom that is in the old delta, and at least one body atom that is in the old dataset but not in the new one was used to derive the fact
                2. facts that are derived from applying rules to the first symmetric difference as calculated below, where at least one body atom that is in either the new or old dataset, and at least one body atom that is in the second symmetric differnce was used to derive the fact
            */

            // calculate the symmetric differences, this is for the second part of the task described above
            let symm_diff1: Vec<Triple> = i_o.intersection(&i_n).cloned().collect::<HashSet<Triple>>().difference(&delta_new).cloned().collect();
            let symm_diff2: Vec<Triple> = delta_old.intersection(&i_n).cloned().collect::<HashSet<Triple>>().difference(&delta_new).cloned().collect();

            let mut removed_this_iteration: Vec<Triple> = Vec::new();
            let mut bindings: Vec<HashMap<String, u32>>;
            let mut inferred: Triple;

            // for each rule, calculate which facts are no longer derived from it
            for rule in self.reasoner.rules.clone() {
                // this is the first part of the task as described above
                removed_this_iteration.extend(self.reasoner.evaluate_rule_with_restrictions(&rule, &i_o, &delta_old, &i_on).into_iter());

                // this is the second part of the task as described above
                bindings = self.reasoner.evaluate_rule_with_delta_improved(&rule, &symm_diff1, &symm_diff2);
                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary) {

                        for conclusion in &rule.conclusion {
                            inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary);
                            removed_this_iteration.push(inferred);
                        }
                    }
                }

            }

            /*
            now we calculate which facts newly hold and thus should be added
            there are two parts in this formula:
                1. facts that are derived from applying rules to the new dataset, where at least one body atom that is in the new delta, and at least one body atom that is in the new dataset but not in the old one was used to derive the fact
                2. facts that are derived from applying rules to the first symmetric difference as calculated below, where at least one body atom that is in either the new or old dataset, and at least one body atom that is in the second symmetric differnce was used to derive the fact
            */

            // calculate the symmetric differences, this is for the second part of the task described above
            let symm_diff1: Vec<Triple> = i_n.intersection(&i_o).cloned().collect::<HashSet<Triple>>().difference(&delta_old).cloned().collect();
            let symm_diff2: Vec<Triple> = delta_new.intersection(&i_o).cloned().collect::<HashSet<Triple>>().difference(&delta_old).cloned().collect();

            let mut added_this_iteration: Vec<Triple> = Vec::new();
            let mut bindings: Vec<HashMap<String, u32>>;
            let mut inferred: Triple;

            // for each rule, calculate which facts are newly derived from it
            for rule in self.reasoner.rules.clone() {
                // this is the first part of the task as described above
                added_this_iteration.extend(self.reasoner.evaluate_rule_with_restrictions(&rule, &i_n, &delta_new, &i_no).into_iter());

                // this is the second part of the task as described above
                bindings = self.reasoner.evaluate_rule_with_delta_improved(&rule, &symm_diff1, &symm_diff2);
                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary) {

                        for conclusion in &rule.conclusion {
                            inferred = construct_triple(conclusion, &binding, &mut self.reasoner.dictionary);
                            added_this_iteration.push(inferred);
                        }
                    }
                }

            }

            // update the i-th multiset of the trace
            multiset_subtract(&mut self.trace[i], removed_this_iteration);
            multiset_add(&mut self.trace[i], added_this_iteration);

            // contains multiset i of the new trace
            n_n = &self.trace[i];
        }

        // the algorithm has ended, now we update the index manager containing the facts

        // remove facts that no longer hold
        for triple in i_on {
            self.reasoner.index_manager.delete(&triple);
        }

        // add facts that have been newly derived
        for triple in i_no {
            self.reasoner.index_manager.insert(&triple);
        }
    }

    /// Decodes a triple into human-readable format using the [`Reasoner`]'s dictionary.
    /// This is useful when printing a triple for debugging purposes.
    ///
    /// # Examples
    ///
    /// ```
    /// use shared::triple::Triple;
    /// use datalog::maintenance::CountingMaintenance;
    ///
    /// let mut counting_maintenance = CountingMaintenance::new();
    /// counting_maintenance.reasoner.add_abox_triple("a", "edge", "b");
    ///
    /// let triple = &counting_maintenance.reasoner.index_manager.query(None, None, None)[0];
    ///
    /// assert_eq!(counting_maintenance.decode_triple(triple), String::from("a edge b"));
    /// ```
    pub fn decode_triple(&self, triple: &Triple) -> String {
        let subject = self.reasoner.dictionary.decode(triple.subject).unwrap_or("unknown");
        let predicate = self.reasoner.dictionary.decode(triple.predicate).unwrap_or("unknown");
        let object = self.reasoner.dictionary.decode(triple.object).unwrap_or("unknown");

        format!("{} {} {}", subject, predicate, object)
    }

    /// Decodes the trace into human-readable format.
    pub fn decode_trace(&self) -> Vec<HashMap<String, u32>> {
        let mut trace = Vec::new();

        for counts in &self.trace {
            let mut decoded = HashMap::new();

            for (key, value) in counts {
                let decoded_triple = self.decode_triple(key);
                decoded.insert(decoded_triple, value.to_owned());
            }

            trace.push(decoded)
        }

        trace
    }

    /// Prints the trace in a human-readable format.
    pub fn print_trace(&self) {
        let decoded = self.decode_trace();
        println!("{:#?}", decoded);
    }
}

/// Increases the count of elements in a multiset by the amount of times the element appears in the [`Vec`].
///
/// Elements that were not previously present are added to the multiset and returned.
///
/// # Examples
///
/// ```
/// use shared::triple::Triple;
/// use std::collections::HashMap;
/// use datalog::maintenance::{Multiset, multiset_add};
///
/// let mut multiset: Multiset = HashMap::from([
///     (Triple { subject: 1, predicate: 2, object: 3}, 1)
/// ]);
///
/// let triples: Vec<Triple> = vec![
///     Triple { subject: 1, predicate: 2, object: 3 },
///     Triple { subject: 1, predicate: 2, object: 3 },
///     Triple { subject: 3, predicate: 4, object: 1 },
/// ];
///
/// let added: Vec<Triple> = multiset_add(&mut multiset, triples);
///
/// assert_eq!(added, vec![Triple { subject: 3, predicate: 4, object: 1 }]);
/// assert_eq!(multiset.get(&Triple { subject: 1, predicate: 2, object: 3}), Some(&3));
/// assert_eq!(multiset.get(&Triple { subject: 3, predicate: 4, object: 1}), Some(&1));
/// ```
pub fn multiset_add(multiset: &mut Multiset, added: Vec<Triple>) -> Vec<Triple> {
    let mut new = Vec::new();

    for triple in added {
        if let None = multiset.get(&triple) {
            new.push(triple.clone());
        }

        multiset.entry(triple).and_modify(|count| *count += 1).or_insert(1);
    }

    new
}

/// Subtracts the count of elements in a multiset by the amount of times the element appears in the [`Vec`].
///
/// Elements whose count reaches 0 or less are removed from the multiset and returned.
///
/// # Examples
///
/// ```
/// use shared::triple::Triple;
/// use std::collections::HashMap;
/// use datalog::maintenance::{Multiset, multiset_subtract};
///
/// let mut multiset: Multiset = HashMap::from([
///     (Triple { subject: 1, predicate: 2, object: 3}, 2),
///     (Triple { subject: 3, predicate: 4, object: 1}, 3)
/// ]);
///
/// let triples: Vec<Triple> = vec![
///     Triple { subject: 1, predicate: 2, object: 3 },
///     Triple { subject: 1, predicate: 2, object: 3 },
///     Triple { subject: 3, predicate: 4, object: 1 },
/// ];
///
/// let removed: Vec<Triple> = multiset_subtract(&mut multiset, triples);
///
/// assert_eq!(removed, vec![Triple { subject: 1, predicate: 2, object: 3 }]);
/// assert_eq!(multiset.get(&Triple { subject: 1, predicate: 2, object: 3}), None);
/// assert_eq!(multiset.get(&Triple { subject: 3, predicate: 4, object: 1}), Some(&2));
/// ```
pub fn multiset_subtract(multiset: &mut Multiset, subtracted: Vec<Triple>) -> Vec<Triple> {
    let mut removed = Vec::new();

    for triple in subtracted {
        multiset.entry(triple.clone()).and_modify(|count| *count -= 1);

        if let Some(value) = multiset.get(&triple) {
            if value == &0 {
                multiset.remove(&triple);
                removed.push(triple);
            }
        }
    }

    removed
}









#[cfg(test)]
mod tests {
    use super::*;
    use shared::terms::Term;
    use shared::rule::Rule;
    use std::time::Instant;


    #[test]
    fn new_from_test() {
        let mut reasoner = Reasoner::new();
        reasoner.add_abox_triple("a", "edge", "b");

        let counting_maintenance = CountingMaintenance::from(reasoner);
        let triples = counting_maintenance.reasoner.index_manager.query(None, None, None);

        assert_eq!(counting_maintenance.trace.is_empty(), true);
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn multiset_add_test() {
        let mut multiset: Multiset = HashMap::new();

        let triples = vec![
            Triple { subject: 1, predicate: 2, object: 3 },
            Triple { subject: 1, predicate: 2, object: 3 },
            Triple { subject: 3, predicate: 4, object: 1 },
            Triple { subject: 1, predicate: 4, object: 5 },
        ];

        multiset_add(&mut multiset, triples.clone());

        assert_eq!(multiset.get(&Triple { subject: 1, predicate: 2, object: 3 }).unwrap(), &2);
        assert_eq!(multiset.get(&Triple { subject: 3, predicate: 4, object: 1 }).unwrap(), &1);
        assert_eq!(multiset.get(&Triple { subject: 1, predicate: 4, object: 5 }).unwrap(), &1);
    }

    #[test]
    fn multiset_subtract_test() {
        let mut multiset = HashMap::new();

        let triples = vec![
            Triple { subject: 1, predicate: 2, object: 3 },
            Triple { subject: 1, predicate: 2, object: 3 },
            Triple { subject: 3, predicate: 4, object: 1 },
            Triple { subject: 1, predicate: 4, object: 5 },
        ];

        multiset_add(&mut multiset, triples.clone());


        let triples = vec![
            Triple { subject: 1, predicate: 2, object: 3 },
            Triple { subject: 3, predicate: 4, object: 1 },
        ];

        multiset_subtract(&mut multiset, triples);

        assert_eq!(multiset.get(&Triple { subject: 1, predicate: 2, object: 3 }).unwrap(), &1);
        assert_eq!(multiset.get(&Triple { subject: 3, predicate: 4, object: 1 }), None);
        assert_eq!(multiset.get(&Triple { subject: 1, predicate: 4, object: 5 }).unwrap(), &1);
    }

    #[test]
    fn semi_naive_counting_transitivity_test() {
        let mut counting_maintenance = CountingMaintenance::new();

        // println!("Explicit facts: [");

        for i in 0..2 {
        let subject = format!("person{}", i);
        let object = format!("person{}", (i + 1) % 5); // Wraps around to connect the last person with the first

        // println!("    \"{} likes {}\",", subject, object);
        counting_maintenance.reasoner.add_abox_triple(&subject, "likes", &object);
        }

        // println!("]");

        // Add transitivity rule: likes(X, Y) & likes(Y, Z) => likes(X, Z)
        counting_maintenance.reasoner.add_rule(Rule {
        premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("likes")),
                Term::Variable("y".to_string()),
            ),
            (
                Term::Variable("y".to_string()),
                Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("likes")),
                Term::Variable("z".to_string()),
            ),
        ],
        conclusion: vec![(
            Term::Variable("x".to_string()),
            Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("likes")),
            Term::Variable("z".to_string()),
        )],
        filters: vec![],
    });

    // Run the optimized inference
    let start = Instant::now();
    let inferred_facts = counting_maintenance.semi_naive_counting();
    let duration = start.elapsed();
    println!("Inference took: {:?}", duration);

    // let mut readable_inferred_facts = Vec::new();

    // for triple in inferred_facts.clone() {
    //         let subject = cmg.kg.dictionary.decode(triple.subject).unwrap();
    //         let predicate = cmg.kg.dictionary.decode(triple.predicate).unwrap();
    //         let object = cmg.kg.dictionary.decode(triple.object).unwrap();

    //         readable_inferred_facts.push(format!("{} {} {}", subject, predicate, object));

    // }

    // println!("inferred facts: {:#?}", readable_inferred_facts);
    // cmg.print_trace();


    let count = counting_maintenance.trace[1].get(&inferred_facts[0]).unwrap().to_owned();
    assert_eq!(count, 1);

    }


    #[test]
    fn semi_naive_counting_graph_test() {
        let mut counting_maintenance = CountingMaintenance::new();

        counting_maintenance.reasoner.add_abox_triple("a", "edge", "b");
        counting_maintenance.reasoner.add_abox_triple("b", "edge", "c");
        counting_maintenance.reasoner.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("edge"));
        let reachable = Term::Constant(counting_maintenance.reasoner.dictionary.encode("reachable"));

        counting_maintenance.reasoner.add_rule(Rule {
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

        counting_maintenance.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(counting_maintenance.reasoner.dictionary.clone().encode("reachable")),
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        // cmg.kg.add_rule(Rule {
        //     premise: vec![
        //     (
        //         Term::Variable("x".to_string()),
        //         Term::Constant(cmg.kg.dictionary.clone().encode("edge")),
        //         Term::Variable("y".to_string()),
        //     ),
        //     ],
        //     conclusion: vec![(
        //         Term::Variable("y".to_string()),
        //         Term::Constant(cmg.kg.dictionary.clone().encode("edge")),
        //         Term::Variable("x".to_string()),
        //     )],
        //     filters: vec![],
        // });

    let inferred_facts = counting_maintenance.semi_naive_counting();

    // cmg.print_trace();

    let all_facts = counting_maintenance.reasoner.index_manager.query(None, None, None);

    assert_eq!(7, all_facts.len());
    assert_eq!(4, inferred_facts.len());
    }

    #[test]
    fn semi_naive_counting_reflexive_test() {
        let mut cmg = CountingMaintenance::new();

        cmg.reasoner.add_abox_triple("a", "r", "b");

        let encoded_r = Term::Constant(cmg.reasoner.dictionary.encode("r"));

        cmg.reasoner.add_rule(Rule {
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

        let _ = cmg.semi_naive_counting();
        // cmg.print_trace();

        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("r"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("r"),
                    object: cmg.reasoner.dictionary.encode("a"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("r"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1)
            ])
        );

        assert_eq!(test_trace, cmg.trace);
    }

    #[test]
    fn maintenance_counting_graph_test() {
        let mut cmg = CountingMaintenance::new();

        cmg.reasoner.add_abox_triple("a", "edge", "b");
        cmg.reasoner.add_abox_triple("b", "edge", "c");
        cmg.reasoner.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(cmg.reasoner.dictionary.clone().encode("edge"));
        let reachable = Term::Constant(cmg.reasoner.dictionary.encode("reachable"));

        cmg.reasoner.add_rule(Rule {
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

        cmg.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(cmg.reasoner.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(cmg.reasoner.dictionary.clone().encode("reachable")),
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let _ = cmg.semi_naive_counting();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("b"),
                predicate: cmg.reasoner.dictionary.encode("edge"),
                object: cmg.reasoner.dictionary.encode("c"),
            },
        ];

        let start = Instant::now();
        cmg.maintenance_counting(Vec::new(), removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);


        // cmg.print_trace();

        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(HashMap::new());
        test_trace.push(HashMap::new());

        assert_eq!(test_trace, cmg.trace);

        let all_facts = cmg.reasoner.index_manager.query(None, None, None);

        let test_facts = vec![
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("b"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("b"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
        ];

        // println!("Trace");
        // println!("{:#?}", cmg.decode_trace());

        // println!("Facts:");
        // let mut facts = vec![];
        // for fact in &all_facts {
        //     facts.push(cmg.decode_triple(fact));
        // }
        // println!("{:#?}", facts);

        assert_eq!(test_facts.into_iter().collect::<HashSet<Triple>>(), all_facts.into_iter().collect::<HashSet<Triple>>());
    }

    #[test]
    fn maintenance_counting_small_test() {
        let mut cmg = CountingMaintenance::new();

        cmg.reasoner.add_abox_triple("a", "t", "b");
        cmg.reasoner.add_abox_triple("f", "t", "b");
        cmg.reasoner.add_abox_triple("c", "t", "d");

        let encoded_r = Term::Constant(cmg.reasoner.dictionary.encode("r"));
        let encoded_t = Term::Constant(cmg.reasoner.dictionary.encode("t"));

        cmg.reasoner.add_rule(Rule {
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

        let _ = cmg.semi_naive_counting();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("f"),
                predicate: cmg.reasoner.dictionary.encode("t"),
                object: cmg.reasoner.dictionary.encode("b"),
            },
            Triple {
                subject: cmg.reasoner.dictionary.encode("a"),
                predicate: cmg.reasoner.dictionary.encode("t"),
                object: cmg.reasoner.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("g"),
                predicate: cmg.reasoner.dictionary.encode("t"),
                object: cmg.reasoner.dictionary.encode("d"),
            }
        ];

        let start = Instant::now();
        cmg.maintenance_counting(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        // cmg.print_trace();

        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("c"),
                    predicate: cmg.reasoner.dictionary.encode("t"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("g"),
                    predicate: cmg.reasoner.dictionary.encode("t"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("d"),
                    predicate: cmg.reasoner.dictionary.encode("r"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 2)
            ])
        );

        test_trace.push(HashMap::new());

        assert_eq!(test_trace, cmg.trace);

        let all_facts = cmg.reasoner.index_manager.query(None, None, None);

        let test_facts = vec![
            Triple {
                    subject: cmg.reasoner.dictionary.encode("c"),
                    predicate: cmg.reasoner.dictionary.encode("t"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("g"),
                    predicate: cmg.reasoner.dictionary.encode("t"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("d"),
                    predicate: cmg.reasoner.dictionary.encode("r"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
        ];

        // println!("Trace");
        // println!("{:#?}", cmg.decode_trace());

        // println!("Facts:");
        // let mut facts = vec![];
        // for fact in &all_facts {
        //     facts.push(cmg.decode_triple(fact));
        // }
        // println!("{:#?}", facts);

        assert_eq!(test_facts.into_iter().collect::<HashSet<Triple>>(), all_facts.into_iter().collect::<HashSet<Triple>>());
    }

    #[test]
    fn maintenance_counting_reflexive_test() {
        let mut cmg = CountingMaintenance::new();

        cmg.reasoner.add_abox_triple("a", "r", "b");

        let encoded_r = Term::Constant(cmg.reasoner.dictionary.encode("r"));

        cmg.reasoner.add_rule(Rule {
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

        let _ = cmg.semi_naive_counting();
        // cmg.print_trace();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("a"),
                predicate: cmg.reasoner.dictionary.encode("r"),
                object: cmg.reasoner.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![];

        let start = Instant::now();
        cmg.maintenance_counting(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let _ = cmg.reasoner.index_manager.query(None, None, None);

        // cmg.print_trace();

        let test_trace = vec![HashMap::new(), HashMap::new(), HashMap::new()];

        assert_eq!(test_trace, cmg.trace);

        let all_facts = cmg.reasoner.index_manager.query(None, None, None);

        // println!("Trace");
        // println!("{:#?}", cmg.decode_trace());

        // println!("Facts:");
        // let mut facts = vec![];
        // for fact in &all_facts {
        //     facts.push(cmg.decode_triple(fact));
        // }
        // println!("{:#?}", facts);

        assert_eq!(HashSet::new(), all_facts.into_iter().collect::<HashSet<Triple>>());
    }

    #[test]
    fn maintenance_counting_graph_alternative_route_test() {
        let mut cmg = CountingMaintenance::new();

        cmg.reasoner.add_abox_triple("a", "edge", "b");
        cmg.reasoner.add_abox_triple("b", "edge", "c");
        cmg.reasoner.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(cmg.reasoner.dictionary.clone().encode("edge"));
        let reachable = Term::Constant(cmg.reasoner.dictionary.encode("reachable"));

        cmg.reasoner.add_rule(Rule {
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

        cmg.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(cmg.reasoner.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(cmg.reasoner.dictionary.clone().encode("reachable")),
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let _ = cmg.semi_naive_counting();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("b"),
                predicate: cmg.reasoner.dictionary.encode("edge"),
                object: cmg.reasoner.dictionary.encode("c"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: cmg.reasoner.dictionary.encode("b"),
                predicate: cmg.reasoner.dictionary.encode("edge"),
                object: cmg.reasoner.dictionary.encode("e"),
            },
        ];

        let all_facts = cmg.reasoner.index_manager.query(None, None, None);

        println!("Old facts:");
        let mut facts = vec![];
        for fact in &all_facts {
            facts.push(cmg.decode_triple(fact));
        }
        println!("{:#?}", facts);

        let start = Instant::now();
        cmg.maintenance_counting(added, removed);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);


        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("e"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("e"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1),
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("e"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(HashMap::new());

        assert_eq!(test_trace, cmg.trace);

        let all_facts = cmg.reasoner.index_manager.query(None, None, None);

        let test_facts = vec![
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("b"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("edge"),
                    object: cmg.reasoner.dictionary.encode("e"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("b"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("e"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("e"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("b"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("e"),
                },
            Triple {
                    subject: cmg.reasoner.dictionary.encode("a"),
                    predicate: cmg.reasoner.dictionary.encode("reachable"),
                    object: cmg.reasoner.dictionary.encode("d"),
                },
        ];

        // println!("Trace");
        // println!("{:#?}", cmg.decode_trace());

        println!("Facts:");
        let mut facts = vec![];
        for fact in &all_facts {
            facts.push(cmg.decode_triple(fact));
        }
        println!("{:#?}", facts);

        assert_eq!(test_facts.into_iter().collect::<HashSet<Triple>>(), all_facts.into_iter().collect::<HashSet<Triple>>());
    }
}
