use crate::reasoning::{Reasoner, construct_triple};
use crate::reasoning::{evaluate_filters};

use shared::triple::Triple;
use std::collections::{/*BTreeMap, */HashMap, HashSet};


/// A wrapper struct to keep track of the trace.
/// Contains methods for counting incremental view maintenance.
pub struct CountingMaintenanceGraph {
    /// The knowledge graph to perform incremental view maintenance on
    pub kg:Reasoner,
    /// The trace
    pub trace: Vec<HashMap<Triple, u32>>,
}

impl CountingMaintenanceGraph {
    pub fn new() -> CountingMaintenanceGraph {
        CountingMaintenanceGraph {
            kg:Reasoner::new(),
            trace: Vec::new(),
        }
    }


    pub fn semi_naive_counting(&mut self) -> Vec<Triple> {

        // initialisaties
        let all_initial: HashSet<Triple> = self.kg.index_manager.query(None, None, None).into_iter().collect();
        let mut all_facts: HashSet<Triple> = all_initial.iter().cloned().collect();

        let mut counts = HashMap::new();
        counts.extend(all_initial.clone().into_iter().map(|k| (k, 1)));

        let mut inferred_so_far = Vec::new();

        let mut big_i = HashSet::new();

        loop {
            let mut delta: HashSet<Triple> = counts.keys().cloned().collect();
            delta = delta.difference(&big_i).cloned().collect();

            self.trace.push(counts);

            if delta.is_empty() {
                break;
            }

            big_i = big_i.union(&delta).cloned().collect();

            counts = HashMap::new();

            for rule in self.kg.rules.clone() {
                // evaluate the rule using the facts in delta
                let bindings = self.kg.evaluate_rule_with_delta_improved(&rule, &all_facts.iter().cloned().collect(), &Vec::from_iter(delta.clone()));

                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.kg.dictionary) {
                        for conclusion in &rule.conclusion {
                            let inferred = construct_triple(conclusion, &binding, &mut self.kg.dictionary);

                            // update the count of the inferred triple, or insert the triple with multiplicity 1 if it was not present
                            counts.entry(inferred.clone()).and_modify(|count| *count += 1).or_insert(1);

                            if !all_facts.contains(&inferred) {
                                all_facts.insert(inferred.clone());
                                inferred_so_far.push(inferred.clone());
                                self.kg.index_manager.insert(&inferred);
                            }
                        }
                    }
                }
            }        }

        inferred_so_far
    }


    // pub fn semi_naive_counting(&mut self) -> Vec<Triple> {
    //     // begin N = originele feiten + toepassen van regels = inferred_so_far

    //     // initialisaties
    //     let all_initial = self.kg.index_manager.query(None, None, None);
    //     let mut all_facts: HashSet<Triple> = all_initial.iter().cloned().collect();

    //     let mut counts = HashMap::new();
    //     counts.extend(all_initial.clone().into_iter().map(|k| (k, 1)));

    //     let mut delta = all_initial;
    //     let mut inferred_so_far = Vec::new();

    //     loop {

    //         let mut new_delta = HashSet::new();



    //         // paper: eerst alle rule instances toevoegen, en daarna degenen die we al hadden wegdoen
    //         // praktijk: voor het toevoegen van rule instances, kijken of ze er al inzitten

    //         // delta berekenen
    //         for rule in self.kg.rules.clone() {
    //             // evaluate the rule using the facts in delta
    //             let bindings = self.kg.evaluate_rule_with_delta_improved(&rule, &all_facts.iter().cloned().collect(), &delta);

    //             for binding in bindings {
    //                 // check if the binding adheres to the filters of the rule
    //                 if evaluate_filters(&binding, &rule.filters, &self.kg.dictionary) {
    //                     for conclusion in &rule.conclusion {
    //                         let inferred = construct_triple(conclusion, &binding, &mut self.kg.dictionary);
                            
    //                         // update the count of the inferred triple, or insert the triple with multiplicity 1 if it was not present
    //                         counts.entry(inferred.clone()).and_modify(|count| *count += 1).or_insert(1);

    //                         if !all_facts.contains(&inferred) {
    //                             new_delta.insert(inferred.clone());
    //                             self.kg.index_manager.insert(&inferred);
    //                         }
    //                     }
    //                 }
    //             }
    //         }


    //         // delta leeg -> stop
    //         if new_delta.is_empty() {
    //             break;
    //         }

    //         self.trace.push(counts);

    //         // add delta to inferred_so_far
    //         for fact in &new_delta {
    //             all_facts.insert(fact.clone());
    //             inferred_so_far.push(fact.clone());
    //         }

    //         delta = new_delta.iter().cloned().collect();

    //         counts = HashMap::new();

    //     }

    //     inferred_so_far
    // }

    pub fn maintenance_counting(&mut self, added_facts: Vec<Triple>, removed_facts: Vec<Triple>) {
        let mut i_o: HashSet<Triple> = HashSet::new();
        let mut i_n: HashSet<Triple> = HashSet::new();
        let mut i_no: HashSet<Triple> = HashSet::new();
        let mut i_on: HashSet<Triple> = HashSet::new();

        let mut i = 0;

        let mut n_o = self.trace[i].clone();

        let _ = multiset_add(&mut self.trace[i], added_facts);
        let _ = multiset_subtract(&mut self.trace[i], removed_facts);


        let mut n_n = &self.trace[i];

        let mut delta_o: HashSet<Triple>;
        let mut delta_n: HashSet<Triple>;

        loop {

            delta_o = n_o.keys().cloned().collect();
            delta_o = delta_o.difference(&i_o).cloned().collect();


            delta_n = n_n.keys().cloned().collect();
            delta_n = delta_n.difference(&i_n).cloned().collect();


            if delta_o.is_empty() && delta_n.is_empty() {
                break;
            }

            i_o = i_o.union(&delta_o).cloned().collect();
            i_n = i_n.union(&delta_n).cloned().collect();


            let mut diff1: HashSet<Triple> = i_on.difference(&delta_n).cloned().collect();
            let mut diff2: HashSet<Triple> = delta_o.difference(&i_n).cloned().collect();

            i_on = diff1.union(&diff2).cloned().collect();

            diff1 = i_no.difference(&delta_o).cloned().collect();
            diff2 = delta_n.difference(&i_o).cloned().collect();

            i_no = diff1.union(&diff2).cloned().collect();


            if delta_o.difference(&delta_n).next() == None && i_on.is_empty() && i_no.is_empty() {
                break;
            }

            i += 1;

            n_o = self.trace[i].clone();


            let symm_dif: Vec<Triple> = i_o.intersection(&i_n).cloned().collect::<HashSet<Triple>>().difference(&delta_n).cloned().collect();
            let delta: Vec<Triple> = delta_o.intersection(&i_n).cloned().collect::<HashSet<Triple>>().difference(&delta_n).cloned().collect();

            let mut removed = Vec::new();

            for rule in self.kg.rules.clone() {
                let bindings = self.kg.evaluate_rule_with_delta_improved(&rule, &symm_dif, &delta);

                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.kg.dictionary) {
                        for conclusion in &rule.conclusion {
                            let inferred = construct_triple(conclusion, &binding, &mut self.kg.dictionary);

                            removed.push(inferred);
                        }
                    }
                }

                removed.extend(self.kg.evaluate_rule_with_restrictions(&rule, &i_o, &delta_o, &i_on).into_iter());
            }


            let _ = multiset_subtract(&mut self.trace[i], removed);




            let symm_dif: Vec<Triple> = i_n.intersection(&i_o).cloned().collect::<HashSet<Triple>>().difference(&delta_o).cloned().collect();
            let delta: Vec<Triple> = delta_n.intersection(&i_o).cloned().collect::<HashSet<Triple>>().difference(&delta_o).cloned().collect();

            let mut added = Vec::new();

            for rule in self.kg.rules.clone() {
                let bindings = self.kg.evaluate_rule_with_delta_improved(&rule, &symm_dif, &delta);

                for binding in bindings {
                    // check if the binding adheres to the filters of the rule
                    if evaluate_filters(&binding, &rule.filters, &self.kg.dictionary) {
                        for conclusion in &rule.conclusion {
                            let inferred = construct_triple(conclusion, &binding, &mut self.kg.dictionary);

                            added.push(inferred);
                        }
                    }
                }

                added.extend(self.kg.evaluate_rule_with_restrictions(&rule, &i_n, &delta_n, &i_no).into_iter());
            }

            let _ = multiset_add(&mut self.trace[i], added);

            n_n = &self.trace[i];


        }
    }

    pub fn decode_trace(&self) -> Vec<HashMap<String, u32>> {
        let mut trace = Vec::new();

        for counts in &self.trace {

            let mut decoded = HashMap::new();
            for (key, value) in counts {

                let subject = self.kg.dictionary.decode(key.subject).unwrap_or("unknown");
                let predicate = self.kg.dictionary.decode(key.predicate).unwrap_or("unknown");
                let object = self.kg.dictionary.decode(key.object).unwrap_or("unknown");

                decoded.insert(format!("{} {} {}", subject, predicate, object), value.to_owned());
            }

            trace.push(decoded)
        }


        trace
    }

    pub fn decode_triple(&self, triple: &Triple) -> String {
        let subject = self.kg.dictionary.decode(triple.subject).unwrap_or("unknown");
        let predicate = self.kg.dictionary.decode(triple.predicate).unwrap_or("unknown");
        let object = self.kg.dictionary.decode(triple.object).unwrap_or("unknown");

        format!("{} {} {}", subject, predicate, object)
    }

    pub fn print_trace(&self) {
        let decoded = self.decode_trace();
        println!("{:#?}", decoded);
    }
}



pub fn multiset_add(multiset: &mut HashMap<Triple, u32>, added: Vec<Triple>) -> Vec<Triple> {
    let mut new = Vec::new();

    for triple in added {
        if let None = multiset.get(&triple) {
            new.push(triple.clone());
        }

        multiset.entry(triple).and_modify(|count| *count += 1).or_insert(1);
    }

    new
}

pub fn multiset_subtract(multiset: &mut HashMap<Triple, u32>, subtracted: Vec<Triple>) -> Vec<Triple> {
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
    fn multiset_add_test() {
        let mut multiset = HashMap::new();

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
        let mut cmg = CountingMaintenanceGraph::new();

        // println!("Explicit facts: [");

        for i in 0..2 {
        let subject = format!("person{}", i);
        let object = format!("person{}", (i + 1) % 5); // Wraps around to connect the last person with the first

        // println!("    \"{} likes {}\",", subject, object);
        cmg.kg.add_abox_triple(&subject, "likes", &object);
        }

        // println!("]");

        // Add transitivity rule: likes(X, Y) & likes(Y, Z) => likes(X, Z)
        cmg.kg.add_rule(Rule {
        premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("likes")),
                Term::Variable("y".to_string()),
            ),
            (
                Term::Variable("y".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("likes")),
                Term::Variable("z".to_string()),
            ),
        ],
        conclusion: vec![(
            Term::Variable("x".to_string()),
            Term::Constant(cmg.kg.dictionary.clone().encode("likes")),
            Term::Variable("z".to_string()),
        )],
        filters: vec![],
    });

    // Run the optimized inference
    let start = Instant::now();
    let inferred_facts = cmg.semi_naive_counting();
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


    let count = cmg.trace[1].get(&inferred_facts[0]).unwrap().to_owned();
    assert_eq!(count, 1);

    }


    #[test]
    fn semi_naive_counting_graph_test() {
        let mut cmg = CountingMaintenanceGraph::new();

        cmg.kg.add_abox_triple("a", "edge", "b");
        cmg.kg.add_abox_triple("b", "edge", "c");
        cmg.kg.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(cmg.kg.dictionary.clone().encode("edge"));
        let reachable = Term::Constant(cmg.kg.dictionary.encode("reachable"));

        cmg.kg.add_rule(Rule {
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

        cmg.kg.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("reachable")),
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

    let inferred_facts = cmg.semi_naive_counting();

    // cmg.print_trace();

    let all_facts = cmg.kg.index_manager.query(None, None, None);

    assert_eq!(7, all_facts.len());
    assert_eq!(4, inferred_facts.len());
    }

    #[test]
    fn semi_naive_counting_reflexive_test() {
        let mut cmg = CountingMaintenanceGraph::new();

        cmg.kg.add_abox_triple("a", "r", "b");

        let encoded_r = Term::Constant(cmg.kg.dictionary.encode("r"));

        cmg.kg.add_rule(Rule {
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
                    subject: cmg.kg.dictionary.encode("a"),
                    predicate: cmg.kg.dictionary.encode("r"),
                    object: cmg.kg.dictionary.encode("b"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("b"),
                    predicate: cmg.kg.dictionary.encode("r"),
                    object: cmg.kg.dictionary.encode("a"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("a"),
                    predicate: cmg.kg.dictionary.encode("r"),
                    object: cmg.kg.dictionary.encode("b"),
                }, 1)
            ])
        );

        assert_eq!(test_trace, cmg.trace);
    }

    #[test]
    fn maintenance_counting_graph_test() {
        let mut cmg = CountingMaintenanceGraph::new();

        cmg.kg.add_abox_triple("a", "edge", "b");
        cmg.kg.add_abox_triple("b", "edge", "c");
        cmg.kg.add_abox_triple("e", "edge", "d");

        let edge = Term::Constant(cmg.kg.dictionary.clone().encode("edge"));
        let reachable = Term::Constant(cmg.kg.dictionary.encode("reachable"));

        cmg.kg.add_rule(Rule {
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

        cmg.kg.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("edge")),
                Term::Variable("y".to_string()),
            ),
            ],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                Term::Constant(cmg.kg.dictionary.clone().encode("reachable")),
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
        let _ = cmg.semi_naive_counting();

        let removed: Vec<Triple> = vec![
            Triple {
                subject: cmg.kg.dictionary.encode("b"),
                predicate: cmg.kg.dictionary.encode("edge"),
                object: cmg.kg.dictionary.encode("c"),
            },
        ];

        cmg.maintenance_counting(Vec::new(), removed);


        // cmg.print_trace();

        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("a"),
                    predicate: cmg.kg.dictionary.encode("edge"),
                    object: cmg.kg.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.kg.dictionary.encode("e"),
                    predicate: cmg.kg.dictionary.encode("edge"),
                    object: cmg.kg.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("a"),
                    predicate: cmg.kg.dictionary.encode("reachable"),
                    object: cmg.kg.dictionary.encode("b"),
                }, 1),
                (Triple {
                    subject: cmg.kg.dictionary.encode("e"),
                    predicate: cmg.kg.dictionary.encode("reachable"),
                    object: cmg.kg.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(HashMap::new());
        test_trace.push(HashMap::new());

        assert_eq!(test_trace, cmg.trace);
    }

    #[test]
    fn maintenance_counting_small_test() {
        let mut cmg = CountingMaintenanceGraph::new();

        cmg.kg.add_abox_triple("a", "t", "b");
        cmg.kg.add_abox_triple("f", "t", "b");
        cmg.kg.add_abox_triple("c", "t", "d");

        let encoded_r = Term::Constant(cmg.kg.dictionary.encode("r"));
        let encoded_t = Term::Constant(cmg.kg.dictionary.encode("t"));

        cmg.kg.add_rule(Rule {
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
                subject: cmg.kg.dictionary.encode("f"),
                predicate: cmg.kg.dictionary.encode("t"),
                object: cmg.kg.dictionary.encode("b"),
            },
            Triple {
                subject: cmg.kg.dictionary.encode("a"),
                predicate: cmg.kg.dictionary.encode("t"),
                object: cmg.kg.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![
            Triple {
                subject: cmg.kg.dictionary.encode("g"),
                predicate: cmg.kg.dictionary.encode("t"),
                object: cmg.kg.dictionary.encode("d"),
            }
        ];

        cmg.maintenance_counting(added, removed);

        // let all_facts = cmg.kg.index_manager.query(None, None, None);

        // cmg.print_trace();

        let mut test_trace = Vec::new();

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("c"),
                    predicate: cmg.kg.dictionary.encode("t"),
                    object: cmg.kg.dictionary.encode("d"),
                }, 1),
                (Triple {
                    subject: cmg.kg.dictionary.encode("g"),
                    predicate: cmg.kg.dictionary.encode("t"),
                    object: cmg.kg.dictionary.encode("d"),
                }, 1)
            ])
        );

        test_trace.push(
            HashMap::from([
                (Triple {
                    subject: cmg.kg.dictionary.encode("d"),
                    predicate: cmg.kg.dictionary.encode("r"),
                    object: cmg.kg.dictionary.encode("d"),
                }, 2)
            ])
        );

        test_trace.push(HashMap::new());
        // test_trace.push(HashMap::new());

        assert_eq!(test_trace, cmg.trace);
    }

    #[test]
    fn maintenance_counting_reflexive_test() {
        let mut cmg = CountingMaintenanceGraph::new();

        cmg.kg.add_abox_triple("a", "r", "b");

        let encoded_r = Term::Constant(cmg.kg.dictionary.encode("r"));

        cmg.kg.add_rule(Rule {
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
                subject: cmg.kg.dictionary.encode("a"),
                predicate: cmg.kg.dictionary.encode("r"),
                object: cmg.kg.dictionary.encode("b"),
            },
        ];

        let added: Vec<Triple> = vec![];

        cmg.maintenance_counting(added, removed);

        let _ = cmg.kg.index_manager.query(None, None, None);

        // cmg.print_trace();

        let test_trace = vec![HashMap::new(), HashMap::new(), HashMap::new()];

        assert_eq!(test_trace, cmg.trace);
    }
}