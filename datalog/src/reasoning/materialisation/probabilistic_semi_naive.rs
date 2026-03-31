/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use shared::dictionary::Dictionary;
use shared::probabilistic_rule::{combine_probabilities, ProbabilisticRule};
use shared::probability_store::ProbabilityStore;
use shared::triple::Triple;
use std::collections::{BTreeMap, HashSet};
use crate::reasoning::convert_string_binding_to_u32;
use crate::reasoning::Reasoner;
use crate::reasoning::materialisation::probabilistic_infer_generic::{
    ProbInferResult, ProbabilisticInferenceStrategy,
};
use crate::reasoning::materialisation::replace_variables_with_bound_values;
use crate::reasoning::rules::{evaluate_filters, join_premise_with_hash_join};

/// Semi-naive materialisation strategy with probability propagation.
///
/// Extends the classical semi-naive approach:
/// - For each binding set, looks up premise triple probabilities
/// - Combines them via the rule's `ProbCombination` mode
/// - Applies threshold pruning (skip low-confidence conclusions)
/// - Uses max-probability semantics for multiple derivation paths
///   (monotone convergence guaranteed since probabilities only increase)
struct ProbabilisticSemiNaiveStrategy {
    start_idx_for_delta: usize,
}

impl ProbabilisticSemiNaiveStrategy {
    /// Find all bindings where at least one premise is matched by delta facts.
    /// Returns bindings along with the matched premise triples (for probability lookup).
    fn find_premise_solutions_with_triples(
        &mut self,
        dict: &Dictionary,
        rule: &ProbabilisticRule,
        all_facts: &[Triple],
        delta_facts: &[Triple],
    ) -> Vec<(BTreeMap<String, String>, Vec<Triple>)> {
        let nr_premises = rule.rule.premise.len();
        let mut results = Vec::new();

        for i in 0..nr_premises {
            let mut current_bindings = vec![BTreeMap::new()];

            // At least one premise matched by delta facts
            current_bindings = join_premise_with_hash_join(
                &rule.rule.premise[i],
                delta_facts,
                current_bindings,
                dict,
            );

            // Join remaining premises with all facts
            for j in 0..nr_premises {
                if j == i {
                    continue;
                }
                current_bindings = join_premise_with_hash_join(
                    &rule.rule.premise[j],
                    all_facts,
                    current_bindings,
                    dict,
                );
                if current_bindings.is_empty() {
                    break;
                }
            }

            for binding in current_bindings {
                // Reconstruct matched premise triples from the binding
                let u32_binding = convert_string_binding_to_u32(&binding, dict);
                let matched_triples = resolve_premise_triples(&rule.rule.premise, &u32_binding);
                results.push((binding, matched_triples));
            }
        }

        results
    }
}

/// Resolve premise patterns to concrete triples using the current binding.
fn resolve_premise_triples(
    premises: &[shared::terms::TriplePattern],
    bindings: &std::collections::HashMap<String, u32>,
) -> Vec<Triple> {
    premises
        .iter()
        .filter_map(|pat| {
            let s = resolve_term(&pat.0, bindings)?;
            let p = resolve_term(&pat.1, bindings)?;
            let o = resolve_term(&pat.2, bindings)?;
            Some(Triple { subject: s, predicate: p, object: o })
        })
        .collect()
}

fn resolve_term(term: &shared::terms::Term, bindings: &std::collections::HashMap<String, u32>) -> Option<u32> {
    match term {
        shared::terms::Term::Variable(v) => bindings.get(v).copied(),
        shared::terms::Term::Constant(c) => Some(*c),
        shared::terms::Term::QuotedTriple(_) => None,
    }
}

impl ProbabilisticInferenceStrategy for ProbabilisticSemiNaiveStrategy {
    fn infer_round(
        &mut self,
        dictionary: &mut Dictionary,
        rules: &[ProbabilisticRule],
        all_facts: &[Triple],
        known_facts: &HashSet<Triple>,
        prob_store: &mut ProbabilityStore,
    ) -> ProbInferResult {
        let mut new_facts: HashSet<Triple> = HashSet::new();
        let mut probability_changed = false;

        let end_idx = all_facts.len();
        let delta_facts = &all_facts[self.start_idx_for_delta..end_idx];
        self.start_idx_for_delta = end_idx;

        for rule in rules {
            let binding_sets =
                self.find_premise_solutions_with_triples(dictionary, rule, all_facts, delta_facts);

            for (binding, matched_triples) in &binding_sets {
                let u32_binding = convert_string_binding_to_u32(binding, dictionary);

                if !evaluate_filters(&u32_binding, &rule.rule.filters, dictionary) {
                    continue;
                }

                // Collect probabilities of matched premise triples
                let premise_probs: Vec<f64> = matched_triples
                    .iter()
                    .map(|t| prob_store.get_probability(t))
                    .collect();

                // Combine probabilities according to the rule's combination mode
                let conclusion_prob = combine_probabilities(
                    &premise_probs,
                    &rule.combination,
                    rule.rule_confidence,
                );

                // Threshold pruning: skip low-confidence conclusions
                if conclusion_prob < rule.threshold {
                    continue;
                }

                // Generate conclusion triples
                for conclusion in &rule.rule.conclusion {
                    let inferred_fact =
                        replace_variables_with_bound_values(conclusion, &u32_binding, dictionary);

                    let is_new = !known_facts.contains(&inferred_fact);

                    if is_new && !new_facts.contains(&inferred_fact) {
                        // Brand new triple: set its probability directly
                        prob_store.set_probability(&inferred_fact, conclusion_prob);
                        new_facts.insert(inferred_fact);
                    } else if is_new {
                        // New this round but already derived via another path: max semantics
                        let old = prob_store.get_probability(&inferred_fact);
                        if conclusion_prob > old {
                            prob_store.set_probability(&inferred_fact, conclusion_prob);
                        }
                    } else {
                        // Already known from previous rounds: max-probability update
                        if prob_store.update_max(&inferred_fact, conclusion_prob) {
                            probability_changed = true;
                        }
                    }
                }
            }
        }

        ProbInferResult {
            new_facts,
            probability_changed,
        }
    }
}

impl Reasoner {
    /// Run probabilistic semi-naive materialisation.
    /// Infers new facts using probabilistic rules with probability propagation.
    pub fn infer_new_facts_probabilistic_semi_naive(&mut self) -> Vec<Triple> {
        self.infer_with_probabilistic_strategy(ProbabilisticSemiNaiveStrategy {
            start_idx_for_delta: 0,
        })
    }
}
