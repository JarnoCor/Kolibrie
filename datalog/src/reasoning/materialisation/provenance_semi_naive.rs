/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Semi-naive materialisation strategy with provenance semiring tags.
//!
//! Extends the classical semi-naive approach with provenance propagation:
//! - For each binding set, looks up premise triple tags from the [`TagStore`]
//! - Combines them via the provenance's conjunction (⊗) operation
//! - Uses the provenance's disjunction (⊕) for multiple derivation paths
//! - Convergence is detected via the provenance's saturation check

use shared::dictionary::Dictionary;
use shared::provenance::Provenance;
use shared::rule::Rule;
use shared::tag_store::TagStore;
use shared::triple::Triple;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use crate::reasoning::convert_string_binding_to_u32;
use crate::reasoning::Reasoner;
use crate::reasoning::materialisation::provenance_infer_generic::{
    ProvenanceInferResult, ProvenanceInferenceStrategy,
};
use crate::reasoning::materialisation::replace_variables_with_bound_values;
use crate::reasoning::rules::{evaluate_filters, join_premise_with_hash_join};

/// Semi-naive materialisation strategy parameterized by a provenance semiring.
struct ProvenanceSemiNaiveStrategy {
    start_idx_for_delta: usize,
}

impl ProvenanceSemiNaiveStrategy {
    /// Find all bindings where at least one premise is matched by delta facts.
    /// Returns bindings along with the matched premise triples (for tag lookup).
    fn find_premise_solutions_with_triples(
        &mut self,
        dict: &Dictionary,
        rule: &Rule,
        all_facts: &[Triple],
        delta_facts: &[Triple],
    ) -> Vec<(BTreeMap<String, String>, Vec<Triple>)> {
        let nr_premises = rule.premise.len();
        let mut results = Vec::new();
        let mut seen_derivations = BTreeSet::new();

        for i in 0..nr_premises {
            let mut current_bindings = vec![BTreeMap::new()];

            // At least one premise matched by delta facts
            current_bindings = join_premise_with_hash_join(
                &rule.premise[i],
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
                    &rule.premise[j],
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
                let matched_triples = resolve_premise_triples(&rule.premise, &u32_binding);
                if seen_derivations.insert((binding.clone(), matched_triples.clone())) {
                    results.push((binding, matched_triples));
                }
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

fn resolve_term(
    term: &shared::terms::Term,
    bindings: &std::collections::HashMap<String, u32>,
) -> Option<u32> {
    match term {
        shared::terms::Term::Variable(v) => bindings.get(v).copied(),
        shared::terms::Term::Constant(c) => Some(*c),
        shared::terms::Term::QuotedTriple(_) => None,
    }
}

impl<P: Provenance> ProvenanceInferenceStrategy<P> for ProvenanceSemiNaiveStrategy {
    fn infer_round(
        &mut self,
        dictionary: &mut Dictionary,
        rules: &[Rule],
        all_facts: &[Triple],
        known_facts: &HashSet<Triple>,
        tag_store: &mut TagStore<P>,
    ) -> ProvenanceInferResult {
        let mut new_facts: HashSet<Triple> = HashSet::new();
        let mut tag_changed = false;
        let provenance = tag_store.provenance().clone();

        let end_idx = all_facts.len();
        let delta_facts = &all_facts[self.start_idx_for_delta..end_idx];
        self.start_idx_for_delta = end_idx;

        for rule in rules {
            let binding_sets =
                self.find_premise_solutions_with_triples(dictionary, rule, all_facts, delta_facts);

            for (binding, matched_triples) in &binding_sets {
                let u32_binding = convert_string_binding_to_u32(binding, dictionary);

                if !evaluate_filters(&u32_binding, &rule.filters, dictionary) {
                    continue;
                }

                // Combine premise tags via conjunction (⊗)
                let conclusion_tag = matched_triples
                    .iter()
                    .map(|t| tag_store.get_tag(t))
                    .fold(provenance.one(), |acc, tag| {
                        provenance.conjunction(&acc, &tag)
                    });

                // Skip if the conclusion tag is zero (impossible derivation)
                if conclusion_tag == provenance.zero() {
                    continue;
                }

                // Generate conclusion triples
                for conclusion in &rule.conclusion {
                    let inferred_fact =
                        replace_variables_with_bound_values(conclusion, &u32_binding, dictionary);

                    let is_new = !known_facts.contains(&inferred_fact);

                    if is_new && !new_facts.contains(&inferred_fact) {
                        // Brand new triple: set its tag directly
                        tag_store.set_tag(&inferred_fact, conclusion_tag.clone());
                        new_facts.insert(inferred_fact);
                    } else {
                        // Already seen (this round or earlier): disjunction (⊕)
                        if tag_store.update_disjunction(&inferred_fact, &conclusion_tag) {
                            if !is_new {
                                // Only flag tag_changed for facts from previous rounds
                                tag_changed = true;
                            }
                        }
                    }
                }
            }
        }

        ProvenanceInferResult {
            new_facts,
            tag_changed,
        }
    }
}

impl Reasoner {
    /// Run provenance-based semi-naive materialisation.
    ///
    /// Infers new facts using plain rules with provenance tag propagation.
    /// The provenance semiring determines how tags are combined:
    /// - Conjunction (⊗) for joining premise tags
    /// - Disjunction (⊕) for alternative derivation paths
    pub fn infer_new_facts_with_provenance<P: Provenance>(
        &mut self,
        provenance: P,
    ) -> (Vec<Triple>, TagStore<P>) {
        let mut tag_store = TagStore::new(provenance.clone());

        // Seed tag store from probability seeds.
        // Sort by triple for deterministic ID assignment (needed by TopKProofs which
        // uses the ID as a variable index in the probability table).
        // Zero cost for other provenances — their tag_from_probability_with_id ignores id.
        let mut seeds: Vec<(&Triple, f64)> =
            self.probability_seeds.iter().map(|(t, &p)| (t, p)).collect();
        seeds.sort_by_key(|(t, _)| *t);
        for (id, (triple, prob)) in seeds.iter().enumerate() {
            let tag = provenance.tag_from_probability_with_id(*prob, id);
            tag_store.set_tag(triple, tag);
        }

        let new_facts = self.infer_with_provenance_strategy(
            ProvenanceSemiNaiveStrategy { start_idx_for_delta: 0 },
            &mut tag_store,
        );

        (new_facts, tag_store)
    }
}
