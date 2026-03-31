/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::collections::HashMap;
use crate::triple::Triple;
use crate::dictionary::Dictionary;
use crate::quoted_triple_store::QuotedTripleStore;

/// Stores probability values for triples separately from the triple store.
///
/// Design rationale: The `Triple` struct derives Hash/Eq/Ord and is used
/// pervasively in HashSet/HashMap/Vec and the 6-way UnifiedIndex. Adding
/// f64 would break these semantics. Instead, probabilities live in a
/// parallel store keyed by Triple.
///
/// Absence of a triple in this store means probability 1.0 (certain fact).
#[derive(Debug, Clone)]
pub struct ProbabilityStore {
    triple_prob: HashMap<Triple, f64>,
}

/// Epsilon for probability convergence detection
pub const PROB_EPSILON: f64 = 1e-9;

impl ProbabilityStore {
    pub fn new() -> Self {
        Self {
            triple_prob: HashMap::new(),
        }
    }

    /// Store or update the probability of a triple. Clamps to (0.0, 1.0].
    pub fn set_probability(&mut self, triple: &Triple, prob: f64) {
        let clamped = prob.max(0.0).min(1.0);
        if (clamped - 1.0).abs() < PROB_EPSILON {
            // Probability 1.0 = certain, no need to store
            self.triple_prob.remove(triple);
        } else {
            self.triple_prob.insert(triple.clone(), clamped);
        }
    }

    /// Get the probability of a triple. Returns 1.0 if not stored (certain fact).
    pub fn get_probability(&self, triple: &Triple) -> f64 {
        self.triple_prob.get(triple).copied().unwrap_or(1.0)
    }

    /// Update probability using max semantics (monotone convergence).
    /// Returns true if the probability actually changed.
    pub fn update_max(&mut self, triple: &Triple, new_prob: f64) -> bool {
        let old_prob = self.get_probability(triple);
        if new_prob > old_prob + PROB_EPSILON {
            self.set_probability(triple, new_prob);
            true
        } else {
            false
        }
    }

    /// Returns an iterator over triples with probability >= threshold.
    pub fn triples_above_threshold(&self, threshold: f64) -> Vec<&Triple> {
        if threshold <= PROB_EPSILON {
            // All triples qualify (including certain ones not in store)
            return self.triple_prob.keys().collect();
        }
        self.triple_prob
            .iter()
            .filter(|(_, &p)| p >= threshold)
            .map(|(t, _)| t)
            .collect()
    }

    /// Returns the number of triples with explicit probabilities.
    pub fn len(&self) -> usize {
        self.triple_prob.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triple_prob.is_empty()
    }

    /// Check if a triple has an explicit (non-certain) probability stored.
    pub fn has_explicit_probability(&self, triple: &Triple) -> bool {
        self.triple_prob.contains_key(triple)
    }

    /// Iterate over all (triple, probability) pairs with explicit probabilities.
    pub fn iter(&self) -> impl Iterator<Item = (&Triple, &f64)> {
        self.triple_prob.iter()
    }

    /// Export probabilities as RDF-star triples for SPARQL query access.
    ///
    /// For each triple with an explicit probability, generates:
    ///   << s p o >> prob_predicate_id "probability_value"
    ///
    /// The probability value is stored as a literal string in the dictionary.
    pub fn encode_as_rdf_star(
        &self,
        dict: &mut Dictionary,
        qt_store: &mut QuotedTripleStore,
    ) -> Vec<Triple> {
        let prob_pred_id = dict.encode("http://www.w3.org/ns/prob#value");
        let mut result = Vec::with_capacity(self.triple_prob.len());

        for (triple, &prob) in &self.triple_prob {
            let qt_id = qt_store.encode(triple.subject, triple.predicate, triple.object);
            let prob_literal = format!("\"{}\"^^<http://www.w3.org/2001/XMLSchema#double>", prob);
            let prob_obj_id = dict.encode(&prob_literal);

            result.push(Triple {
                subject: qt_id,
                predicate: prob_pred_id,
                object: prob_obj_id,
            });
        }

        result
    }

    /// Import probabilities from RDF-star triples.
    ///
    /// Looks for triples of the form: << s p o >> prob_predicate_id literal_prob
    /// and extracts the probability value.
    pub fn import_from_rdf_star(
        &mut self,
        triples: &[Triple],
        dict: &Dictionary,
        qt_store: &QuotedTripleStore,
        prob_predicate_id: u32,
    ) {
        use crate::quoted_triple_store::is_quoted_triple_id;

        for triple in triples {
            if triple.predicate != prob_predicate_id {
                continue;
            }
            if !is_quoted_triple_id(triple.subject) {
                continue;
            }

            if let Some((s, p, o)) = qt_store.decode(triple.subject) {
                if let Some(prob_str) = dict.decode(triple.object) {
                    if let Some(prob) = parse_probability_literal(prob_str) {
                        self.set_probability(
                            &Triple { subject: s, predicate: p, object: o },
                            prob,
                        );
                    }
                }
            }
        }
    }
}

impl Default for ProbabilityStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a probability value from an RDF literal string.
/// Handles both plain numbers and typed literals like "0.7"^^xsd:double.
fn parse_probability_literal(s: &str) -> Option<f64> {
    // Try plain float first
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    // Try typed literal: "0.7"^^<...>
    if let Some(start) = s.find('"') {
        if let Some(end) = s[start + 1..].find('"') {
            let num_str = &s[start + 1..start + 1 + end];
            return num_str.parse::<f64>().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: u32, p: u32, o: u32) -> Triple {
        Triple { subject: s, predicate: p, object: o }
    }

    #[test]
    fn test_default_probability_is_certain() {
        let store = ProbabilityStore::new();
        assert_eq!(store.get_probability(&make_triple(1, 2, 3)), 1.0);
    }

    #[test]
    fn test_set_and_get() {
        let mut store = ProbabilityStore::new();
        let t = make_triple(1, 2, 3);
        store.set_probability(&t, 0.7);
        assert!((store.get_probability(&t) - 0.7).abs() < PROB_EPSILON);
    }

    #[test]
    fn test_clamp_bounds() {
        let mut store = ProbabilityStore::new();
        let t = make_triple(1, 2, 3);

        store.set_probability(&t, 1.5);
        assert!((store.get_probability(&t) - 1.0).abs() < PROB_EPSILON);

        store.set_probability(&t, -0.5);
        assert!((store.get_probability(&t) - 0.0).abs() < PROB_EPSILON);
    }

    #[test]
    fn test_update_max() {
        let mut store = ProbabilityStore::new();
        let t = make_triple(1, 2, 3);
        store.set_probability(&t, 0.5);

        assert!(store.update_max(&t, 0.8)); // should update
        assert!((store.get_probability(&t) - 0.8).abs() < PROB_EPSILON);

        assert!(!store.update_max(&t, 0.6)); // should not update (lower)
        assert!((store.get_probability(&t) - 0.8).abs() < PROB_EPSILON);
    }

    #[test]
    fn test_triples_above_threshold() {
        let mut store = ProbabilityStore::new();
        store.set_probability(&make_triple(1, 2, 3), 0.3);
        store.set_probability(&make_triple(4, 5, 6), 0.7);
        store.set_probability(&make_triple(7, 8, 9), 0.9);

        let above_05 = store.triples_above_threshold(0.5);
        assert_eq!(above_05.len(), 2);
    }

    #[test]
    fn test_certain_probability_not_stored() {
        let mut store = ProbabilityStore::new();
        let t = make_triple(1, 2, 3);
        store.set_probability(&t, 1.0);
        assert!(!store.has_explicit_probability(&t));
        assert_eq!(store.get_probability(&t), 1.0);
    }

    #[test]
    fn test_parse_probability_literal() {
        assert_eq!(parse_probability_literal("0.7"), Some(0.7));
        assert_eq!(
            parse_probability_literal("\"0.7\"^^<http://www.w3.org/2001/XMLSchema#double>"),
            Some(0.7)
        );
        assert_eq!(parse_probability_literal("not_a_number"), None);
    }

    #[test]
    fn test_rdf_star_roundtrip() {
        let mut store = ProbabilityStore::new();
        let t = make_triple(1, 2, 3);
        store.set_probability(&t, 0.75);

        let mut dict = Dictionary::new();
        let mut qt_store = QuotedTripleStore::new();

        let rdf_star_triples = store.encode_as_rdf_star(&mut dict, &mut qt_store);
        assert_eq!(rdf_star_triples.len(), 1);

        let prob_pred_id = dict.encode("http://www.w3.org/ns/prob#value");
        let mut store2 = ProbabilityStore::new();
        store2.import_from_rdf_star(&rdf_star_triples, &dict, &qt_store, prob_pred_id);

        assert!((store2.get_probability(&t) - 0.75).abs() < PROB_EPSILON);
    }
}
