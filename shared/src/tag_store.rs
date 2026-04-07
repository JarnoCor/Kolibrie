/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Generic tag store for provenance-annotated triples.
//!
//! Stores provenance tags (from a [`Provenance`] semiring) for triples,
//! using the provenance's operations for disjunction and convergence checks.

use std::collections::HashMap;
use crate::provenance::Provenance;
use crate::triple::Triple;
use crate::dictionary::Dictionary;
use crate::quoted_triple_store::QuotedTripleStore;

/// Stores provenance tags for triples, parameterized by a [`Provenance`] semiring.
///
/// Absence of a triple in the store means it carries the `one()` tag (certain fact).
#[derive(Debug, Clone)]
pub struct TagStore<P: Provenance> {
    tags: HashMap<Triple, P::Tag>,
    provenance: P,
}

impl<P: Provenance> TagStore<P> {
    pub fn new(provenance: P) -> Self {
        Self {
            tags: HashMap::new(),
            provenance,
        }
    }

    /// Get the provenance reference.
    pub fn provenance(&self) -> &P {
        &self.provenance
    }

    /// Get the tag for a triple. Returns `one()` if not explicitly stored (certain fact).
    pub fn get_tag(&self, triple: &Triple) -> P::Tag {
        self.tags.get(triple).cloned().unwrap_or_else(|| self.provenance.one())
    }

    /// Set the tag for a triple. If the tag equals `one()`, it is removed (implicit certainty).
    pub fn set_tag(&mut self, triple: &Triple, tag: P::Tag) {
        if tag == self.provenance.one() {
            self.tags.remove(triple);
        } else {
            self.tags.insert(triple.clone(), tag);
        }
    }

    /// Update a triple's tag using disjunction (⊕) with a new derivation tag.
    ///
    /// This implements the semiring semantics: when the same fact is derived
    /// via multiple paths, the tags are combined with ⊕.
    ///
    /// Returns `true` if the tag changed (not yet saturated).
    pub fn update_disjunction(&mut self, triple: &Triple, new_tag: &P::Tag) -> bool {
        let old_tag = self.get_tag(triple);
        let combined = self.provenance.disjunction(&old_tag, new_tag);
        if self.provenance.is_saturated(&old_tag, &combined) {
            false
        } else {
            self.set_tag(triple, combined);
            true
        }
    }

    /// Check if a triple has an explicit (non-one) tag stored.
    pub fn has_explicit_tag(&self, triple: &Triple) -> bool {
        self.tags.contains_key(triple)
    }

    /// Returns the number of triples with explicit tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Iterate over all (triple, tag) pairs with explicit tags.
    pub fn iter(&self) -> impl Iterator<Item = (&Triple, &P::Tag)> {
        self.tags.iter()
    }

    /// Export tags as RDF-star triples for SPARQL query access.
    ///
    /// For each triple with an explicit tag, generates:
    ///   << s p o >> prob_predicate_id "probability_value"
    ///
    /// Uses `recover_probability` to convert tags to f64 for serialization.
    pub fn encode_as_rdf_star(
        &self,
        dict: &mut Dictionary,
        qt_store: &mut QuotedTripleStore,
    ) -> Vec<Triple> {
        let prob_pred_id = dict.encode("http://www.w3.org/ns/prob#value");
        let mut result = Vec::with_capacity(self.tags.len());

        for (triple, tag) in &self.tags {
            let qt_id = qt_store.encode(triple.subject, triple.predicate, triple.object);
            let prob = self.provenance.recover_probability(tag);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{MinMaxProbability, AddMultProbability, BooleanProvenance};

    fn make_triple(s: u32, p: u32, o: u32) -> Triple {
        Triple { subject: s, predicate: p, object: o }
    }

    #[test]
    fn default_tag_is_one() {
        let store = TagStore::new(MinMaxProbability);
        let tag = store.get_tag(&make_triple(1, 2, 3));
        assert!((tag - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_and_get_tag() {
        let mut store = TagStore::new(MinMaxProbability);
        let t = make_triple(1, 2, 3);
        store.set_tag(&t, 0.7);
        assert!((store.get_tag(&t) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn one_tag_not_stored_explicitly() {
        let mut store = TagStore::new(MinMaxProbability);
        let t = make_triple(1, 2, 3);
        store.set_tag(&t, 1.0);
        assert!(!store.has_explicit_tag(&t));
    }

    #[test]
    fn update_disjunction_minmax() {
        let mut store = TagStore::new(MinMaxProbability);
        let t = make_triple(1, 2, 3);
        store.set_tag(&t, 0.5);

        // max(0.5, 0.8) = 0.8 → changed
        assert!(store.update_disjunction(&t, &0.8));
        assert!((store.get_tag(&t) - 0.8).abs() < 1e-9);

        // max(0.8, 0.6) = 0.8 → no change (saturated)
        assert!(!store.update_disjunction(&t, &0.6));
        assert!((store.get_tag(&t) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn update_disjunction_addmult() {
        let mut store = TagStore::new(AddMultProbability);
        let t = make_triple(1, 2, 3);
        store.set_tag(&t, 0.3);

        // noisy-OR: 0.3 + 0.4 - 0.3*0.4 = 0.7 - 0.12 = 0.58 → changed
        assert!(store.update_disjunction(&t, &0.4));
        assert!((store.get_tag(&t) - 0.58).abs() < 1e-9);
    }

    #[test]
    fn boolean_store_basic() {
        let mut store = TagStore::new(BooleanProvenance);
        let t = make_triple(1, 2, 3);
        store.set_tag(&t, false);

        // false ⊕ true = true → changed
        assert!(store.update_disjunction(&t, &true));
        assert_eq!(store.get_tag(&t), true);

        // true ⊕ true = true → no change
        assert!(!store.update_disjunction(&t, &true));
    }

    #[test]
    fn rdf_star_encoding() {
        let mut store = TagStore::new(MinMaxProbability);
        store.set_tag(&make_triple(1, 2, 3), 0.75);

        let mut dict = Dictionary::new();
        let mut qt_store = QuotedTripleStore::new();
        let triples = store.encode_as_rdf_star(&mut dict, &mut qt_store);

        assert_eq!(triples.len(), 1);
    }
}
