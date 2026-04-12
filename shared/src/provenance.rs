/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Provenance semiring framework for probabilistic reasoning.
//!
//! Based on the theory of provenance semirings (Green et al., PODS 2007) and
//! extended provenance structures (Scallop, Li et al., 2023).
//!
//! In a provenance semiring, each derived fact is annotated with a *tag* from
//! a semiring (T, ⊕, ⊗, 0, 1). Union of derivation paths uses ⊕ (disjunction),
//! and joining premises uses ⊗ (conjunction). The provenance is applied
//! *uniformly* to the entire program — not per-rule.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Trait defining a provenance semiring for annotated datalog evaluation.
///
/// Implementors provide:
/// - A tag space (`Tag`) with identity elements (`zero`, `one`)
/// - Semiring operations: `disjunction` (⊕) and `conjunction` (⊗)
/// - Negation support (`negate`, ⊖) for stratified negation
/// - Saturation (`saturate`, ⊜) for fixpoint convergence detection
/// - Conversion to/from probability values for I/O
///
/// # Semiring Laws (expected to hold)
/// - ⊕ is commutative, associative, with identity `zero`
/// - ⊗ is commutative, associative, with identity `one`
/// - ⊗ distributes over ⊕
/// - `zero` is an annihilator for ⊗: `a ⊗ zero = zero`
pub trait Provenance: Clone + 'static {
    /// The type of annotations (tags) on derived facts.
    type Tag: Clone + PartialEq + std::fmt::Debug;

    /// Additive identity (false / empty derivation).
    fn zero(&self) -> Self::Tag;

    /// Multiplicative identity (true / certain fact).
    fn one(&self) -> Self::Tag;

    /// Disjunction (⊕): combines tags from alternative derivation paths.
    /// Used when the same fact is derived via multiple rule applications.
    fn disjunction(&self, a: &Self::Tag, b: &Self::Tag) -> Self::Tag;

    /// Conjunction (⊗): combines tags from joined premises.
    /// Used when multiple premise triples contribute to a single derivation.
    fn conjunction(&self, a: &Self::Tag, b: &Self::Tag) -> Self::Tag;

    /// Negation (⊖): used for stratified negation-as-failure.
    fn negate(&self, a: &Self::Tag) -> Self::Tag;

    /// Saturation (⊜): applied at fixpoint to stabilize tags.
    /// For simple provenances this is typically the identity function.
    fn saturate(&self, a: &Self::Tag) -> Self::Tag;

    /// Convert a probability value (from input data) to a tag.
    fn tag_from_probability(&self, prob: f64) -> Self::Tag;

    /// Convert a probability and a unique input-fact index to a tag.
    /// Default ignores `_id` and delegates to `tag_from_probability`.
    /// Override for provenances that track variable identities (e.g. TopKProofs).
    fn tag_from_probability_with_id(&self, prob: f64, _id: usize) -> Self::Tag {
        self.tag_from_probability(prob)
    }

    /// Recover a probability value from a tag (for output / RDF-star encoding).
    fn recover_probability(&self, tag: &Self::Tag) -> f64;

    /// Check whether a tag has converged (old ≈ new after saturation).
    /// Returns `true` if the tag is considered stable.
    fn is_saturated(&self, old: &Self::Tag, new: &Self::Tag) -> bool;
}

// ---------------------------------------------------------------------------
// Concrete provenance implementations
// ---------------------------------------------------------------------------

/// Epsilon for floating-point convergence comparisons.
const PROB_EPSILON: f64 = 1e-9;

/// Min-max probability provenance.
///
/// - Tag: f64 probability in [0, 1]
/// - ⊕ = max (optimistic disjunction)
/// - ⊗ = min (conservative conjunction)
///
/// This is the simplest monotone provenance and corresponds to the
/// possibilistic / fuzzy semantics commonly used in fuzzy Datalog.
#[derive(Debug, Clone)]
pub struct MinMaxProbability;

impl Provenance for MinMaxProbability {
    type Tag = f64;

    fn zero(&self) -> f64 { 0.0 }
    fn one(&self) -> f64 { 1.0 }

    fn disjunction(&self, a: &f64, b: &f64) -> f64 {
        a.max(*b)
    }

    fn conjunction(&self, a: &f64, b: &f64) -> f64 {
        a.min(*b)
    }

    fn negate(&self, a: &f64) -> f64 {
        1.0 - a
    }

    fn saturate(&self, a: &f64) -> f64 {
        *a
    }

    fn tag_from_probability(&self, prob: f64) -> f64 {
        prob.clamp(0.0, 1.0)
    }

    fn recover_probability(&self, tag: &f64) -> f64 {
        *tag
    }

    fn is_saturated(&self, old: &f64, new: &f64) -> bool {
        (old - new).abs() < PROB_EPSILON
    }
}

/// Add-multiply probability provenance.
///
/// - Tag: f64 probability in [0, 1]
/// - ⊕ = noisy-OR (inclusion-exclusion): P(A∨B) = P(A) + P(B) - P(A)·P(B)
/// - ⊗ = multiplication: a * b
///
/// Models independent probabilistic events. This is the standard
/// probabilistic semiring used in ProbDatalog / PRRDF and corresponds to
/// Scallop's `add-mult-prob` provenance. The noisy-OR formula is exact for
/// independent events and naturally bounded in [0, 1] without clamping.
#[derive(Debug, Clone)]
pub struct AddMultProbability;

impl Provenance for AddMultProbability {
    type Tag = f64;

    fn zero(&self) -> f64 { 0.0 }
    fn one(&self) -> f64 { 1.0 }

    fn disjunction(&self, a: &f64, b: &f64) -> f64 {
        a + b - a * b
    }

    fn conjunction(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }

    fn negate(&self, a: &f64) -> f64 {
        1.0 - a
    }

    fn saturate(&self, a: &f64) -> f64 {
        *a
    }

    fn tag_from_probability(&self, prob: f64) -> f64 {
        prob.clamp(0.0, 1.0)
    }

    fn recover_probability(&self, tag: &f64) -> f64 {
        *tag
    }

    fn is_saturated(&self, old: &f64, new: &f64) -> bool {
        (old - new).abs() < PROB_EPSILON
    }
}

/// Boolean provenance (classical two-valued logic).
///
/// - Tag: bool
/// - ⊕ = OR
/// - ⊗ = AND
///
/// Degenerate case: no probability tracking. Equivalent to classical
/// datalog materialisation. Useful for testing and as a baseline.
#[derive(Debug, Clone)]
pub struct BooleanProvenance;

impl Provenance for BooleanProvenance {
    type Tag = bool;

    fn zero(&self) -> bool { false }
    fn one(&self) -> bool { true }

    fn disjunction(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn conjunction(&self, a: &bool, b: &bool) -> bool {
        *a && *b
    }

    fn negate(&self, a: &bool) -> bool {
        !a
    }

    fn saturate(&self, a: &bool) -> bool {
        *a
    }

    fn tag_from_probability(&self, prob: f64) -> bool {
        prob > 0.0
    }

    fn recover_probability(&self, tag: &bool) -> f64 {
        if *tag { 1.0 } else { 0.0 }
    }

    fn is_saturated(&self, old: &bool, new: &bool) -> bool {
        old == new
    }
}

// ---------------------------------------------------------------------------
// TopKProofs provenance
// ---------------------------------------------------------------------------

/// A proof is a set of input-variable IDs (all must hold simultaneously).
pub type Proof = BTreeSet<u32>;
/// A tag is at most k proofs, ranked by descending probability.
pub type TopKTag = Vec<Proof>;

/// Top-K proof-tracking provenance.
///
/// ⊕ = union of proof lists (keep top-k by probability, deduplicate)
/// ⊗ = Cartesian product of proofs (merge variable ID sets)
/// `recover_probability` = WMC via inclusion-exclusion over retained proofs
///
/// # Probability table (`prob_table`)
///
/// `prob_table` is an `Arc<Mutex<Vec<f64>>>`. The shared reference (`Arc`) means
/// that all clones of a `TopKProofs` instance consult and update the **same** table,
/// which is essential because the reasoner clones the provenance before seeding.
/// Interior mutability (`Mutex`) lets `tag_from_probability_with_id` write into the
/// table through `&self` (as required by the `Provenance` trait).
///
/// **Invariant**: `prob_table[i]` is the probability of the input fact assigned ID `i`
/// during seeding. The table must be fully populated (all seeds written) before any
/// call to `recover_probability` or proof-sorting inside `disjunction`/`conjunction`.
/// After seeding, the table is read-only.
///
/// # k constraint
///
/// `k` must be in `[1, 63]`. The upper bound is an **implementation detail**:
/// `recover_probability` uses a `u64` bitmask to iterate over all 2^m subsets of
/// retained proofs (inclusion-exclusion), which requires `m ≤ 63`. This is not a
/// theoretical semiring constraint — the semiring is well-defined for any k.
/// In practice k ≤ 10 is recommended to keep WMC (O(2^k)) fast.
#[derive(Debug, Clone)]
pub struct TopKProofs {
    pub k: usize,
    prob_table: Arc<Mutex<Vec<f64>>>,
}

impl TopKProofs {
    pub fn new(k: usize) -> Self {
        // k <= 63 is an implementation limit: recover_probability uses a u64 bitmask
        // to enumerate 2^m subsets, requiring m <= 63. This is NOT a semiring constraint.
        // Use assert! (not debug_assert!) so release builds also catch invalid k.
        assert!(k >= 1 && k <= 63, "k must be in [1, 63]: u64 bitmask limit in recover_probability");
        Self { k, prob_table: Arc::new(Mutex::new(Vec::new())) }
    }
}

fn proof_prob(proof: &Proof, table: &[f64]) -> f64 {
    proof.iter().map(|&v| *table.get(v as usize).unwrap_or(&1.0)).product()
}

impl Provenance for TopKProofs {
    type Tag = TopKTag;

    fn zero(&self) -> TopKTag { vec![] }
    fn one(&self) -> TopKTag { vec![BTreeSet::new()] }

    /// ⊕ = union, dedup, sort by descending probability, truncate to k
    fn disjunction(&self, a: &TopKTag, b: &TopKTag) -> TopKTag {
        let mut merged: Vec<Proof> = a.iter().chain(b.iter()).cloned().collect();
        merged.sort();
        merged.dedup();
        let table = self.prob_table.lock().unwrap();
        merged.sort_by(|p, q|
            proof_prob(q, &table).partial_cmp(&proof_prob(p, &table))
                .unwrap_or(std::cmp::Ordering::Equal));
        drop(table);
        merged.truncate(self.k);
        merged
    }

    /// ⊗ = Cartesian product — merge variable ID sets (BTreeSet handles shared IDs)
    fn conjunction(&self, a: &TopKTag, b: &TopKTag) -> TopKTag {
        if a.is_empty() || b.is_empty() { return vec![]; }
        let mut result: Vec<Proof> = a.iter().flat_map(|pa|
            b.iter().map(move |pb| pa.iter().chain(pb.iter()).copied().collect())
        ).collect();
        result.sort();
        result.dedup();
        let table = self.prob_table.lock().unwrap();
        result.sort_by(|p, q|
            proof_prob(q, &table).partial_cmp(&proof_prob(p, &table))
                .unwrap_or(std::cmp::Ordering::Equal));
        drop(table);
        result.truncate(self.k);
        result
    }

    /// Negation is **not supported** for proof-tracking provenance.
    ///
    /// `TopKProofs` is designed for **positive programs only** (no FILTER NOT EXISTS,
    /// no negation-as-failure). DNF proof sets have no natural complement that remains
    /// a finite proof set; a "negated proof" would require complement WMC which is
    /// exponential in the number of variables.
    ///
    /// If stratified negation is required, use `BooleanProvenance`, `MinMaxProbability`,
    /// or `AddMultProbability`. The current materialisation flow in
    /// `provenance_semi_naive.rs` does not call `negate()`, but the `Provenance` trait
    /// requires it — hence the explicit panic to catch any future accidental use.
    fn negate(&self, _a: &TopKTag) -> TopKTag {
        unimplemented!("negation is not defined for TopKProofs (positive programs only); \
                        use BooleanProvenance, MinMaxProbability, or AddMultProbability \
                        for programs with stratified negation")
    }

    fn saturate(&self, a: &TopKTag) -> TopKTag { a.clone() }

    fn tag_from_probability(&self, prob: f64) -> TopKTag {
        let mut table = self.prob_table.lock().unwrap();
        let id = table.len() as u32;
        table.push(prob.clamp(0.0, 1.0));
        let mut proof = BTreeSet::new();
        proof.insert(id);
        vec![proof]
    }

    fn tag_from_probability_with_id(&self, prob: f64, id: usize) -> TopKTag {
        let mut table = self.prob_table.lock().unwrap();
        if id >= table.len() { table.resize(id + 1, 0.0); }
        table[id] = prob.clamp(0.0, 1.0);
        drop(table);
        let mut proof = BTreeSet::new();
        proof.insert(id as u32);
        vec![proof]
    }

    /// WMC via inclusion-exclusion over the DNF of the **retained** proofs. O(2^k).
    ///
    /// P(η₁ ∨ … ∨ ηₘ) = Σ_{S⊆{0..m}, S≠∅} (-1)^(|S|+1) · Π_{v ∈ ∪_{i∈S} ηᵢ} P(v)
    ///
    /// **Exactness caveat 1 — truncation**: if k < (total derivation paths), lower-ranked
    /// proofs were discarded during `⊕` truncation and are not counted — the result is
    /// then a conservative lower-bound approximation, not globally exact WMC. Choose k
    /// large enough to retain every proof for exact results (at O(2^k) WMC cost).
    ///
    /// **Exactness caveat 2 — redundant proof slots**: proofs are ranked by raw
    /// `proof_prob` (product of variable probabilities). This can waste scarce k slots
    /// on subsumed proofs — e.g., both `{x}` and `{x, y}` may be retained even though
    /// `{x}` already covers the mass of `{x, y}`. The WMC result is still correct for
    /// the retained proofs, but the lower bound can be weaker than if subsumption-pruning
    /// were applied. This is a known approximation tradeoff in the current implementation.
    fn recover_probability(&self, tag: &TopKTag) -> f64 {
        if tag.is_empty() { return 0.0; }
        let table = self.prob_table.lock().unwrap();
        let m = tag.len();
        let mut total = 0.0_f64;
        for mask in 1u64..(1u64 << m) {
            let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
            let vars: BTreeSet<u32> = (0..m)
                .filter(|&i| mask & (1 << i) != 0)
                .flat_map(|i| tag[i].iter().copied())
                .collect();
            let prod: f64 = vars.iter()
                .map(|&v| *table.get(v as usize).unwrap_or(&1.0))
                .product();
            total += sign * prod;
        }
        total.clamp(0.0, 1.0)
    }

    fn is_saturated(&self, old: &TopKTag, new: &TopKTag) -> bool { old == new }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Semiring law tests for MinMaxProbability ---

    #[test]
    fn minmax_identities() {
        let p = MinMaxProbability;
        let a = 0.7;
        // ⊕ identity: a ⊕ 0 = a
        assert!((p.disjunction(&a, &p.zero()) - a).abs() < PROB_EPSILON);
        // ⊗ identity: a ⊗ 1 = a
        assert!((p.conjunction(&a, &p.one()) - a).abs() < PROB_EPSILON);
    }

    #[test]
    fn minmax_annihilator() {
        let p = MinMaxProbability;
        // a ⊗ 0 = 0
        assert!((p.conjunction(&0.7, &p.zero()) - p.zero()).abs() < PROB_EPSILON);
    }

    #[test]
    fn minmax_commutativity() {
        let p = MinMaxProbability;
        let (a, b) = (0.3, 0.8);
        assert!((p.disjunction(&a, &b) - p.disjunction(&b, &a)).abs() < PROB_EPSILON);
        assert!((p.conjunction(&a, &b) - p.conjunction(&b, &a)).abs() < PROB_EPSILON);
    }

    #[test]
    fn minmax_associativity() {
        let p = MinMaxProbability;
        let (a, b, c) = (0.3, 0.5, 0.8);
        let lhs = p.disjunction(&p.disjunction(&a, &b), &c);
        let rhs = p.disjunction(&a, &p.disjunction(&b, &c));
        assert!((lhs - rhs).abs() < PROB_EPSILON);

        let lhs = p.conjunction(&p.conjunction(&a, &b), &c);
        let rhs = p.conjunction(&a, &p.conjunction(&b, &c));
        assert!((lhs - rhs).abs() < PROB_EPSILON);
    }

    // --- Semiring law tests for AddMultProbability ---

    #[test]
    fn addmult_identities() {
        let p = AddMultProbability;
        let a = 0.7;
        assert!((p.disjunction(&a, &p.zero()) - a).abs() < PROB_EPSILON);
        assert!((p.conjunction(&a, &p.one()) - a).abs() < PROB_EPSILON);
    }

    #[test]
    fn addmult_annihilator() {
        let p = AddMultProbability;
        assert!((p.conjunction(&0.7, &p.zero()) - p.zero()).abs() < PROB_EPSILON);
    }

    #[test]
    fn addmult_commutativity() {
        let p = AddMultProbability;
        let (a, b) = (0.3, 0.4);
        assert!((p.disjunction(&a, &b) - p.disjunction(&b, &a)).abs() < PROB_EPSILON);
        assert!((p.conjunction(&a, &b) - p.conjunction(&b, &a)).abs() < PROB_EPSILON);
    }

    #[test]
    fn addmult_conjunction_is_product() {
        let p = AddMultProbability;
        let result = p.conjunction(&0.8, &0.7);
        assert!((result - 0.56).abs() < PROB_EPSILON);
    }

    #[test]
    fn addmult_disjunction_noisy_or() {
        let p = AddMultProbability;
        let result = p.disjunction(&0.7, &0.6);
        // noisy-OR: 0.7 + 0.6 - 0.7*0.6 = 1.3 - 0.42 = 0.88
        assert!((result - 0.88).abs() < PROB_EPSILON);
    }

    // --- Semiring law tests for BooleanProvenance ---

    #[test]
    fn boolean_identities() {
        let p = BooleanProvenance;
        assert_eq!(p.disjunction(&true, &p.zero()), true);
        assert_eq!(p.disjunction(&false, &p.zero()), false);
        assert_eq!(p.conjunction(&true, &p.one()), true);
        assert_eq!(p.conjunction(&false, &p.one()), false);
    }

    #[test]
    fn boolean_annihilator() {
        let p = BooleanProvenance;
        assert_eq!(p.conjunction(&true, &p.zero()), false);
    }

    // --- Conversion tests ---

    #[test]
    fn minmax_roundtrip() {
        let p = MinMaxProbability;
        let tag = p.tag_from_probability(0.75);
        assert!((p.recover_probability(&tag) - 0.75).abs() < PROB_EPSILON);
    }

    #[test]
    fn boolean_from_probability() {
        let p = BooleanProvenance;
        assert_eq!(p.tag_from_probability(0.5), true);
        assert_eq!(p.tag_from_probability(0.0), false);
    }

    // --- Saturation tests ---

    #[test]
    fn minmax_saturation_convergence() {
        let p = MinMaxProbability;
        assert!(p.is_saturated(&0.7, &0.7));
        assert!(!p.is_saturated(&0.7, &0.8));
    }

    #[test]
    fn boolean_saturation_convergence() {
        let p = BooleanProvenance;
        assert!(p.is_saturated(&true, &true));
        assert!(!p.is_saturated(&true, &false));
    }

    // --- TopKProofs tests ---

    fn make_proof(ids: &[u32]) -> Proof {
        ids.iter().copied().collect()
    }

    #[test]
    fn topk_zero_and_one() {
        let p = TopKProofs::new(5);
        assert_eq!(p.zero(), vec![]);
        assert_eq!(p.one(), vec![BTreeSet::new()]);
    }

    #[test]
    fn topk_conjunction_identity() {
        let p = TopKProofs::new(5);
        // a ⊗ one = a
        let a = vec![make_proof(&[0])];
        let result = p.conjunction(&a, &p.one());
        assert_eq!(result, a);
        // zero ⊗ a = zero
        let zero_result = p.conjunction(&p.zero(), &a);
        assert_eq!(zero_result, vec![]);
    }

    #[test]
    fn topk_conjunction_shared_variable() {
        let p = TopKProofs::new(5);
        // {0} ⊗ {0, 1} = {{0, 1}} (union, no duplication)
        let a = vec![make_proof(&[0])];
        let b = vec![make_proof(&[0, 1])];
        let result = p.conjunction(&a, &b);
        assert_eq!(result, vec![make_proof(&[0, 1])]);
    }

    #[test]
    fn topk_disjunction_dedup_and_truncate() {
        let p = TopKProofs::new(2);
        // [{0}] ⊕ [{0}, {1}] = [{0}, {1}] (dedup, truncate to 2)
        let a = vec![make_proof(&[0])];
        let b = vec![make_proof(&[0]), make_proof(&[1])];
        let result = p.disjunction(&a, &b);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn topk_wmc_single_proof() {
        let p = TopKProofs::new(5);
        p.tag_from_probability_with_id(0.8, 0);
        let tag = vec![make_proof(&[0])];
        let prob = p.recover_probability(&tag);
        assert!((prob - 0.8).abs() < PROB_EPSILON, "expected 0.8, got {}", prob);
    }

    #[test]
    fn topk_wmc_two_independent() {
        let p = TopKProofs::new(5);
        p.tag_from_probability_with_id(0.8, 0);
        p.tag_from_probability_with_id(0.6, 1);
        // {0} and {1} are independent — noisy-OR: 0.8 + 0.6 - 0.48 = 0.92
        let tag = vec![make_proof(&[0]), make_proof(&[1])];
        let prob = p.recover_probability(&tag);
        assert!((prob - 0.92).abs() < PROB_EPSILON, "expected 0.92, got {}", prob);
    }

    #[test]
    fn topk_wmc_overlap_canonical() {
        // Canonical overlap example: proof1={0,1}, proof2={0,2}
        // P(x)=0.8, P(y)=0.6, P(z)=0.5
        // Exact: P(x∧y) + P(x∧z) - P(x∧y∧z) = 0.48 + 0.40 - 0.24 = 0.64
        // noisy-OR of 0.48 and 0.40 = 0.688 (wrong — overcounts x)
        let p = TopKProofs::new(5);
        p.tag_from_probability_with_id(0.8, 0); // x
        p.tag_from_probability_with_id(0.6, 1); // y
        p.tag_from_probability_with_id(0.5, 2); // z
        let tag = vec![make_proof(&[0, 1]), make_proof(&[0, 2])];
        let wmc = p.recover_probability(&tag);
        assert!((wmc - 0.64).abs() < PROB_EPSILON, "expected 0.64, got {}", wmc);
    }

    #[test]
    fn topk_saturation() {
        let p = TopKProofs::new(5);
        let a = vec![make_proof(&[0])];
        let b = vec![make_proof(&[1])];
        assert!(p.is_saturated(&a, &a));
        assert!(!p.is_saturated(&a, &b));
    }

}
