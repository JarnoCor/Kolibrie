/*
 * Copyright © 2024 Volodymyr Kadzhaia
 * Copyright © 2024 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */
use crate::terms::{Term, TriplePattern};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Debug, Clone, Copy, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Triple {
    pub subject: u32,
    pub predicate: u32,
    pub object: u32,
}

impl Triple {

    /// Converts a triple (subject, predicate, object) (all u32 values) into a TriplePattern
    /// Simply wraps each element into a constant
    pub fn to_pattern(&self) -> TriplePattern {
        (
            Term::Constant(self.subject),
            Term::Constant(self.predicate),
            Term::Constant(self.object),
        )
    }
}

#[derive(PartialEq, Debug, Clone, Copy, Eq, PartialOrd, Hash)]
pub struct TimestampedTriple {
    pub triple: Triple,
    pub timestamp: u64,
}

impl Ord for TimestampedTriple {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp
            .cmp(&other.timestamp)
            .then_with(|| self.triple.cmp(&other.triple))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_test_1() {
        let triple_1 = TimestampedTriple {
            triple: Triple {
                subject: 1,
                object: 2,
                predicate: 3,
            },
            timestamp: 1,
        };

        let triple_2 = TimestampedTriple {
            triple: Triple {
                subject: 1,
                object: 2,
                predicate: 3,
            },
            timestamp: 2,
        };

        assert!(triple_1 < triple_2);
        assert!(!(triple_1 == triple_2));
        assert!(!(triple_1 > triple_2));
    }
}
