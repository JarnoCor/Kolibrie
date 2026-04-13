use std::cmp::min;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::collections::hash_set::{Iter, IntoIter};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::mpsc::{Receiver, channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::mem;

use shared::{dictionary::Dictionary, terms::TriplePattern, triple::{TimestampedTriple, Triple}};

use crate::{maintenance::construct_triple, reasoning::{Reasoner, rules::{evaluate_filters, join_rule}}};

#[derive(Clone, Debug)]
pub enum ReportStrategy {
    NonEmptyContent,
    OnContentChange,
    OnWindowClose,
    Periodic(usize),
}

impl Default for ReportStrategy {
    fn default() -> Self {
        ReportStrategy::OnWindowClose
    }
}

#[derive(Clone, Debug)]
pub enum Tick {
    TimeDriven,
    TupleDriven,
    BatchDriven,
}

impl Default for Tick {
    fn default() -> Self {
        Tick::TimeDriven
    }
}

pub struct Report {
    strategies: Vec<ReportStrategy>,
    last_change: BTreeSet<TimestampedTriple>,
}

impl Report {
    pub fn new() -> Report {
        Report {
            strategies: Vec::new(),
            last_change: BTreeSet::new(),
        }
    }
    pub fn add(&mut self, strategy: ReportStrategy) {
        self.strategies.push(strategy);
    }
    pub fn report(&mut self, close_time: usize, content: &BTreeSet<TimestampedTriple>, ts: usize) -> bool {
        self.strategies.iter().all(|strategy| match strategy {
            ReportStrategy::NonEmptyContent => content.len() > 0,
            ReportStrategy::OnContentChange => {
                let comp = content.eq(&self.last_change);
                self.last_change = content.clone();
                comp
            }
            ReportStrategy::OnWindowClose => close_time <= ts,
            ReportStrategy::Periodic(period) => ts % period == 0,
        })
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct ContentContainer<I>
where
    I: Eq + PartialEq + Clone + Debug + Hash + Send,
{
    elements: HashSet<I>,
    last_timestamp_changed: usize,
    origin: String
}

impl<I> ContentContainer<I>
where
    I: Eq + PartialEq + Clone + Debug + Hash + Send,
{
    fn new() -> ContentContainer<I> {
        ContentContainer {
            elements: HashSet::new(),
            last_timestamp_changed: 0,
            origin: String::default()
        }
    }
    fn new_with_origin(origin : &str) -> ContentContainer<I> {
        ContentContainer {
            elements: HashSet::new(),
            last_timestamp_changed: 0,
            origin: origin.to_string()
        }
    }
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    fn add(&mut self, triple: I, ts: usize) {
        self.elements.insert(triple);
        self.last_timestamp_changed = ts;
    }
    pub fn get_last_timestamp_changed(&self) -> usize {
        self.last_timestamp_changed
    }

    pub fn iter(&self) -> Iter<'_, I> {
        self.elements.iter()
    }
    pub fn into_iter(mut self) -> IntoIter<I> {
        let map = mem::take(&mut self.elements);
        map.into_iter()
    }
}

pub struct IMARSWindow {
    reasoner: Reasoner,
    width: usize,
    slide: usize,
    timestamped_contents: BTreeSet<TimestampedTriple>,
    report: Report,
    tick: Tick,
    app_time: usize,
    consumer: Option<Sender<ContentContainer<TimestampedTriple>>>,
    // Make callbacks Send so they can be safely transferred to worker threads
    call_back: Option<Box<dyn FnMut(ContentContainer<TimestampedTriple>) -> () + Send + 'static>>,
}

impl IMARSWindow {
    pub fn new(width: usize, slide: usize, report: Report, tick: Tick) -> Self {
        IMARSWindow {
            reasoner: Reasoner::new(),
            width,
            slide,
            timestamped_contents: BTreeSet::new(),
            report,
            tick,
            app_time: 0,
            consumer: None,
            call_back: None
        }
    }

    pub fn add_to_window(&mut self, added_triples: Vec<TimestampedTriple>, ts: usize) {
        let now = self.app_time + self.width;

        // perform the IMaRS maintenance algorithm to add and remove triples
        self.maintenance(added_triples.clone(), now);

        let max_timestamp_triple = added_triples.iter().max_by(|x, y| x.timestamp.cmp(&y.timestamp));

        // drive the window forward, based on the tick type
        if let Some(TimestampedTriple { triple: _, timestamp: max_timestamp }) = max_timestamp_triple {
            if self.report.report(self.app_time+self.width, &self.timestamped_contents, ts) {

                match self.tick {
                    Tick::TimeDriven => {
                        if ts >= self.app_time + self.slide {
                            self.app_time = ts;

                            let mut contents = ContentContainer::new();
                            for triple in self.timestamped_contents.clone() {
                                contents.add(triple, ts);
                            }

                            if let Some(sender) = &self.consumer {
                                if let Err(e) = sender.send(contents.clone()) {
                                    eprintln!("Failed to send window content to consumer: {:?}", e);
                                }
                            }
                            // single threaded consumer using callback
                            if let Some(call_back) = &mut self.call_back {
                                (call_back)(contents);
                            }
                        }
                    }

                    Tick::TupleDriven => {
                        if *max_timestamp >= (self.app_time + self.slide).try_into().unwrap() {
                            self.app_time = *max_timestamp as usize;

                            let mut contents = ContentContainer::new();
                            for triple in self.timestamped_contents.clone() {
                                contents.add(triple, ts);
                            }

                            if let Some(sender) = &self.consumer {
                                if let Err(e) = sender.send(contents.clone()) {
                                    eprintln!("Failed to send window content to consumer: {:?}", e);
                                }
                            }
                            // single threaded consumer using callback
                            if let Some(call_back) = &mut self.call_back {
                                (call_back)(contents);
                            }
                        }
                    }

                    _ => (),
                };

            }
        }
    }

    pub fn maintenance(&mut self, added_triples: Vec<TimestampedTriple>, now: usize) {
        // remove all triples that fall out of the window
        while let Some(triple) = self.timestamped_contents.first() {
            if triple.timestamp < now.try_into().unwrap() {
                self.timestamped_contents.pop_first();
            } else {
                break;
            }
        }

        // this tracks which triples were inserted and their timestamp
        // the reason this does not hold timestamped triples is to easily convert the keys (plain triples) to a set later
        let mut inserted: HashSet<Triple> = HashSet::new();

        // add the new triples
        for triple in added_triples {

            // case 1: the triple renews another one already present
            if let Some(present_triple) = self.timestamped_contents.get(&triple) {

                if triple.timestamp > present_triple.timestamp {
                    let to_remove = present_triple.clone();

                    self.timestamped_contents.remove(&to_remove);
                    self.timestamped_contents.insert(triple.clone());

                    inserted.insert(triple.triple);
                }
            }
            // case 2: the triple was not already present
            else {
                self.timestamped_contents.insert(triple.clone());
                inserted.insert(triple.triple);
            }
        }

        let mut materialisation: HashSet<Triple> = self.timestamped_contents.iter()
                .cloned()
                .map(|triple| triple.triple)
                .collect();

        let mut bindings: Vec<HashMap<String, u32>>;
        let mut inferred: Triple;

        // infer new triples, but only consider rules where at least one body atom can be matched with a new triple
        for rule in self.reasoner.rules.clone() {

            bindings = join_rule(&rule, &materialisation, &inserted);

            for binding in bindings {
                if evaluate_filters(&binding, &rule.filters, &self.reasoner.dictionary.write().unwrap()) {

                    for conclusion in &rule.conclusion {
                        inferred = construct_triple(conclusion,  &binding, &mut self.reasoner.dictionary.write().unwrap());

                        let min_timestamp = self.find_min_timestamp(&rule.premise, &binding, &mut self.reasoner.dictionary.write().unwrap());

                        self.timestamped_contents.insert(TimestampedTriple { triple: inferred.clone(), timestamp: min_timestamp });

                        materialisation.insert(inferred.clone());
                        inserted.insert(inferred);

                    }
                }
            }
        }
    }

    /// Find which triples were used in the binding process, and find the smallest timestamp associated with them.
    fn find_min_timestamp(&self, premises: &Vec<TriplePattern>, binding: &HashMap<String, u32>, dict: &mut Dictionary) -> u64 {
        let mut min_timestamp = u64::MAX;

        let mut constructed: Triple;
        for premise in premises {
            constructed = construct_triple(premise, binding, dict);

            // get the triples expiration timestamp
            if let Some(timestamped_triple) = self.timestamped_contents.iter()
                    .find(|timestamped_triple| timestamped_triple.triple == constructed)
            {
                min_timestamp = min(min_timestamp, timestamped_triple.timestamp);
            }
        }

        min_timestamp
    }
}









#[cfg(test)]
mod tests {
    use super::*;


    // #[test]
}
