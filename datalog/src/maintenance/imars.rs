use std::collections::{HashMap, HashSet};

use shared::triple::{TimestampedTriple, Triple};

use crate::reasoning::Reasoner;

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
    last_change: HashMap<usize, Triple>,
}

impl Report {
    pub fn new() -> Report {
        Report {
            strategies: Vec::new(),
            last_change: HashMap::new(),
        }
    }
    pub fn add(&mut self, strategy: ReportStrategy) {
        self.strategies.push(strategy);
    }
    pub fn report(&mut self, close_time: usize, content: &HashMap<usize, Triple>, ts: usize) -> bool {
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

pub struct IMARSWindow {
    reasoner: Reasoner,
    width: usize,
    slide: usize,
    timestamped_contents: HashMap<usize, Triple>,
    report: Report,
    tick: Tick,
    app_time: usize,
}

/*
needed:
- reasoner
- slide
- width
- app_time
- report
- open / close --> Window
- content container
- tick
- callback
*/

impl IMARSWindow {
    pub fn new(width: usize, slide: usize, report: Report, tick: Tick) -> Self {
        IMARSWindow {
            reasoner: Reasoner::new(),
            width,
            slide,
            timestamped_contents: HashMap::new(),
            report,
            tick,
            app_time: 0
        }
    }

    pub fn add_to_window(&mut self, added_triples: Vec<TimestampedTriple>, ts: usize) {

        self.maintenance(&added_triples);

        let max_timestamp_triple = added_triples.iter().max_by(|x, y| x.timestamp.cmp(&y.timestamp));

        if let Some(TimestampedTriple { triple: _, timestamp: max_timestamp }) = max_timestamp_triple {

            if self.report.report(self.app_time+self.width, &self.timestamped_contents, ts) {

                match self.tick {
                    Tick::TimeDriven => {
                        if ts > self.app_time {
                            self.app_time = ts;
                        }
                    }

                    Tick::TupleDriven => {
                        if *max_timestamp > self.app_time as u64 {
                            self.app_time = *max_timestamp as usize;
                        }
                    }

                    _ => (),
                };

            }
        }
    }

    pub fn maintenance(&mut self, added_triples: &Vec<TimestampedTriple>) {
        let mut net_additions: HashSet<Triple> = HashSet::new();
        let mut net_deletions: HashSet<Triple> = HashSet::new();
        let mut insertions: HashSet<Triple> = HashSet::new();
        let mut new: HashSet<Triple> = HashSet::new();
        let mut renewed: HashSet<Triple> = HashSet::new();

        self.maintenance_rules_m1(added_triples, &mut net_additions, &mut net_deletions, &mut insertions, &mut new, &mut renewed);
    }

    fn maintenance_rules_m1(&mut self,
        added_triples: &Vec<TimestampedTriple>,
        net_additions: &mut HashSet<Triple>,
        net_deletions: &mut HashSet<Triple>,
        insertions: &mut HashSet<Triple>,
        new: &mut HashSet<Triple>,
        renewed: &mut HashSet<Triple>
    ) {
        todo!();
    }
}
