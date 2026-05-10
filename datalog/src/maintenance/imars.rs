use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::collections::hash_set::{Iter, IntoIter};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::mpsc::{Receiver, channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::mem;

// use shared::{dictionary, triple};
use shared::{/*dictionary::Dictionary, terms::TriplePattern, */triple::{TimestampedTriple, Triple}};

use crate::reasoning::rules::join_rule_with_timestamps;
use crate::{maintenance::construct_triple, reasoning::{Reasoner, rules::{evaluate_filters/*, join_rule*/}}};

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

#[derive(Clone)]
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
    pub fn report(&mut self, close_time: usize, content: &BTreeMap<u64, HashSet<Triple>>, ts: usize) -> bool {
        let mut contents = BTreeSet::new();
        for (timestamp, set) in content.clone() {
            for triple in set {
                contents.insert(TimestampedTriple { triple, timestamp });
            }
        }
        self.strategies.iter().all(|strategy| match strategy {
            ReportStrategy::NonEmptyContent => content.len() > 0,
            ReportStrategy::OnContentChange => {
                let comp = contents.eq(&self.last_change);
                self.last_change = contents.clone();
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
    pub fn new() -> ContentContainer<I> {
        ContentContainer {
            elements: HashSet::new(),
            last_timestamp_changed: 0,
            origin: String::default()
        }
    }
    #[allow(dead_code)]
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
    pub reasoner: Reasoner,
    pub width: usize,
    pub slide: usize,
    pub timeline: BTreeMap<u64, HashSet<Triple>>, // used to track the triples per timestamp
    pub index: HashMap<Triple, u64>, // tracks which timestamps are associated with each triple

    report: Report,
    tick: Tick,
    pub app_time: usize,
    pub last_emit_time: usize,
    consumer: Option<Sender<ContentContainer<TimestampedTriple>>>,
    // Make callbacks Send so they can be safely transferred to worker threads
    call_back: Option<Box<dyn FnMut(ContentContainer<TimestampedTriple>) -> () + Send + 'static>>,
}

impl Clone for IMARSWindow {
    fn clone(&self) -> Self {
        Self {
            reasoner: self.reasoner.clone(),
            width: self.width.clone(),
            slide: self.slide.clone(),
            timeline: self.timeline.clone(),
            index: self.index.clone(),
            report: self.report.clone(),
            tick: self.tick.clone(),
            app_time: self.app_time.clone(),
            last_emit_time: self.last_emit_time.clone(),
            consumer: self.consumer.clone(),
            call_back: None
        }
    }
}

impl IMARSWindow {
    pub fn new(width: usize, slide: usize, report: Report, tick: Tick) -> Self {
        IMARSWindow {
            reasoner: Reasoner::new(),
            width,
            slide,
            timeline: BTreeMap::new(),
            index: HashMap::new(),
            report,
            tick,
            app_time: 0,
            last_emit_time: 0,
            consumer: None,
            call_back: None
        }
    }

    pub fn add_to_window(&mut self, added_triples: Vec<TimestampedTriple>, ts: usize) {

        // drive the window forward, based on the tick type
        let new_time;
        let max_timestamp_triple = added_triples.iter().cloned().max_by(|x, y| x.timestamp.cmp(&y.timestamp));

        new_time = match self.tick {
                Tick::TimeDriven => ts,
                Tick::TupleDriven => {
                    if let Some(TimestampedTriple { triple: _, timestamp: max_timestamp }) = max_timestamp_triple {
                        max_timestamp as usize
                    } else {
                        0
                    }
                },
                _ => self.app_time
            };

        if new_time > self.app_time {
            self.app_time = new_time;
        }

        let now = self.app_time;

        // perform the IMaRS maintenance algorithm to add and remove triples
        self.maintenance(added_triples, now);

        // emit the contents of the window, based on the tick type
        if let Some(TimestampedTriple { triple: _, timestamp: max_timestamp }) = max_timestamp_triple {
            if self.report.report(self.app_time+self.width, &self.timeline, ts) {

                match self.tick {
                    Tick::TimeDriven => {
                        self.app_time = ts;
                        if ts >= self.last_emit_time + self.slide {
                            self.last_emit_time = ts;

                            let mut contents = ContentContainer::new();
                            for (timestamp, set) in self.timeline.clone() {
                                for triple in set {
                                    contents.add(TimestampedTriple { triple, timestamp }, ts);
                                }
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
                        self.app_time = max_timestamp as usize;
                        if max_timestamp >= (self.last_emit_time + self.slide).try_into().unwrap() {
                            self.last_emit_time = max_timestamp as usize;

                            let mut contents = ContentContainer::new();
                            for (timestamp, set) in self.timeline.clone() {
                                for triple in set {
                                    contents.add(TimestampedTriple { triple, timestamp }, ts);
                                }
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
        let expired: Vec<u64> = self.timeline.range(0..now as u64)
            .map(|(timestamp, _)| *timestamp)
            .collect();

        let mut deleted: HashSet<Triple> = HashSet::new();
        for timestamp in expired {
            if let Some(triples) = self.timeline.remove(&timestamp) {
                for triple in triples {
                    if let Some(&current_expiration) = self.index.get(&triple) {
                        // only remove the triple if the timestamp is the current one
                        if current_expiration <= timestamp {
                            self.index.remove(&triple);
                            self.reasoner.index_manager.delete(&triple);
                            deleted.insert(triple);
                        }
                    }
                }
            }
        }

        let added = self.insert(&deleted, &added_triples);

        for (triple, _) in added.iter() {
            self.reasoner.index_manager.insert(triple);
        }
    }

    fn insert(&mut self, deleted: &HashSet<Triple>, to_add: &Vec<TimestampedTriple>) -> HashMap<Triple, u64> {
        let valid_facts: HashMap<Triple, u64> = self.index.iter()
                .filter(|(triple, _)| !deleted.contains(*triple))
                .map(|(t, e)| (*t, *e))
                .collect();

        // the capacity is chosen arbitratily
        let mut timeline_updates: HashMap<u64, Vec<Triple>> = HashMap::with_capacity(to_add.len() * 3);

        let mut n_a: HashMap<Triple, u64> = to_add.iter()
                .filter_map(|triple: &TimestampedTriple| {
                    let expiration = triple.timestamp + self.width as u64;
                    let current_expiration = self.index.get(&triple.triple).copied().unwrap_or(0);
                    if expiration > current_expiration {
                        self.index.insert(triple.triple, expiration);
                        timeline_updates.entry(expiration).or_insert_with(|| Vec::new()).push(triple.triple);

                        Some((triple.triple, expiration))
                    } else {
                        None
                    }

                })
                .collect();

        let mut delta: HashMap<Triple, u64>;
        let mut added: HashMap<Triple, u64> = HashMap::new();

        let dictionary = self.reasoner.dictionary.clone();
        let mut dictionary = dictionary.write().unwrap();

        loop {
            // calculate the unprocessed changes
            delta = n_a.iter()
                    .filter(|(triple, timestamp)| {
                        let current_expiration = valid_facts.get(*triple).copied().unwrap_or(0);
                        let added_expiration = added.get(*triple).copied().unwrap_or(0);

                        **timestamp > current_expiration && **timestamp > added_expiration
                    })
                    .map(|(triple, timestamp)| (triple.clone(), *timestamp))
                    .collect();

            // if there are no more changes, stop
            if delta.is_empty() {
                break;
            }

            // add the delta to added
            added.extend(delta.clone());

            // clear the contents of the previous iteration
            n_a.clear();

            let facts: HashMap<Triple, u64> = valid_facts.iter()
                    .map(|(t, e)| (*t, *e))
                    .chain(added.iter().map(|(t, e)| (*t, *e)))
                    .collect();

            let mut results: Vec<(HashMap<String, u32>, u64)>;
            let mut inferred: Triple;
            for rule in self.reasoner.rules.clone().iter() {
                results = join_rule_with_timestamps(rule, &facts, &delta);
                for (binding, min_timestamp) in results.iter() {
                    if evaluate_filters(binding, &rule.filters, &dictionary) {
                        for conclusion in &rule.conclusion {
                            inferred = construct_triple(conclusion,  binding, &mut dictionary);

                            let entry = n_a.entry(inferred).or_insert(0);
                            if *min_timestamp > *entry {
                                *entry = *min_timestamp;
                                timeline_updates.entry(*min_timestamp).or_insert_with(|| Vec::with_capacity(rule.conclusion.len())).push(inferred);
                            }

                        }
                    }
                }
            }
        }

        drop(dictionary);

        // commit timeline updates
        for (timestamp, triples) in timeline_updates.drain() {
            self.timeline.entry(timestamp).or_insert_with(HashSet::new).extend(triples);
        }

        // return the added triples
        added
    }

    pub fn register(&mut self) -> Receiver<ContentContainer<TimestampedTriple>> {
        let (send, recv) = channel::<ContentContainer<TimestampedTriple>>();
        self.consumer.replace(send);
        recv
    }
    pub fn register_callback(
        &mut self,
        function: Box<dyn FnMut(ContentContainer<TimestampedTriple>) -> () + Send + 'static>,
    ) {
        self.call_back.replace(function);
    }
    pub fn flush(&mut self) {
        let mut content = ContentContainer::new();
        for (timestamp, set) in self.timeline.clone() {
            for triple in set {
                content.add(TimestampedTriple { triple, timestamp }, self.app_time);
            }
        }

        if let Some(call_back) = &mut self.call_back {
            (call_back)(content.clone());
        }
        if let Some(sender) = &self.consumer {
            let _ = sender.send(content.clone());
        }
    }
    pub fn stop(&mut self) {
        self.consumer.take();
    }



    pub fn decode_triple(&self, triple: &Triple) -> String {
        let dictionary = self.reasoner.dictionary.write().unwrap();

        let subject = dictionary.decode(triple.subject).unwrap_or("unknown");
        let predicate = dictionary.decode(triple.predicate).unwrap_or("unknown");
        let object = dictionary.decode(triple.object).unwrap_or("unknown");

        format!("{} {} {}", subject, predicate, object)
    }
}

#[allow(dead_code)]
struct ConsumerInner<I>
where
    I: Eq + PartialEq + Clone + Debug + Hash + Send,
{
    data: Mutex<Vec<ContentContainer<I>>>,
}

#[allow(dead_code)]
pub struct Consumer<I>
where
    I: Eq + PartialEq + Clone + Debug + Hash + Send,
{
    inner: Arc<ConsumerInner<I>>,
}

#[allow(dead_code)]
impl<I> Consumer<I>
where
    I: Eq + PartialEq + Clone + Debug + Hash + Send + 'static,
{
    pub fn new() -> Consumer<I> {
        Consumer {
            inner: Arc::new(ConsumerInner {
                data: Mutex::new(Vec::new()),
            }),
        }
    }
    pub fn start(&self, receiver: Receiver<ContentContainer<I>>) {
        let consumer_temp = self.inner.clone();
        thread::spawn(move || loop {
            match receiver.recv() {
                Ok(content) => {
                    // dbg!("Found graph {:?}", &content);
                    consumer_temp.data.lock().unwrap().push(content);
                }
                Err(_) => {
                    // dbg!("Shutting down!");
                    break;
                }
            }
        });
    }
    fn len(&self) -> usize {
        self.inner.data.lock().unwrap().len()
    }
}









#[cfg(test)]
mod tests {
    use shared::{rule::Rule, terms::Term};

    use super::*;
    use std::time::{Duration, Instant};


    fn vec_equal<T: Eq+Hash>(vec1: &Vec<T>, vec2: &Vec<T>) -> bool {
        let mut counts: HashMap<&T, u32> = HashMap::new();

        for element in vec1 {
            let value = counts.entry(element).or_insert(0);
            *value += 1;
        }

        for element in vec2 {
            match counts.get_mut(element) {
                Some(count) => {
                    if *count > 0 {
                        *count -= 1;
                    }
                },
                None => return false,
            }
        }

        true
    }

    #[test]
    fn new_window_test() {
        let mut report = Report::new();
        report.add(ReportStrategy::Periodic(1));
        let mut window = IMARSWindow::new(5, 2, report, Tick::TimeDriven);

        let receiver = window.register();
        let consumer = Consumer::new();
        consumer.start(receiver);

        for i in 1..=10 {
            let triple = TimestampedTriple {
                triple: Triple {
                    subject: i,
                    predicate: 2,
                    object: 3,
                },
                timestamp: i as u64
            };
            window.add_to_window(vec![triple], i as usize);
        }

        window.stop();
        thread::sleep(Duration::from_secs(1));
        assert_eq!(5, consumer.len());

        let triples = window.reasoner.index_manager.query(None, None, None);
        // println!("{:#?}", triples);
        // println!("{}", triples.len());
        assert_eq!(6, triples.len());
    }

    #[test]
    fn maintenance_graph_test() {
        let mut report = Report::new();
        report.add(ReportStrategy::Periodic(1));
        let mut window = IMARSWindow::new(1, 0, report, Tick::TimeDriven);

        let receiver = window.register();
        let consumer = Consumer::new();
        consumer.start(receiver);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let mut initial_facts = vec![];

        let s = dictionary.encode("a");
        let p = dictionary.encode("edge");
        let o = dictionary.encode("b");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 3 };
        initial_facts.push(timestamped_triple);

        let s = dictionary.encode("b");
        let p = dictionary.encode("edge");
        let o = dictionary.encode("c");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 1 };
        initial_facts.push(timestamped_triple);

        let s = dictionary.encode("e");
        let p = dictionary.encode("edge");
        let o = dictionary.encode("d");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 3 };
        initial_facts.push(timestamped_triple);

        let edge = Term::Constant(dictionary.encode("edge"));
        let reachable = Term::Constant(dictionary.encode("reachable"));

        drop(dictionary);

        window.reasoner.add_rule(Rule {
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
            negative_premise: vec![],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                reachable,
                Term::Variable("z".to_string()),
            )],
            filters: vec![],
        });

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let encoded_edge = Term::Constant(dictionary.encode("edge"));
        let encoded_reachable = Term::Constant(dictionary.encode("reachable"));

        drop(dictionary);

        window.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_edge,
                Term::Variable("y".to_string()),
            ),
            ],
            negative_premise: vec![],
            conclusion: vec![(
                Term::Variable("x".to_string()),
                encoded_reachable,
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();


        let added: Vec<TimestampedTriple> = vec![
            TimestampedTriple {
                triple: Triple {
                    subject: dictionary.encode("b"),
                    predicate: dictionary.encode("edge"),
                    object: dictionary.encode("e"),
                },
                timestamp: 3
            }
        ];

        drop(dictionary);

        window.add_to_window(initial_facts, 1);

        let start = Instant::now();
        window.add_to_window(added, 3);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = window.reasoner.index_manager.query(None, None, None);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let test_materialisation = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("e"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("edge"),
                object: dictionary.encode("e"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("e"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("e"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("reachable"),
                object: dictionary.encode("e"),
            },
        ];

        drop(dictionary);

        assert!(vec_equal(&test_materialisation, &materialisation));
    }

    #[test]
    fn maintenance_small_test() {
        let mut report = Report::new();
        report.add(ReportStrategy::Periodic(1));
        let mut window = IMARSWindow::new(1, 0, report, Tick::TimeDriven);

        let receiver = window.register();
        let consumer = Consumer::new();
        consumer.start(receiver);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let mut initial_facts = vec![];

        let s = dictionary.encode("a");
        let p = dictionary.encode("t");
        let o = dictionary.encode("b");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 3 };
        initial_facts.push(timestamped_triple);

        let s = dictionary.encode("f");
        let p = dictionary.encode("t");
        let o = dictionary.encode("b");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 1 };
        initial_facts.push(timestamped_triple);

        let s = dictionary.encode("c");
        let p = dictionary.encode("t");
        let o = dictionary.encode("d");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 3 };
        initial_facts.push(timestamped_triple);

        let encoded_r = Term::Constant(dictionary.encode("r"));
        let encoded_t = Term::Constant(dictionary.encode("t"));

        drop(dictionary);

        window.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_t,
                Term::Variable("y".to_string()),
            ),
            ],
            negative_premise: vec![],
            conclusion: vec![(
                Term::Variable("y".to_string()),
                encoded_r,
                Term::Variable("y".to_string()),
            )],
            filters: vec![],
        });

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();


        let added: Vec<TimestampedTriple> = vec![
            TimestampedTriple {
                triple: Triple {
                    subject: dictionary.encode("g"),
                    predicate: dictionary.encode("t"),
                    object: dictionary.encode("d"),
                },
                timestamp: 3
            }
        ];

        drop(dictionary);

        window.add_to_window(initial_facts, 1);

        let start = Instant::now();
        window.add_to_window(added, 3);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = window.reasoner.index_manager.query(None, None, None);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let test_materialisation = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("c"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("g"),
                predicate: dictionary.encode("t"),
                object: dictionary.encode("d"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("d"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("d"),
            },
        ];

        drop(dictionary);

        // println!("Facts:");
        // let mut facts = vec![];
        // for fact in &window.reasoner.index_manager.query(None, None, None) {
        //     facts.push(window.decode_triple(fact));
        // }
        // println!("{:#?}", facts);

        // println!("{:#?}", materialisation);
        // println!("Test materialisation:");
        // println!("{:#?}", test_materialisation);

        assert!(vec_equal(&test_materialisation, &materialisation));
    }

    #[test]
    fn maintenance_reflexive_test() {
        let mut report = Report::new();
        report.add(ReportStrategy::Periodic(1));
        let mut window = IMARSWindow::new(1, 0, report, Tick::TimeDriven);

        let receiver = window.register();
        let consumer = Consumer::new();
        consumer.start(receiver);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let mut initial_facts = vec![];

        let s = dictionary.encode("a");
        let p = dictionary.encode("r");
        let o = dictionary.encode("b");
        let triple = Triple { subject: s, predicate: p, object: o };
        let timestamped_triple = TimestampedTriple { triple: triple, timestamp: 1 };
        initial_facts.push(timestamped_triple);

        let encoded_r = Term::Constant(dictionary.encode("r"));

        drop(dictionary);

        window.reasoner.add_rule(Rule {
            premise: vec![
            (
                Term::Variable("x".to_string()),
                encoded_r.clone(),
                Term::Variable("y".to_string()),
            ),
            ],
            negative_premise: vec![],
            conclusion: vec![(
                Term::Variable("y".to_string()),
                encoded_r,
                Term::Variable("x".to_string()),
            )],
            filters: vec![],
        });

        let added: Vec<TimestampedTriple> = vec![];

        window.add_to_window(initial_facts, 1);

        let dictionary = &window.reasoner.dictionary;
        let mut dictionary = dictionary.write().unwrap();

        let materialisation = window.reasoner.index_manager.query(None, None, None);
        let test_materialisation = vec![
            Triple {
                subject: dictionary.encode("a"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("b"),
            },
            Triple {
                subject: dictionary.encode("b"),
                predicate: dictionary.encode("r"),
                object: dictionary.encode("a"),
            },
        ];

        drop(dictionary);

        assert!(vec_equal(&test_materialisation, &materialisation));

        let start = Instant::now();
        window.add_to_window(added, 3);
        let duration = start.elapsed();
        println!("Inference took: {:?}", duration);

        let materialisation = window.reasoner.index_manager.query(None, None, None);
        let test_materialisation = vec![];

        // println!("Facts:");
        // let mut facts = vec![];
        // for fact in &window.reasoner.index_manager.query(None, None, None) {
        //     facts.push(window.decode_triple(fact));
        // }
        // println!("{:#?}", facts);

        println!("{:#?}", materialisation);
        println!("Test materialisation:");
        println!("{:#?}", test_materialisation);

        assert!(vec_equal(&test_materialisation, &materialisation));
    }
}

// expiration time = 6