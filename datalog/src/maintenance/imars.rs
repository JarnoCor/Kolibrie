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
    reasoner: Reasoner,
    width: usize,
    slide: usize,
    timestamped_contents: BTreeSet<TimestampedTriple>,
    report: Report,
    tick: Tick,
    app_time: usize,
    last_emit_time: usize,
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

        let now = self.app_time; // self.app_time.saturating_sub(self.width);

        // perform the IMaRS maintenance algorithm to add and remove triples
        self.maintenance(added_triples, now);

        // emit the contents of the window, based on the tick type
        if let Some(TimestampedTriple { triple: _, timestamp: max_timestamp }) = max_timestamp_triple {
            if self.report.report(self.app_time+self.width, &self.timestamped_contents, ts) {

                match self.tick {
                    Tick::TimeDriven => {
                        self.app_time = ts;
                        if ts >= self.last_emit_time + self.slide {
                            self.last_emit_time = ts;

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
                        self.app_time = max_timestamp as usize;
                        if max_timestamp >= (self.last_emit_time + self.slide).try_into().unwrap() {
                            self.last_emit_time = max_timestamp as usize;

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
                let deleted = self.timestamped_contents.pop_first().unwrap();
                self.reasoner.index_manager.delete(&deleted.triple);
            } else {
                break;
            }
        }

        // this tracks which triples were inserted and their timestamp
        // the reason this does not hold timestamped triples is to easily convert the keys (plain triples) to a set later
        let mut inserted: HashSet<Triple> = HashSet::new();

        // add the new triples
        for triple in added_triples {
            let expiration: u64 = triple.timestamp + self.width as u64;

            let expiration_triple = TimestampedTriple { triple: triple.triple.clone(), timestamp: expiration };

            // case 1: the triple renews another one already present
            if let Some(present_triple) = self.timestamped_contents.iter()
                    .find(|timestamped_triple| timestamped_triple.triple == expiration_triple.triple) {

                if expiration > present_triple.timestamp {
                    let to_remove = present_triple.clone();

                    self.timestamped_contents.remove(&to_remove);
                    self.timestamped_contents.insert(expiration_triple);

                    inserted.insert(triple.triple);
                }
            }
            // case 2: the triple was not already present
            else {
                self.timestamped_contents.insert(expiration_triple);

                self.reasoner.index_manager.insert(&triple.triple);
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

                        let expiration_triple = TimestampedTriple { triple: inferred.clone(), timestamp: min_timestamp };

                        // case 1: the triple renews another one already present
                        if let Some(present_triple) = self.timestamped_contents.iter()
                                .find(|timestamped_triple| timestamped_triple.triple == expiration_triple.triple) {

                            if min_timestamp > present_triple.timestamp {
                                let to_remove = present_triple.clone();

                                self.timestamped_contents.remove(&to_remove);
                                self.timestamped_contents.insert(expiration_triple.clone());

                                inserted.insert(inferred);
                            }
                        }
                        // case 2: the triple was not already present
                        else {
                            self.timestamped_contents.insert(expiration_triple.clone());

                            self.reasoner.index_manager.insert(&inferred);
                            materialisation.insert(inferred.clone());
                            inserted.insert(inferred);
                        }

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
        for triple in self.timestamped_contents.clone() {
            content.add(triple, self.app_time);
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
struct Consumer<I>
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
    fn new() -> Consumer<I> {
        Consumer {
            inner: Arc::new(ConsumerInner {
                data: Mutex::new(Vec::new()),
            }),
        }
    }
    fn start(&self, receiver: Receiver<ContentContainer<I>>) {
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
        // TODO: is dit correct?
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