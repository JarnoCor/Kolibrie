//! A module containing useful structs and methods for performing
//! Incremental View Maintenance (IVM) in various ways.
//!
//! There is currently only one way of performing IVM, using a counting-based approach.
//! More methods are coming soon.

mod counting;
mod dred;

pub use counting::CountingMaintenance;
pub use counting::Multiset;
pub use counting::multiset_add;
pub use counting::multiset_subtract;
