#[cfg(feature = "profile")]
use std::collections::BTreeMap;
#[cfg(feature = "profile")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "profile")]
use std::time::Instant;

#[cfg(not(feature = "profile"))]
#[derive(Clone, Copy)]
pub struct NoStamp;

#[cfg(feature = "profile")]
type Stamp = Instant;
#[cfg(not(feature = "profile"))]
type Stamp = NoStamp;

#[cfg(feature = "profile")]
#[derive(Default)]
struct Counters {
    nanos: BTreeMap<&'static str, u128>,
    calls: BTreeMap<&'static str, u64>,
}

#[cfg(feature = "profile")]
fn counters() -> &'static Mutex<Counters> {
    static STORE: OnceLock<Mutex<Counters>> = OnceLock::new();
    STORE.get_or_init(Default::default)
}

pub fn enabled() -> bool {
    cfg!(feature = "profile")
}

#[inline]
pub fn start() -> Stamp {
    #[cfg(feature = "profile")]
    {
        Instant::now()
    }
    #[cfg(not(feature = "profile"))]
    {
        NoStamp
    }
}

#[inline]
pub fn record(name: &'static str, stamp: Stamp) {
    #[cfg(feature = "profile")]
    {
        let mut guard = counters()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard.nanos.entry(name).or_insert(0) += stamp.elapsed().as_nanos();
        *guard.calls.entry(name).or_insert(0) += 1;
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = (name, stamp);
    }
}

pub fn reset() {
    #[cfg(feature = "profile")]
    {
        let mut guard = counters()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.nanos.clear();
        guard.calls.clear();
    }
}

pub fn snapshot() -> Vec<(&'static str, f64, u64)> {
    #[cfg(feature = "profile")]
    {
        let guard = counters()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut rows: Vec<(&'static str, f64, u64)> = guard
            .nanos
            .iter()
            .map(|(name, nanos)| {
                (
                    *name,
                    *nanos as f64 / 1e6,
                    guard.calls.get(name).copied().unwrap_or(0),
                )
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rows
    }
    #[cfg(not(feature = "profile"))]
    {
        Vec::new()
    }
}
