pub mod attention;
pub mod golden;
pub mod kernels;
pub mod matmul;
pub mod model;
pub mod profile;
pub mod rope;
pub mod sample;

pub use attention::attention_causal;
pub use golden::{GoldenSet, WorkloadIndex};
pub use kernels::{rmsnorm, rmsnorm_par, softmax_rows, sumsq};
pub use matmul::{linear, matmul, transpose};
pub use model::{Engine, ModelSpec, Weights};
pub use rope::RopeTable;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
