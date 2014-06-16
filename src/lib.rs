#![crate_type = "lib"]
#![crate_id = "la#0.1"]

#![feature(globs)]
#![feature(macro_rules)]

pub use approxeq::ApproxEq;
pub use matrix::Matrix;
pub use decomp::cholesky::CholeskyDecomposition;
pub use decomp::eigen::EigenDecomposition;
pub use decomp::lu::LUDecomposition;
pub use decomp::qr::QRDecomposition;
pub use decomp::svd::SVD;

mod macros;
mod approxeq;
mod matrix;
mod internalutil;
pub mod util;

mod decomp {
  pub mod cholesky;
  pub mod eigen;
  pub mod lu;
  pub mod qr;
  pub mod svd;
}

pub mod eig {
  pub mod powermethod;
}

