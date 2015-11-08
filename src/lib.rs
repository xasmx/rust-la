extern crate rand;
extern crate num;

pub use approxeq::ApproxEq;
pub use decomp::cholesky::CholeskyDecomposition;
pub use decomp::eigen::EigenDecomposition;
pub use decomp::lu::LUDecomposition;
pub use decomp::qr::QRDecomposition;
pub use decomp::svd::SVD;
pub use matrix::Matrix;

pub mod macros;
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

//pub mod eig {
//  pub mod powermethod;
//}

