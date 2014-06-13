#![crate_type = "lib"]
#![crate_id = "la#0.1"]

#![feature(globs)]
#![feature(macro_rules)]

pub mod approxeq;

pub mod matrix;

pub mod util;

pub mod decomp {
  pub mod cholesky;
  pub mod eigen;
  pub mod lu;
  pub mod qr;
  pub mod svd;
}

pub mod eig {
  pub mod powermethod;
}

