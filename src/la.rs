#[crate_type = "lib"];
#[link(name = "la", vers = "0.1")];

pub mod matrix;

pub mod util;

pub mod decomp {
  pub mod cholesky;
  pub mod eigen;
  pub mod lu;
  pub mod qr;
  pub mod svd;
}
