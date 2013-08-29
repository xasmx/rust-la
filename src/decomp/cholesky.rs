use std::num;
use std::num::{One, Zero};

use super::super::matrix::*;
use super::super::util::{alloc_dirty_vec};

pub struct CholeskyDecomposition<T> {
  l : Matrix<T>
}

// Ported from JAMA.
// Cholesky Decomposition.
//
// For a symmetric, positive definite matrix A, the Cholesky decomposition
// is an lower triangular matrix L so that A = L*L'.
//
// If the matrix is not symmetric or positive definite, None is returned.
impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Algebraic> CholeskyDecomposition<T> {
  fn new(m : &Matrix<T>) -> Option<CholeskyDecomposition<T>> {
    if(m.noRows != m.noCols) {
      return None
    }
    let n = m.noRows;
    let mut data : ~[T] = alloc_dirty_vec(n * n);

    for j in range(0u, n) {
      let rowjIdx = j * n;
      let mut d : T = Zero::zero();
      for k in range(0u, j) {
        let rowkIdx = k * n;
        let mut s : T = Zero::zero();
        for i in range(0u, k) {
          s = s + data[rowkIdx + i] * data[rowjIdx + i];
        }
        s = (m.data[rowjIdx + k] - s) / data[rowkIdx + k];
        data[rowjIdx + k] = s.clone();
        d = d + s * s;
        if(m.data[rowkIdx + j] != m.data[rowjIdx + k]) {
          return None
        }
      }

      d = m.data[rowjIdx + j] - d;
      if(d <= Zero::zero()) {
        return None
      }
      data[rowjIdx + j] = num::sqrt(if d > Zero::zero() { d } else { Zero::zero() });
      for k in range(j + 1, n) {
        data[rowjIdx + k] = Zero::zero();
      }
    }

    Some(CholeskyDecomposition { l : Matrix { noRows : n, noCols : n, data : data } })
  }

  #[inline]
  fn get_l<'lt>(&'lt self) -> &'lt Matrix<T> { &self.l }

  // Solve: Ax = b
  pub fn solve(&self, b : &Matrix<T>) -> Matrix<T> {
    let l = &self.l;
    assert!(l.noRows == b.noRows);
    let n = l.noRows;
    let mut xdata = b.data.clone();
    let nx = b.noCols;

    // Solve L*Y = B
    for k in range(0u, n) {
      for j in range(0u, nx) {
        for i in range(0u, k) {
          xdata[k * nx + j] = xdata[k * nx + j] - xdata[i * nx + j] * l.data[k * n + i];
        }
        xdata[k * nx + j] = xdata[k * nx + j] / l.data[k * n + k];
      }
    }

    // Solve L'*X = Y
    for k in range(0u, n).invert() {
      for j in range(0u, nx) {
        for i in range(k + 1, n) {
          xdata[k * nx + j] = xdata[k * nx + j] - xdata[i * nx + j] * l.data[i * n + k];
        }
        xdata[k * nx + j] = xdata[k * nx + j] / l.data[k * n + k];
      }
    }

    Matrix { noRows : n, noCols : nx, data : xdata }
  }
}

#[test]
fn cholesky_square_pos_def_test() {
  let a = matrix(3, 3, ~[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]);
  match(CholeskyDecomposition::new(&a)) {
    None => assert!(false),
    Some(c) => { assert!(c.get_l() * c.get_l().t() == a) }
  }
}

#[test]
fn cholesky_not_pos_def_test() {
  let a = matrix(3, 3, ~[4.0, 12.0, -16.0, 12.0, 37.0, 43.0, -16.0, 43.0, 98.0]);
  match(CholeskyDecomposition::new(&a)) {
    None => (),
    _ => assert!(false)
  }
}

#[test]
fn cholesky_not_square_test() {
  let a = matrix(2, 3, ~[4.0, 12.0, -16.0, 12.0, 37.0, 43.0]);
  match(CholeskyDecomposition::new(&a)) {
    None => (),
    _ => assert!(false)
  }
}

#[test]
fn cholesky_solve_test() {
  let a = matrix(3, 3, ~[2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
  match(CholeskyDecomposition::new(&a)) {
    None => assert!(false),
    Some(c) => {
      let b = vector(~[1.0, 2.0, 3.0]);
      let x = c.solve(&b);
      assert!(x.approx_eq(&vector(~[-1.0, 3.0, 3.0])));
    }
  }
}

#[test]
#[should_fail]
fn cholesky_solve_test__incompatible() {
  let a = matrix(3, 3, ~[2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
  match(CholeskyDecomposition::new(&a)) {
    None => assert!(false),
    Some(c) => {
      let b = vector(~[1.0, 2.0, 3.0, 4.0]);
      let _ = c.solve(&b);
    }
  }
}
