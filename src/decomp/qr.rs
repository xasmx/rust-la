use std::num::{One, Zero};

use super::super::matrix::*;
use super::super::util::{alloc_dirty_vec, hypot};

pub struct QRDecomposition<T> {
  qr : Matrix<T>,
  rdiag : ~[T]
}

// Ported from JAMA.
// QR Decomposition.
//
// For an m-by-n matrix A with m >= n, the QR decomposition is an m-by-n
// orthogonal matrix Q and an n-by-n upper triangular matrix R so that
// A = Q*R.
//
// The QR decompostion always exists, even if the matrix does not have
// full rank.  The primary use of the QR decomposition is in the least
// squares solution of nonsquare systems of simultaneous linear equations.
// This will fail if is_full_rank() returns false.
impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Algebraic + Orderable + Signed> QRDecomposition<T> {
  pub fn new(m : &Matrix<T>) -> QRDecomposition<T> {
    let mut qrdata = m.data.clone();
    let n = m.noCols;
    let m = m.noRows;
    let mut rdiag = alloc_dirty_vec(n);

    for k in range(0u, n) {
      // Compute 2-norm of k-th column without under/overflow.
      let mut nrm : T = Zero::zero();
      for i in range(k, m) {
        nrm = hypot(nrm, qrdata[i * n + k].clone());
      }

      if(nrm != Zero::zero()) {
        // Form k-th Householder vector.
        if(qrdata[k * n + k] < Zero::zero()) {
          nrm = - nrm;
        }
        for i in range(k, m) {
          qrdata[i * n + k] = qrdata[i * n + k] / nrm;
        }
        qrdata[k * n + k] = qrdata[k * n + k] + One::one();

        // Apply transformation to remaining columns.
        for j in range(k + 1, n) {
          let mut s : T = Zero::zero();
          for i in range(k, m) {
            s = s + qrdata[i * n + k] * qrdata[i * n + j];
          }
          s = - s / qrdata[k * n + k];
          for i in range(k, m) {
            qrdata[i * n + j] = qrdata[i * n + j] + s * qrdata[i * n + k];
          }
        }
      }

      rdiag[k] = - nrm;
    }

    QRDecomposition { qr : Matrix { noRows : m, noCols : n, data : qrdata }, rdiag : rdiag }
  }

  pub fn is_full_rank(&self) -> bool {
    for j in range(0u, self.qr.noCols) {
      if(self.rdiag[j] == Zero::zero()) {
        return false;
      }
    }
    return true;
  }

  // Return the Householder vectors
  // Lower trapezoidal matrix whose columns define the reflections
  pub fn get_h(&self) -> Matrix<T> {
    let mut hdata = alloc_dirty_vec(self.qr.noRows * self.qr.noCols);

    let m = self.qr.noRows;
    let n = self.qr.noCols;

    for i in range(0u, m) {
      for j in range(0u, n) {
        hdata[i * self.qr.noCols + j] = if(i >= j) { self.qr.data[i * self.qr.noCols + j].clone() } else { Zero::zero() }
      }
    }

    Matrix { noRows : self.qr.noRows, noCols : self.qr.noCols, data : hdata }
  }

  // Return the upper triangular factor
  pub fn get_r(&self) -> Matrix<T> {
    let n = self.qr.noCols;
    let mut rdata = alloc_dirty_vec(n * n);

    for i in range(0u, n) {
      for j in range(0u, n) {
        if(i < j) {
          rdata[i * n + j] = self.qr.data[i * self.qr.noCols + j].clone();
        } else if(i == j) {
          rdata[i * n + j] = self.rdiag[i].clone();
        } else {
          rdata[i * n + j] = Zero::zero();
        }
      }
    }

    Matrix { noRows : n, noCols : n, data : rdata }
  }

  // Generate and return the (economy-sized) orthogonal factor
  pub fn get_q(&self) -> Matrix<T> {
    let n = self.qr.noCols;
    let m = self.qr.noRows;
    let mut qdata = alloc_dirty_vec(m * n);

    for k in range(0u, n).invert() {
      for i in range(0u, m) {
        qdata[i * n + k] = Zero::zero();
      }
      qdata[k * n + k] = One::one();
      for j in range(k, n) {
        if(self.qr.data[k * n + k] != Zero::zero()) {
          let mut s : T = Zero::zero();
          for i in range(k, m) {
            s = s + self.qr.data[i * n + k] * qdata[i * n + j];
          }
          s = - s / self.qr.data[k * n + k];
          for i in range(k, m) {
            qdata[i * n + j] = qdata[i * n + j] + s * self.qr.data[i * n + k];
          }
        }
      }
    }

    Matrix { noRows : m, noCols : n, data : qdata }
  }

  // Least squares solution of A*X = B
  // B : A Matrix with as many rows as A and any number of columns.
  // returns X that minimizes the two norm of Q*R*X-B.
  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    assert!(b.noRows == self.qr.noRows);
    if(!self.is_full_rank()) {
      return None
    }

    let nx = b.noCols;
    let mut xdata = b.data.clone();

    let n = self.qr.noCols;
    let m = self.qr.noRows;

    // Compute Y = transpose(Q)*B
    for k in range(0u, n) {
      for j in range(0u, nx) {
        let mut s : T = Zero::zero();
        for i in range(k, m) {
          s = s + self.qr.data[i * self.qr.noCols + k] * xdata[i * nx + j];
        }
        s = - s / self.qr.data[k * self.qr.noCols + k];
        for i in range(k, m) {
          xdata[i * nx + j] = xdata[i * nx + j] + s * self.qr.data[i * self.qr.noCols + k];
        }
      }
    }

    // Solve R*X = Y;
    for k in range(0u, n).invert() {
      for j in range(0u, nx) {
        xdata[k * nx + j] = xdata[k * nx + j] / self.rdiag[k];
      }
      for i in range(0u, k) {
        for j in range(0u, nx) {
          xdata[i * nx + j] = xdata[i * nx + j] - xdata[k * nx + j] * self.qr.data[i * self.qr.noCols + k];
        }
      }
    }

    Some(Matrix { noRows : self.qr.noCols, noCols : b.noCols, data : xdata })
  }
}

#[test]
fn qr_test() {
  let a = matrix(3, 3, ~[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}

#[test]
fn qr_test__m_over_n() {
  let a = matrix(3, 2, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}

/*
// FIXME: Add support for n over m case.
#[test]
fn qr_test__n_over_m() {
  let a = matrix(2, 3, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}
*/
