use std::cmp;
use std::num;
use std::num::{One, Zero};
use std::vec;

use approxeq::ApproxEq;
use matrix::*;
use util::{alloc_dirty_vec};

pub struct QRDecomposition<T> {
  qr : Matrix<T>,
  rdiag : Vec<T>
}

// Based on Apache Commons Math and JAMA.
// QR Decomposition.
//
// For an m-by-n matrix A, the QR decomposition is an m-by-m orthogonal matrix Q
// and an m-by-n upper triangular (or trapezoidal) matrix R, so that A = Q*R.
//
// The QR decompostion always exists, even if the matrix does not have
// full rank.  The primary use of the QR decomposition is in the least
// squares solution of nonsquare systems of simultaneous linear equations.
// This will fail if is_full_rank() returns false.
impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed + Float> QRDecomposition<T> {
  pub fn new(m : &Matrix<T>) -> QRDecomposition<T> {
    // qr: The area above the diagonals stores the corresponding parts of the R matrix.
    //     Each column from diagonal down will store the k:th householder vector in the end. (The elements above are zero for the householder vectors).
    let mut qrdata = m.data.clone();
    let n = m.cols();
    let m = m.rows();
    let diag_count = cmp::min(m, n);
    // rdiag: Will store the diagonal elements of the R matrix.
    let mut rdiag = alloc_dirty_vec(diag_count);

    // Iterate over all columns with a diagonal element (columns 0 .. (diag_count - 1)), zeroing out the elements below the diagonal element.
    for minor in range(0, diag_count) {
      // Zero out elements below the diagonal for the k:th column of m.
      QRDecomposition::perform_householder_reflection(minor, &mut qrdata, m, n, &mut rdiag);
    }

    QRDecomposition { qr : matrix(m, n, qrdata), rdiag : rdiag }
  }

  // Find a reflection hyperplane, that will reflect the minor:th minor vector to (a 0 .. 0)^T and perform the reflection.
  fn perform_householder_reflection(minor : uint, qrdata : &mut Vec<T>, m : uint, n : uint, rdiag : &mut Vec<T>) {
    // Calculate the length of the minor:th minor column vector. As we are dealing with a sub matrix from the diagonal down and right,
    // the column vector corresponds to the row range minor .. (m - 1).
    let mut x_norm_sqr : T = num::zero();
    for i in range(minor, m) {
      let c = qrdata.get((i * n + minor) as uint).clone();
      x_norm_sqr = x_norm_sqr + c * c;
    }

    // We know the size of the reflected coordinate (the lenght of the column vector), but not the sign yet.
    // Reflection will flip the sign of the corresponding coordinate element, so sign of the reflected
    // coordinate will be the opposite of the current coordinate element.
    let a = if *qrdata.get((minor * n + minor) as uint) > num::zero() { - x_norm_sqr.sqrt() } else { x_norm_sqr.sqrt() };
    *rdiag.get_mut(minor) = a.clone();

    // If the length of the column vector is zero, the column is already zero and there's nothing to do.
    if a != num::zero() {
      // Transform qrdata[minor .. (m - 1)][minor] to be the minor:th Householder vector.

      // u is the vector from the reflection point (a * e) to the current point (x).
      // As we are reflecting to an minor:th axis, e = (1 0 0 .. 0 0)^T
      // u = - a * e + x = x - a * e
      //
      // Note that:
      // |u|^2 = <x - ae, x - ae>
      //       = <x, x> - 2a<x, e> + a^2<e, e>
      //       = a^2    - 2a<x, e> + a^2
      //       = 2a^2 - 2a<x, e>
      //       = 2a^2 - 2a*qrdata[k * n + k]	// As <x, e> is the projection of x to the axis aligned unit vector e.
      //       = 2a(a - qrdata[k * n + k])
      *qrdata.get_mut(minor * n + minor) = *qrdata.get(minor * n + minor) - a;

      // Note that now:
      // |u|^2 = 2a(a - (qrdata[k * n + k] + a))
      //       = 2a(- qrdata[k * n + k])
      //       = -2a*qrdata[k * n + k]

      // Transform the rest of the columns of the minor:
      // The reflection matrix is:
      //   H = I - 2uu'/|u|^2.
      //   Hx = (I - 2uu'/|u|^2)x
      //      = x - 2uu'x/|u|^2
      //      = x - 2<x,u>/|u|^2 u
      //      = x - 2<x,u>/(-2a*qrdata[k * n + k]) u
      //      = x + <x,u>/(a*qrdata[k * n + k]) u
      //      = x + factor * u
      for column in range(minor + 1, n) {
        // factor = <x, u>/(a * qrdata[k * n + k])
        let mut x_dot_u : T = num::zero();
        for row in range(minor, m) {
          x_dot_u = x_dot_u + *qrdata.get(row * n + minor) * *qrdata.get(row * n + column);
        }
        let factor = x_dot_u / (a * *qrdata.get(minor * n + minor));

        // Hx = x + factor * u
        for row in range(minor, m) {
          *qrdata.get_mut(row * n + column) = *qrdata.get(row * n + column) + factor * *qrdata.get(row * n + minor);
        }
      }
    }
  }

  pub fn is_full_rank(&self) -> bool {
    for j in range(0u, self.qr.cols()) {
      if *self.rdiag.get(j) == num::zero() {
        return false;
      }
    }
    return true;
  }

  // Return the Householder vectors
  // Lower trapezoidal matrix whose columns define the reflections
  pub fn get_h(&self) -> Matrix<T> {
    let mut hdata = alloc_dirty_vec(self.qr.rows() * self.qr.cols());

    let m = self.qr.rows();
    let n = self.qr.cols();

    for i in range(0u, m) {
      for j in range(0u, n) {
        *hdata.get_mut(i * self.qr.cols() + j) = if i >= j { self.qr.data.get(i * self.qr.cols() + j).clone() } else { num::zero() }
      }
    }

    matrix(self.qr.rows(), self.qr.cols(), hdata)
  }

  // Return the upper triangular factor
  pub fn get_r(&self) -> Matrix<T> {
    let m = self.qr.rows();
    let n = self.qr.cols();
    let mut rdata = alloc_dirty_vec(m * n);

    for i in range(0u, m) {
      for j in range(0u, n) {
        *rdata.get_mut(i * n + j) = if i < j { self.qr.data.get(i * n + j).clone() }
                           else if i == j { self.rdiag.get(i).clone() }
                           else { num::zero() };
      }
    }
    matrix(m, n, rdata)
  }

  // Generate and return the (economy-sized) orthogonal factor
  pub fn get_q(&self) -> Matrix<T> {
    let n = self.qr.cols();
    let m = self.qr.rows();
    let mut qdata = vec::Vec::from_elem(m * m, num::zero());

    // Set the diagonal elements to 1
    for minor in range(0, cmp::min(m, n)) {
      *qdata.get_mut(minor * m + minor) = num::one();
    }

    // Successively apply the iverses of the reflections in reverse order to qdata (identity)
    // to inverse the changes we did to when creating the triangular matrix R, transforming
    // the identity matrix to Q: (Note that a reflection matrix is it's own inverse).
    //   Q = Q_1_inv(Q_2_inv(...(Q_m_inv I))) = Q_1(Q_2(...(Q_m I)))
    for minor in range(0u, cmp::min(m, n)).rev() {
      if *self.qr.data.get(minor * n + minor) != num::zero() {
        // |u|^2 = -2a*qrdata[minor * n + minor]
        //       = -2 * rdiag[minor] * qrdata[minor * n + minor]
        //
        // The reflection matrix is:
        //   H = I - 2uu'/|u|^2.
        //   Hx = (I - 2uu'/|u|^2)x
        //      = x - 2uu'x/|u|^2
        //      = x - 2<x,u>/|u|^2 u
        //      = x - 2<x,u>/(-2a*qrdata[minor * n + minor]) u
        //      = x + <x,u>/(a*qrdata[minor * n + minor]) u
        //      = x + factor * u
        // Iterate over columns k .. (m - 1) of Q.
        for column in range(minor, m) {
          let mut x_dot_u : T = num::zero();
          for row in range(minor, m) {
            x_dot_u = x_dot_u + *self.qr.data.get(row * n + minor) * *qdata.get(row * m + column);
          }
          let factor = x_dot_u / (*self.rdiag.get(minor) * *self.qr.data.get(minor * n + minor));

          for row in range(minor, m) {
            *qdata.get_mut(row * m + column) = *qdata.get(row * m + column) + factor * *self.qr.data.get(row * n + minor);
          }
        }
      }
    }

    matrix(m, m, qdata)
  }

  // Least squares solution of A*X = B
  // B : A Matrix with as many rows as A and any number of columns.
  // returns X that minimizes the two norm of Q*R*X-B.
  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    assert!(b.rows() == self.qr.rows());
    if !self.is_full_rank() {
      return None
    }

    let nx = b.cols();
    let mut xdata = b.data.clone();

    let n = self.qr.cols();
    let m = self.qr.rows();

    // Compute Y = transpose(Q)*B
    for k in range(0u, n) {
      for j in range(0u, nx) {
        let mut s : T = num::zero();
        for i in range(k, m) {
          s = s + *self.qr.data.get(i * self.qr.cols() + k) * *xdata.get(i * nx + j);
        }
        s = - s / *self.qr.data.get(k * self.qr.cols() + k);
        for i in range(k, m) {
          *xdata.get_mut(i * nx + j) = *xdata.get(i * nx + j) + s * *self.qr.data.get(i * self.qr.cols() + k);
        }
      }
    }

    // Solve R*X = Y;
    for k in range(0u, n).rev() {
      for j in range(0u, nx) {
        *xdata.get_mut(k * nx + j) = *xdata.get(k * nx + j) / *self.rdiag.get(k);
      }
      for i in range(0u, k) {
        for j in range(0u, nx) {
          *xdata.get_mut(i * nx + j) = *xdata.get(i * nx + j) - *xdata.get(k * nx + j) * *self.qr.data.get(i * self.qr.cols() + k);
        }
      }
    }

    Some(matrix(self.qr.cols(), b.cols(), xdata))
  }
}

#[test]
fn qr_test() {
  let a = matrix(3, 3, vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}

#[test]
fn qr_test_m_over_n() {
  let a = matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}

#[test]
fn qr_test_n_over_m() {
  let a = matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let qr = QRDecomposition::new(&a);
  assert!((qr.get_q() * qr.get_r()).approx_eq(&a));
}
