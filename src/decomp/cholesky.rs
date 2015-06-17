use num;
use num::Float;

use ApproxEq;
use Matrix;
use internalutil::{alloc_dirty_vec};

/// Cholesky Decomposition (for a real valued matrix).
///
/// Initial implementation based on JAMA.
///
/// For a symmetric, positive definite matrix A, the Cholesky decomposition
/// is an lower triangular matrix L so that A = L*L'.
///
/// If the matrix is not symmetric or positive definite, None is returned.
//
// Solve L one row at a time from row 1 to row n:
//   A = L * L'
//
//   a11 a12 .. .. a1n   L11   0  .. ..   0   L11 L21 .. .. Ln1
//   a21 a22 .. .. a2n   L21 L22  .. ..   0     0 L22 .. .. Ln2
//   ..  ..  .. .. ..  =  ..  ..  .. ..   0 *  ..  .. .. ..  ..
//   ..  ..  .. .. ..     ..  ..  .. ..   0    ..  .. .. ..  ..
//   an1 an2 .. .. ann   Ln1 Ln2  .. .. Lnn     0   0  0  0 Lnn
//
// Solve L[1][_]:
//   A[1][1] = L[1][1] * L[1][1]
//   => L[1][1] = sqrt(A[1][1]).
//   => L[1][2 .. n] = 0.
//
// Solve L[2][_]:
//   A[2][1] = L[2][1] * L[1][1]
//   => L[2][1] = A[2][1] / L[1][1].
//   A[2][2] = L[2][1] * L[2][1] + L[2][2] * L[2][2]
//   => L[2][2] = sqrt(A[2][2] - L[2][1] * L[2][1]).
//   => L[2][3 .. n] = 0.
//
// And in general: j = { 1 .. n }, i = { 1 .. n}
//   For j < i:
//     L[j][i] = 0.
//   For j = i:
//     L[j][i] = sqrt(A[j][i] - SUM { L[j][0 .. (j - 1)] * L'[(0 .. (j - 1)][j] })
//             = sqrt(A[j][j] - SUM { L[j][0 .. (j - 1)]^2 }
//   For j > i:
//     L[j][i] = (A[j][i] - SUM { L[i][0 .. (i - 1)] * L'[0 .. (i - 1)][j]}) / L[i][i].
//             = (A[j][i] - SUM { L[i][0 .. (i - 1)] * L[j][0 .. (i - 1)]) / L[i][i].
//
// As long as we follow the up->down, left->right order to compute the values, all the elements of L accessed on the right
// side will have been computed by the time they are needed.
//
pub struct CholeskyDecomposition<T> {
  l : Matrix<T>
}

impl<T : Float + ApproxEq<T>> CholeskyDecomposition<T> {
  pub fn new(m : &Matrix<T>) -> Option<CholeskyDecomposition<T>> {
    if m.rows() != m.cols() {
      return None
    }

    let n = m.rows();
    let mut data : Vec<T> = alloc_dirty_vec(n * n);

    for j in 0..n {
      // Solve row L[j].

      // d = SUM { L[j][0 .. (j - 1)]^2 }
      let mut d : T = num::zero();

      // Solve L[j][0 .. (j - 1)].
      for k in 0..j {
        // Solve L[j][k].

        // s = SUM { L[k][0 .. (k - 1)] * L'[0 .. (k - 1)][j] }
        //   = SUM { L[k][0 .. (k - 1)] * L[j][0 .. (k - 1) }
        let mut s : T = num::zero();
        for i in 0..k {
          unsafe { s = s + data.get_unchecked(k * n + i).clone() * data.get_unchecked(j * n + i).clone(); }
        }

        // L[j][k] = (A[j][k] - SUM { L[k][0 .. (k - 1)] * L'[0 .. (k - 1)][j] }) / L[k][k].
        unsafe { s = (m.get(j, k) - s) / data.get_unchecked(k * n + k).clone(); }
        unsafe { *data.get_unchecked_mut(j * n + k) = s.clone(); }

        // Gather a sum of squres of L[j][0 .. (j - 1)] to d. Note: s = L[j][k].
        d = d + s * s;

        // Make sure input matrix is symmetric; Cholesky decomposition is not defined for non-symmetric matrixes.
        if m.get(k, j) != m.get(j, k) {
          return None
        }
      }

      // Solve L[j][j]. (Diagonals).
      // L[j][j] = sqrt(A[j][j] - SUM { L[j][0 .. (j - 1)]^2 }).
      d = m.get(j, j) - d;
      if d <= num::zero() {
        // A is not positive definite; Cholesky decomposition does not exists.
        return None
      }
      unsafe { *data.get_unchecked_mut(j * n + j) = d.sqrt(); }

      // Solve L[j][(j + 1) .. (n - 1)]. (Always zero as L is lower triangular).
      for k in (j + 1)..n {
        unsafe { *data.get_unchecked_mut(j * n + k) = num::zero(); }
      }
    }

    Some(CholeskyDecomposition { l : Matrix::new(n, n, data) })
  }

  #[inline]
  pub fn get_l<'lt>(&'lt self) -> &'lt Matrix<T> { &self.l }

  // Solve: Ax = b
  pub fn solve(&self, b : &Matrix<T>) -> Matrix<T> {
    let l = &self.l;
    assert!(l.rows() == b.rows());
    let n = l.rows();
    let mut xdata = b.get_data().clone();
    let nx = b.cols();

    // Solve L*Y = B
    for k in 0..n {
      for j in 0..nx {
        for i in 0..k {
          unsafe { *xdata.get_unchecked_mut(k * nx + j) = xdata.get_unchecked(k * nx + j).clone() - xdata.get_unchecked(i * nx + j).clone() * l.get_data().get_unchecked(k * n + i).clone(); }
        }
        unsafe { *xdata.get_unchecked_mut(k * nx + j) = xdata.get_unchecked(k * nx + j).clone() / l.get_data().get_unchecked(k * n + k).clone(); }
      }
    }

    // Solve L'*X = Y
    for k in (0..n).rev() {
      for j in 0..nx {
        for i in (k + 1)..n {
          unsafe { *xdata.get_unchecked_mut(k * nx + j) = xdata.get_unchecked(k * nx + j).clone() - xdata.get_unchecked(i * nx + j).clone() * l.get_data().get_unchecked(i * n + k).clone(); }
        }
        unsafe { *xdata.get_unchecked_mut(k * nx + j) = xdata.get_unchecked(k * nx + j).clone() / l.get_data().get_unchecked(k * n + k).clone(); }
      }
    }

    Matrix::new(n, nx, xdata)
  }
}

#[test]
fn cholesky_square_pos_def_test() {
  let a = m!(4.0, 12.0, -16.0; 12.0, 37.0, -43.0; -16.0, -43.0, 98.0);
  let c = CholeskyDecomposition::new(&a).unwrap();
  assert!(c.get_l() * c.get_l().t() == a);
  assert!(*c.get_l().get_data() == vec![2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
}

#[test]
fn cholesky_not_pos_def_test() {
  let a = m!(4.0, 12.0, -16.0; 12.0, 37.0, 43.0; -16.0, 43.0, 98.0);
  assert!(CholeskyDecomposition::new(&a).is_none());
}

#[test]
fn cholesky_not_square_test() {
  let a = m!(4.0, 12.0, -16.0; 12.0, 37.0, 43.0);
  assert!(CholeskyDecomposition::new(&a).is_none());
}

#[test]
fn cholesky_solve_test() {
  let a = m!(2.0, 1.0, 0.0; 1.0, 1.0, 0.0; 0.0, 0.0, 1.0);
  let c = CholeskyDecomposition::new(&a).unwrap();
  let b = m!(1.0; 2.0; 3.0);
  assert!(c.solve(&b).approx_eq(&m!(-1.0; 3.0; 3.0)));
}

#[test]
#[should_panic]
fn cholesky_solve_test_incompatible() {
  let a = m!(2.0, 1.0, 0.0; 1.0, 1.0, 0.0; 0.0, 0.0, 1.0);
  let c = CholeskyDecomposition::new(&a).unwrap();
  let b = m!(1.0; 2.0; 3.0; 4.0);
  let _ = c.solve(&b);
}
