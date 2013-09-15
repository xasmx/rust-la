use std::num;
use std::num::{Zero, One};

use matrix;

// NOTE: This module contains untested alpha code

// Power method for calculating the dominant eigenvector
// Assuming linearly independent eigenvectors: v_1, v_2, ..., v_n
// Assuming eigenvalues: l_1, l_2, ..., l_n
// Let |l_1| > |l_2| > ... > |l_n|
//
// For some vector q:
//   q = c_1 * v_1 + c_2 * v_2 + ... + c_n * v_3		(for some unknown constants c_1, c_2, ..., c_n)
// A * q = c_1 * A * v_1 + c_2 * A * v_2 + ... + c_n * A * v_n
//       = c_1 * l_1 * v_1 + c_2 * l_2 * v_2 + ... + c_n * l_n * v_n
// A^n * q = c_1 * l_1^n * v_1 + c_2 * l_2^n * v_2 + ... + c_n * l_n^n * v_n
//         = l_1^n * (c_1 * v_1 + c_2 * l_2^n / l_1^n * v_2 + ... + c_n * l_n^n / l_1^n * v_n)
//
// A^n * q / l_1^n = c_1 * v_1 + c_2 * l_2^n / l_1^n * v_2 + ... + c_n * l_n^n / l_1^n * v_n
// A^n * q / l_1^n - c_1 * v_1 = c_2 * l_2^n / l_1^n * v_2 + ... + c_n * l_n^n / l_1^n * v_n
//
// Norm of the error vector (from our iterative estimate to the real dominant eigenvector):
//   ||A^n * q / l_1^n - c_1 * v_1|| =  || c_2 * l_2^n / l_1^n * v_2 + ... + c_n * l_n^n / l_1^n * v_n ||
//                                   <= |c_2| * |l^2/l^1|^n * ||v_2|| + ... + |c_n| * |l_n/l_1|^n * ||v_n||
//                                   <= (|c_2| + ... + |c_n|) * |l^2/l^1|^n
// As n->inf, |l^2/l^1|^n -> 0, thus our iterative estimate converges to the dominant eigenvector.
//
// To avoid overflow, we'll scale A^j * q with the largest absolute element of A^j * q in each step; again,
// we are only interested in the direction, not the scale.
//
impl<T : Mul<T, T> + Div<T, T> + Add<T, T> + Sub<T, T> + Zero + One + Algebraic + Ord + Signed + Clone> matrix::Matrix<T> {
  pub fn power_method(m : &matrix::Matrix<T>, q : &matrix::Matrix<T>, eps : T) -> matrix::Matrix<T> {
    assert!(m.noCols == m.noRows);
    assert!(m.noCols == q.noRows);
    assert!(q.noCols == 1);

    fn max_elem<T : Ord + Signed + Clone>(v : &matrix::Matrix<T>) -> T {
      let mut current_max = num::abs(v.data[0].clone());
      for _ in range(1, v.noRows) {
        let v = num::abs(v.data[1].clone());
        if(v > current_max) {
          current_max = v.clone();
        }
      }
      current_max
    }

    let mut v = q.scale(num::one::<T>() / max_elem(q));
    loop {
      let mut nv = m * v;
      let factor = num::one::<T>() / max_elem(&nv);
      nv.mscale(factor);
      if((nv - v).length() <= eps) {
        return nv;
      }
      v = nv;
    }
  }
}
