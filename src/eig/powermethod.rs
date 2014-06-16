use std::num;
use std::num::{Zero, One};

use Matrix;

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
// Inverse power method:
// Assuming linearly independent eigenvectors: v_1, v_2, ..., v_n
// Assuming eigenvalues: 1/l_1, 1/l_2, ..., 1/l_n
// Let |1/l_n| > |1/l_{n-1}| > ... > |l_1|
//
// In Inverse power method, we iterate on A^-1 instead of A, which has the same eigenvectors, with inverse eigenvalues:
//   Av = lv
//   Assume: A^-1v = 1/l v
//           AA^-1v = A 1/l v
//           v = 1/l A v = 1/l lv = v.
//
// For some vector q:
//   q = c_n * v_n + ... + c_1 * v_1,		c_n != 0
// Then the inverse power method will converge to the eigenvector with the smallest eigenvalue of A (v_n).
//
// The convergence ratio is: |l_n/l_{n-1}|. If |l_{n-1}| >> |l_n||, the convergence will happen fast.
// Thus, if we shift l_n so that it will be very close to zero, the convergence will happen fast. We can shift
// the eigenvalue by using:
//   Av = lv
//   (A - rI)v = Av - rIv = Av - rv = lv - rv = (l - r)v
// Thus, the matrix (A - rI) has the same eigenvectors as A, with eigenvalues that are shifted down by r.
// Using shifting with inverse power method is called shift-and-invert strategy.
//
// Each step of the inverse and shift iteration is then essentially:
//   q_{j+1} = (A - rI)^-1 q_j / s_{j+1}			, where s_{j+1} is a scaling factor as described with direct power method.
// Note that instead of calculating the inverse: (A - rI)^-1, it's likely faster and more convenient to solve:
//   (A - rI) q'_{j+1} = q_j
// And then set:
//   q_{j+1} = q'_{j+1} / s_{j+1}.
//
// Shifting allows us to find any eigenvector, by shifting the specific eigenvalue to be the smallest. However, in
// order to do this, we need to have a good approximation of the associated eigenvalue to begin with.
//
// The shift-and-inverse stategy gives us a way to calculate an eigenvector, if we have an approximation of an eigenvalue.
// The Rayleigh quotient is a way to calculate an approximation of an eigenvalue, if we have an approximation of an eigenvector.
//
// Note: At each step of shift-and-inverse strategy, we could use a different value for the shift. This can be useful, if we
// don't have an estimate of the eigenvalue to start with to use as a shift: (Rayleigh quotient iteration).
//   - We'll start with a guess of the eigenvector.
//   - We'll use the Rayleigh quotient to approximate an eigenvalue for our guess-approximation of the eigenvector.
//   - We'll use the approximation of the eigenvalue to shift and come up with the next iteration of the eigenvector using shift-and-inverse.
//     (We might want to normalize the vector here, though not strictly necessary).
//   - We reiterate. This algorithm is not guaranteed to converge, but it seldom fails. And when it converges, it generally does so fast.
//
// The Rayleight quotient approximates the eigenvalue by calculating the least squares estimate for Aq = rq.
// Using normal equations we get:
//   r = q'Aq / (q'q).			// Note that if we normalized q, then we'll just have:  r = q'Aq.
//
pub fn power_method
    <T : Mul<T, T> + Div<T, T> + Add<T, T> + Sub<T, T> + Zero + One + Ord + Signed + Clone + Float>
    (m : &Matrix<T>, q : &Matrix<T>, eps : T) -> Matrix<T> {
  assert!(m.cols() == m.rows());
  assert!(m.cols() == q.rows());
  assert!(q.cols() == 1);

  fn max_elem<T : Ord + Signed + Clone>(v : &Matrix<T>) -> T {
    let mut current_max = num::abs(v.get_data().get(0).clone());
    for _ in range(1, v.rows()) {
      let v = num::abs(v.get_data().get(1).clone());
      if v > current_max {
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
    if (nv - v).length() <= eps {
      return nv;
    }
    v = nv;
  }
}
