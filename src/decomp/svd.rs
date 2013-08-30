use std::num;
use std::num::{One, Zero, NumCast};
use std::vec;

use super::super::matrix::*;
use super::super::util::{alloc_dirty_vec, hypot};

pub struct SVD<T> {
  u : Matrix<T>,
  v : Matrix<T>,
  s : ~[T],
  m : uint,
  n : uint
}

// Ported from JAMA.
// Singular Value Decomposition.
//
// For an m-by-n matrix A with m >= n, the singular value decomposition is
// an m-by-n orthogonal matrix U, an n-by-n diagonal matrix S, and
// an n-by-n orthogonal matrix V so that A = U*S*V'.
//
// The singular values, sigma[k] = S[k][k], are ordered so that
// sigma[0] >= sigma[1] >= ... >= sigma[n-1].
//
// The singular value decompostion always exists. The matrix condition number
// and the effective numerical rank can be computed from this decomposition.
impl<T : Num + NumCast + Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Algebraic + Signed + Orderable> SVD<T> {
  pub fn new(a : &Matrix<T>) -> SVD<T> {
    // Derived from LINPACK code.
    // Initialize.
    let mut adata = a.data.clone();
    let m = a.noRows;
    let n = a.noCols;

    assert!(m >= n);

    let nu = num::min(m, n);

    let slen = num::min(m + 1, n);
    let mut sdata : ~[T] = alloc_dirty_vec(slen);

    let ulen = m * nu;
    let mut udata = alloc_dirty_vec(ulen);

    let vlen = n * n;
    let mut vdata = alloc_dirty_vec(vlen);

    let mut edata = alloc_dirty_vec(n);
    let mut workdata : ~[T] = alloc_dirty_vec(m);

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    let nct = num::min(m - 1, n);
    let nrt = num::max(0 as int, num::min((n as int) - 2, m as int)) as uint;
    for k in range(0u, num::max(nct, nrt)) {
      if(k < nct) {
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        sdata[k] = Zero::zero();
        for i in range(k, m) {
          sdata[k] = hypot(sdata[k].clone(), adata[i * n + k].clone());
        }
        if(sdata[k] != Zero::zero()) {
          if(adata[k * n + k] < Zero::zero()) {
            sdata[k] = - sdata[k];
          }
          for i in range(k, m) {
            adata[i * n + k] = adata[i * n + k] / sdata[k];
          }
          adata[k * n + k] = adata[k * n + k] + One::one();
        }
        sdata[k] = - sdata[k];
      }
      for j in range(k + 1, n) {
        if((k < nct) && (sdata[k] != Zero::zero()))  {
          // Apply the transformation.
          let mut t : T = Zero::zero();
          for i in range(k, m) {
            t = t + adata[i * n + k] * adata[i * n + j];
          }
          t = - t / adata[k * n + k];
          for i in range(k, m) {
            adata[i * n + j] = adata[i * n + j] + t * adata[i * n + k];
          }
        }
        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.
        edata[j] = adata[k * n + j].clone();
      }

      if(k < nct) {
        // Place the transformation in U for subsequent back multiplication.
        for i in range(k, m) {
          udata[i * nu + k] = adata[i * n + k].clone();
        }
      }

      if(k < nrt) {
        // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        edata[k] = Zero::zero();
        for i in range(k + 1, n) {
          edata[k] = hypot(edata[k].clone(), edata[i].clone());
        }
        if(edata[k] != Zero::zero()) {
          if(edata[k + 1] < Zero::zero()) {
            edata[k] = - edata[k];
          }
          for i in range(k + 1, n) {
            edata[i] = edata[i] / edata[k];
          }
          edata[k + 1] = edata[k + 1] + One::one();
        }
        edata[k] = - edata[k];
        if((k + 1 < m) && (edata[k] != Zero::zero())) {
          // Apply the transformation.
          for i in range(k + 1, m) {
            workdata[i] = Zero::zero();
          }
          for j in range(k + 1, n) {
            for i in range(k + 1, m) {
              workdata[i] = workdata[i] + edata[j] * adata[i * n + j];
            }
          }
          for j in range(k + 1, n) {
            let t = - edata[j] / edata[k + 1];
            for i in range(k + 1, m) {
              adata[i * n + j] = adata[i * n + j] + t * workdata[i];
            }
          }
        }

        // Place the transformation in V for subsequent back multiplication.
        for i in range(k + 1, n) {
          vdata[i * n + k] = edata[i].clone();
        }
      }
    }

    // Set up the final bidiagonal matrix or order p.
    let mut p = num::min(n, m + 1);
    if(nct < n) {
      sdata[nct] = adata[nct * n + nct].clone();
    }
    if(m < p) {
      sdata[p - 1] = Zero::zero();
    }
    if(nrt + 1 < p) {
      edata[nrt] = adata[nrt * n + (p - 1)].clone();
    }
    edata[p - 1] = Zero::zero();

    // Generate U.
    for j in range(nct, nu) {
      for i in range(0u, m) {
        udata[i * nu + j] = Zero::zero();
      }
      udata[j * nu + j] = One::one();
    }
    for k in range(0u, nct).invert() {
      if(sdata[k] != Zero::zero()) {
        for j in range(k + 1, nu) {
          let mut t : T = Zero::zero();
          for i in range(k, m) {
            t = t + udata[i * nu + k] * udata[i * nu + j];
          }
          t = - t / udata[k * nu + k];
          for i in range(k, m) {
            udata[i * nu + j] = udata[i * nu + j] + t * udata[i * nu + k];
          }
        }
        for i in range(k, m) {
          udata[i * nu + k] = - udata[i * nu + k];
        }
        udata[k * nu + k] = One::one::<T>() + udata[k * nu + k];
        for i in range(0, k) {
          udata[(i as uint) * nu + k] = Zero::zero();
        }
        //let mut i = 0;
        //while(i < ((k as int) - 1)) {
        //  i -= 1;
        //}
      } else {
        for i in range(0u, m) {
          udata[i * nu + k] = Zero::zero();
        }
        udata[k * nu + k] = One::one();
      }
    }

    // Generate V.
    for k in range(0u, n).invert() {
      if((k < nrt) && (edata[k] != Zero::zero())) {
        for j in range(k + 1, nu) {
          let mut t : T = Zero::zero();
          for i in range(k + 1, n) {
            t = t + vdata[i * n + k] * vdata[i * n + j];
          }
          t = - t / vdata[(k + 1) * n + k];
          for i in range(k + 1, n) {
            vdata[i * n + j] = vdata[i * n + j] + t * vdata[i * n + k];
          }
        }
      }
      for i in range(0u, n) {
        vdata[i * n + k] = Zero::zero();
      }
      vdata[k * n + k] = One::one();
    }

    // Main iteration loop for the singular values.
    let pp = p - 1;
    let eps : T = num::cast(num::pow(2.0, -52.0));
    let tiny : T = num::cast(num::pow(2.0, -966.0));
    while(p > 0) {
      // Here is where a test for too many iterations would go.

      // This section of the program inspects for
      // negligible elements in the s and e arrays.  On
      // completion the variables kase and k are set as follows.

      // kase = 1     if s(p) and e[k-1] are negligible and k<p
      // kase = 2     if s(k) is negligible and k<p
      // kase = 3     if e[k-1] is negligible, k<p, and
      //              s(k), ..., s(p) are not negligible (qr step).
      // kase = 4     if e(p-1) is negligible (convergence).
      let mut kase;
      let mut k = (p as int) - 2;
      while(k >= 0) {
        if(num::abs(edata[k].clone()) <= (tiny + eps * (num::abs(sdata[k].clone()) + num::abs(sdata[k + 1].clone())))) {
          edata[k] = Zero::zero();
          break;
        }
        k -= 1;
      }

      if(k == ((p as int) - 2)) {
        kase = 4;
      } else {
        let mut ks = (p as int) - 1;
        while(ks > k) {
          let t = (if ks != (p as int) { num::abs(edata[ks].clone()) } else { Zero::zero() })
                  + (if ks != (k + 1) { num::abs(edata[ks - 1].clone()) } else { Zero::zero() });
          if(num::abs(sdata[ks].clone()) <= (tiny + eps * t)) {
            sdata[ks] = Zero::zero();
            break;
          }
          ks -= 1;
        }
        if(ks == k) {
          kase = 3;
        } else if(ks == ((p as int) - 1)) {
          kase = 1;
        } else {
          kase = 2;
          k = ks;
        }
      }
      k += 1;

      // Perform the task indicated by kase.
      if(kase == 1) {
        // Deflate negligible s(p).
        let mut f = edata[p - 2].clone();
        edata[p - 2] = Zero::zero();
        let mut j = (p as int) - 2;
        while(j >= k) {
          let mut t = hypot(sdata[j].clone(), f.clone());
          let cs = sdata[j] / t;
          let sn = f / t;
          sdata[j] = t;
          if(j != k) {
            f = - sn * edata[j - 1];
            edata[j - 1] = cs * edata[j - 1];
          }

          for i in range(0u, n) {
            t = cs * vdata[i * n + (j as uint)] + sn * vdata[i * n + (p - 1)];
            vdata[i * n + (p - 1)] = - sn * vdata[i * n + (j as uint)] + cs * vdata[i * n + (p - 1)];
            vdata[i * n + (j as uint)] = t;
          }
          j -= 1;
        }
      } else if(kase == 2) {
        // Split at negligible s(k).
        let mut f = edata[k - 1].clone();
        edata[k - 1] = Zero::zero();
        for j in range(k, p as int) {
          let mut t = hypot(sdata[j].clone(), f.clone());
          let cs = sdata[j] / t;
          let sn = f / t;
          sdata[j] = t;
          f = - sn * edata[j];
          edata[j] = cs * edata[j];

          for i in range(0u, m) {
            t = cs * udata[i * nu + (j as uint)] + sn * udata[i * nu + ((k as uint) - 1)];
            udata[i * nu + ((k as uint) - 1)] = - sn * udata[i * nu + (j as uint)] + cs * udata[i * nu + ((k as uint) - 1)];
            udata[i * nu + (j as uint)] = t;
          }
        }
      } else if(kase == 3) {
        // Perform one qr step.

        // Calculate the shift.
        let scale = num::max(
                      num::max(
                        num::max(
                          num::max(num::abs(sdata[p - 1].clone()), num::abs(sdata[p - 2].clone())),
                          num::abs(edata[p - 2].clone())),
                        num::abs(sdata[k].clone())),
                      num::abs(edata[k].clone()));
        let sp = sdata[p - 1] / scale;
        let spm1 = sdata[p - 2] / scale;
        let epm1 = edata[p - 2] / scale;
        let sk = sdata[k] / scale;
        let ek = edata[k] / scale;
        let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / num::cast(2.0);
        let c = (sp * epm1) * (sp * epm1);
        let mut shift = Zero::zero();
        if((b != Zero::zero()) || (c != Zero::zero())) {
          shift = num::sqrt(b * b + c);
          if(b < Zero::zero()) {
            shift = - shift;
          }
          shift = c / (b + shift);
        }

        let mut f = (sk + sp) * (sk - sp) + shift;
        let mut g = sk * ek;

        // Chase zeros.
        for j in range(k, (p as int) - 1) {
          let mut t = hypot(f.clone(), g.clone());
          let mut cs = f / t;
          let mut sn = g / t;
          if(j != k) {
            edata[j - 1] = t;
          }
          f = cs * sdata[j] + sn * edata[j];
          edata[j] = cs * edata[j] - sn * sdata[j];
          g = sn * sdata[j + 1];
          sdata[j + 1] = cs * sdata[j + 1];

          for i in range(0u, n) {
            t = cs * vdata[i * n + (j as uint)] + sn * vdata[i * n + ((j as uint) + 1)];
            vdata[i * n + ((j as uint) + 1)] = - sn * vdata[i * n + (j as uint)] + cs * vdata[i * n + ((j as uint) + 1)];
            vdata[i * n + (j as uint)] = t;
          }

          t = hypot(f.clone(), g.clone());
          cs = f / t;
          sn = g / t;
          sdata[j] = t;
          f = cs * edata[j] + sn * sdata[j + 1];
          sdata[j + 1] = - sn * edata[j] + cs * sdata[j + 1];
          g = sn * edata[j + 1];
          edata[j + 1] = cs * edata[j + 1];
          if(j < ((m as int) - 1)) {
            for i in range(0u, m) {
              t = cs * udata[i * nu + (j as uint)] + sn * udata[i * nu + ((j as uint) + 1)];
              udata[i * nu + ((j as uint) + 1)] = - sn * udata[i * nu + (j as uint)] + cs * udata[i * nu + ((j as uint) + 1)];
              udata[i * nu + (j as uint)] = t;
            }
          }
        }

        edata[p - 2] = f;
      } else if(kase == 4) {
        // Convergence.

        // Make the singular values positive.
        if(sdata[k] <= Zero::zero()) {
          sdata[k] = if(sdata[k] < Zero::zero()) { - sdata[k] } else { Zero::zero() };
          for i in range(0u, pp + 1) {
            vdata[i * n + (k as uint)] = - vdata[i * n + (k as uint)];
          }
        }

        // Order the singular values.
        while(k < (pp as int)) {
          if(sdata[k] >= sdata[k + 1]) {
            break;
          }
          let mut t = sdata[k].clone();
          sdata[k] = sdata[k + 1].clone();
          sdata[k + 1] = t;
          if(k < ((n as int) - 1)) {
            for i in range(0u, n) {
              t = vdata[i * n + ((k as uint) + 1)].clone();
              vdata[i * n + ((k as uint) + 1)] = vdata[i * n + (k as uint)].clone();
              vdata[i * n + (k as uint)] = t;
            }
          }
          if(k < ((m as int) - 1)) {
            for i in range(0u, m) {
              t = udata[i * nu + ((k as uint) + 1)].clone();
              udata[i * nu + ((k as uint) + 1)] = udata[i * nu + (k as uint)].clone();
              udata[i * nu + (k as uint)] = t;
            }
          }
          k += 1;
        }

        p -= 1;
      }
    }

    SVD {
      u : matrix(m, nu, udata),
      v : matrix(n, n, vdata),
      s : sdata,
      m : m,
      n : n
    }
  }

  pub fn get_u<'lt>(&'lt self) -> &'lt Matrix<T> {
    &self.u
  }

  pub fn get_v<'lt>(&'lt self) -> &'lt Matrix<T> {
    &self.v
  }

  pub fn get_s(&self) -> Matrix<T> {
    let mut d = vec::from_elem(self.n * self.n, Zero::zero());
    for i in range(0u, self.n) {
      d[i * self.n + i] = self.s[i].clone();
    }
    Matrix { noRows : self.n, noCols: self.n, data : d }
  }

  pub fn rank(&self) -> uint {
    let eps : T = num::cast(num::pow(2.0, -52.0));
    let maxDim : T = num::cast(num::max(self.m, self.n));
    let tol = maxDim * self.s[0] * eps;
    let mut r = 0;
    for i in range(0u, self.s.len()) {
      if(self.s[i] > tol) {
        r += 1;
      }
    }
    r
  }
}

#[test]
fn svd_test() {
  let a = matrix(3, 3, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test__m_over_n() {
  let a = matrix(3, 2, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

