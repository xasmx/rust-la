use std::cmp;
use std::num;

use ApproxEq;
use Matrix;
use internalutil::{alloc_dirty_vec, hypot};

pub struct SVD<T> {
  u : Matrix<T>,
  s : Matrix<T>,
  v : Matrix<T>
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
impl<T : FloatMath + ApproxEq<T>> SVD<T> {
  pub fn new(a : &Matrix<T>) -> SVD<T> {
    // Derived from LINPACK code.
    // Initialize.
    let mut adata = a.get_data().clone();
    let m = a.rows();
    let n = a.cols();

    assert!(m >= n);

    let nu = cmp::min(m, n);

    let slen = cmp::min(m + 1, n);
    let mut sdata : Vec<T> = alloc_dirty_vec(slen);

    let ulen = m * nu;
    let mut udata = alloc_dirty_vec(ulen);

    let vlen = n * n;
    let mut vdata = alloc_dirty_vec(vlen);

    let mut edata = alloc_dirty_vec(n);
    let mut workdata : Vec<T> = alloc_dirty_vec(m);

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    let nct = cmp::min(m - 1, n);
    let nrt = cmp::max(0 as int, cmp::min((n as int) - 2, m as int)) as uint;
    for k in range(0u, cmp::max(nct, nrt)) {
      if k < nct {
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        *sdata.get_mut(k) = num::zero();
        for i in range(k, m) {
          *sdata.get_mut(k) = hypot(sdata.get(k).clone(), adata.get(i * n + k).clone());
        }
        if *sdata.get(k) != num::zero() {
          if *adata.get(k * n + k) < num::zero() {
            *sdata.get_mut(k) = - *sdata.get(k);
          }
          for i in range(k, m) {
            *adata.get_mut(i * n + k) = *adata.get(i * n + k) / *sdata.get(k);
          }
          *adata.get_mut(k * n + k) = *adata.get(k * n + k) + num::one();
        }
        *sdata.get_mut(k) = - *sdata.get(k);
      }
      for j in range(k + 1, n) {
        if (k < nct) && (*sdata.get(k) != num::zero()) {
          // Apply the transformation.
          let mut t : T = num::zero();
          for i in range(k, m) {
            t = t + *adata.get(i * n + k) * *adata.get(i * n + j);
          }
          t = - t / *adata.get(k * n + k);
          for i in range(k, m) {
            *adata.get_mut(i * n + j) = *adata.get(i * n + j) + t * *adata.get(i * n + k);
          }
        }
        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.
        *edata.get_mut(j) = adata.get(k * n + j).clone();
      }

      if k < nct {
        // Place the transformation in U for subsequent back multiplication.
        for i in range(k, m) {
          *udata.get_mut(i * nu + k) = adata.get(i * n + k).clone();
        }
      }

      if k < nrt {
        // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        *edata.get_mut(k) = num::zero();
        for i in range(k + 1, n) {
          *edata.get_mut(k) = hypot(edata.get(k).clone(), edata.get(i).clone());
        }
        if *edata.get(k) != num::zero() {
          if *edata.get(k + 1) < num::zero() {
            *edata.get_mut(k) = - *edata.get(k);
          }
          for i in range(k + 1, n) {
            *edata.get_mut(i) = *edata.get(i) / *edata.get(k);
          }
          *edata.get_mut(k + 1) = *edata.get(k + 1) + num::one();
        }
        *edata.get_mut(k) = - *edata.get(k);
        if (k + 1 < m) && (*edata.get(k) != num::zero()) {
          // Apply the transformation.
          for i in range(k + 1, m) {
            *workdata.get_mut(i) = num::zero();
          }
          for j in range(k + 1, n) {
            for i in range(k + 1, m) {
              *workdata.get_mut(i) = *workdata.get(i) + *edata.get(j) * *adata.get(i * n + j);
            }
          }
          for j in range(k + 1, n) {
            let t = - *edata.get(j) / *edata.get(k + 1);
            for i in range(k + 1, m) {
              *adata.get_mut(i * n + j) = *adata.get(i * n + j) + t * *workdata.get(i);
            }
          }
        }

        // Place the transformation in V for subsequent back multiplication.
        for i in range(k + 1, n) {
          *vdata.get_mut(i * n + k) = edata.get(i).clone();
        }
      }
    }

    // Set up the final bidiagonal matrix or order p.
    let mut p = cmp::min(n, m + 1);
    if nct < n {
      *sdata.get_mut(nct) = adata.get(nct * n + nct).clone();
    }
    if m < p {
      *sdata.get_mut(p - 1) = num::zero();
    }
    if (nrt + 1) < p {
      *edata.get_mut(nrt) = adata.get(nrt * n + (p - 1)).clone();
    }
    *edata.get_mut(p - 1) = num::zero();

    // Generate U.
    for j in range(nct, nu) {
      for i in range(0u, m) {
        *udata.get_mut(i * nu + j) = num::zero();
      }
      *udata.get_mut(j * nu + j) = num::one();
    }
    for k in range(0u, nct).rev() {
      if *sdata.get(k) != num::zero() {
        for j in range(k + 1, nu) {
          let mut t : T = num::zero();
          for i in range(k, m) {
            t = t + *udata.get(i * nu + k) * *udata.get(i * nu + j);
          }
          t = - t / *udata.get(k * nu + k);
          for i in range(k, m) {
            *udata.get_mut(i * nu + j) = *udata.get(i * nu + j) + t * *udata.get(i * nu + k);
          }
        }
        for i in range(k, m) {
          *udata.get_mut(i * nu + k) = - *udata.get(i * nu + k);
        }
        *udata.get_mut(k * nu + k) = num::one::<T>() + *udata.get(k * nu + k);
        for i in range(0, k) {
          *udata.get_mut((i as uint) * nu + k) = num::zero();
        }
        //let mut i = 0;
        //while i < ((k as int) - 1) {
        //  i -= 1;
        //}
      } else {
        for i in range(0u, m) {
          *udata.get_mut(i * nu + k) = num::zero();
        }
        *udata.get_mut(k * nu + k) = num::one();
      }
    }

    // Generate V.
    for k in range(0u, n).rev() {
      if (k < nrt) && (*edata.get(k) != num::zero()) {
        for j in range(k + 1, nu) {
          let mut t : T = num::zero();
          for i in range(k + 1, n) {
            t = t + *vdata.get(i * n + k) * *vdata.get(i * n + j);
          }
          t = - t / *vdata.get((k + 1) * n + k);
          for i in range(k + 1, n) {
            *vdata.get_mut(i * n + j) = *vdata.get(i * n + j) + t * *vdata.get(i * n + k);
          }
        }
      }
      for i in range(0u, n) {
        *vdata.get_mut(i * n + k) = num::zero();
      }
      *vdata.get_mut(k * n + k) = num::one();
    }

    // Main iteration loop for the singular values.
    let pp = p - 1;
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    let tiny : T = num::cast(2.0f64.powf(-966.0)).unwrap();
    while p > 0 {
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
      while k >= 0 {
        if num::abs(edata.get(k as uint).clone()) <= (tiny + eps * (num::abs(sdata.get(k as uint).clone()) + num::abs(sdata.get((k + 1) as uint).clone()))) {
          *edata.get_mut(k as uint) = num::zero();
          break;
        }
        k -= 1;
      }

      if k == ((p as int) - 2) {
        kase = 4;
      } else {
        let mut ks = (p as int) - 1;
        while ks > k {
          let t = (if ks != (p as int) { num::abs(edata.get(ks as uint).clone()) } else { num::zero() })
                  + (if ks != (k + 1) { num::abs(edata.get((ks - 1) as uint).clone()) } else { num::zero() });
          if num::abs(sdata.get(ks as uint).clone()) <= (tiny + eps * t) {
            *sdata.get_mut(ks as uint) = num::zero();
            break;
          }
          ks -= 1;
        }
        if ks == k {
          kase = 3;
        } else if ks == ((p as int) - 1) {
          kase = 1;
        } else {
          kase = 2;
          k = ks;
        }
      }
      k += 1;

      // Perform the task indicated by kase.
      if kase == 1 {
        // Deflate negligible s(p).
        let mut f = edata.get(p - 2).clone();
        *edata.get_mut(p - 2) = num::zero();
        let mut j = (p as int) - 2;
        while j >= k {
          let mut t = hypot(sdata.get(j as uint).clone(), f.clone());
          let cs = *sdata.get(j as uint) / t;
          let sn = f / t;
          *sdata.get_mut(j as uint) = t;
          if j != k {
            f = - sn * *edata.get((j - 1) as uint);
            *edata.get_mut((j - 1) as uint) = cs * *edata.get((j - 1) as uint);
          }

          for i in range(0u, n) {
            t = cs * *vdata.get(i * n + (j as uint)) + sn * *vdata.get(i * n + (p - 1));
            *vdata.get_mut(i * n + (p - 1)) = - sn * *vdata.get(i * n + (j as uint)) + cs * *vdata.get(i * n + (p - 1));
            *vdata.get_mut(i * n + (j as uint)) = t;
          }
          j -= 1;
        }
      } else if kase == 2 {
        // Split at negligible s(k).
        let mut f = edata.get((k - 1) as uint).clone();
        *edata.get_mut((k - 1) as uint) = num::zero();
        for j in range(k, p as int) {
          let mut t = hypot(sdata.get(j as uint).clone(), f.clone());
          let cs = *sdata.get(j as uint) / t;
          let sn = f / t;
          *sdata.get_mut(j as uint) = t;
          f = - sn * *edata.get(j as uint);
          *edata.get_mut(j as uint) = cs * *edata.get(j as uint);

          for i in range(0u, m) {
            t = cs * *udata.get(i * nu + (j as uint)) + sn * *udata.get(i * nu + ((k as uint) - 1));
            *udata.get_mut(i * nu + ((k as uint) - 1)) = - sn * *udata.get(i * nu + (j as uint)) + cs * *udata.get(i * nu + ((k as uint) - 1));
            *udata.get_mut(i * nu + (j as uint)) = t;
          }
        }
      } else if kase == 3 {
        // Perform one qr step.

        // Calculate the shift.
        let scale = num::abs(sdata.get(p - 1).clone())
                      .max(num::abs(sdata.get(p - 2).clone()))
                      .max(num::abs(edata.get(p - 2).clone()))
                      .max(num::abs(sdata.get(k as uint).clone()))
                      .max(num::abs(edata.get(k as uint).clone()));
        let sp = *sdata.get(p - 1) / scale;
        let spm1 = *sdata.get(p - 2) / scale;
        let epm1 = *edata.get(p - 2) / scale;
        let sk = *sdata.get(k as uint) / scale;
        let ek = *edata.get(k as uint) / scale;
        let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / num::cast(2.0).unwrap();
        let c = (sp * epm1) * (sp * epm1);
        let mut shift = num::zero();
        if (b != num::zero()) || (c != num::zero()) {
          shift = (b * b + c).sqrt();
          if b < num::zero() {
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
          if j != k {
            *edata.get_mut((j - 1) as uint) = t;
          }
          f = cs * *sdata.get(j as uint) + sn * *edata.get(j as uint);
          *edata.get_mut(j as uint) = cs * *edata.get(j as uint) - sn * *sdata.get(j as uint);
          g = sn * *sdata.get((j + 1) as uint);
          *sdata.get_mut((j + 1) as uint) = cs * *sdata.get((j + 1) as uint);

          for i in range(0u, n) {
            t = cs * *vdata.get(i * n + (j as uint)) + sn * *vdata.get(i * n + ((j as uint) + 1));
            *vdata.get_mut(i * n + ((j as uint) + 1)) = - sn * *vdata.get(i * n + (j as uint)) + cs * *vdata.get(i * n + ((j as uint) + 1));
            *vdata.get_mut(i * n + (j as uint)) = t;
          }

          t = hypot(f.clone(), g.clone());
          cs = f / t;
          sn = g / t;
          *sdata.get_mut(j as uint) = t;
          f = cs * *edata.get(j as uint) + sn * *sdata.get((j + 1) as uint);
          *sdata.get_mut((j + 1) as uint) = - sn * *edata.get(j as uint) + cs * *sdata.get((j + 1) as uint);
          g = sn * *edata.get((j + 1) as uint);
          *edata.get_mut((j + 1) as uint) = cs * *edata.get((j + 1) as uint);
          if j < ((m as int) - 1) {
            for i in range(0u, m) {
              t = cs * *udata.get(i * nu + (j as uint)) + sn * *udata.get(i * nu + ((j as uint) + 1));
              *udata.get_mut(i * nu + ((j as uint) + 1)) = - sn * *udata.get(i * nu + (j as uint)) + cs * *udata.get(i * nu + ((j as uint) + 1));
              *udata.get_mut(i * nu + (j as uint)) = t;
            }
          }
        }

        *edata.get_mut(p - 2) = f;
      } else if kase == 4 {
        // Convergence.

        // Make the singular values positive.
        if *sdata.get(k as uint) <= num::zero() {
          *sdata.get_mut(k as uint) = if *sdata.get(k as uint) < num::zero() { - *sdata.get(k as uint) } else { num::zero() };
          for i in range(0u, pp + 1) {
            *vdata.get_mut(i * n + (k as uint)) = - *vdata.get(i * n + (k as uint));
          }
        }

        // Order the singular values.
        while k < (pp as int) {
          if *sdata.get(k as uint) >= *sdata.get((k + 1) as uint) {
            break;
          }
          let mut t = sdata.get(k as uint).clone();
          *sdata.get_mut(k as uint) = sdata.get((k + 1) as uint).clone();
          *sdata.get_mut((k + 1) as uint) = t;
          if k < ((n as int) - 1) {
            for i in range(0u, n) {
              t = vdata.get(i * n + ((k as uint) + 1)).clone();
              *vdata.get_mut(i * n + ((k as uint) + 1)) = vdata.get(i * n + (k as uint)).clone();
              *vdata.get_mut(i * n + (k as uint)) = t;
            }
          }
          if k < ((m as int) - 1) {
            for i in range(0u, m) {
              t = udata.get(i * nu + ((k as uint) + 1)).clone();
              *udata.get_mut(i * nu + ((k as uint) + 1)) = udata.get(i * nu + (k as uint)).clone();
              *udata.get_mut(i * nu + (k as uint)) = t;
            }
          }
          k += 1;
        }

        p -= 1;
      }
    }

    SVD {
      u : Matrix::new(m, nu, udata),
      s : Matrix::diag(sdata),
      v : Matrix::new(n, n, vdata)
    }
  }

  pub fn get_u<'lt>(&'lt self) -> &'lt Matrix<T> {
    &self.u
  }

  pub fn get_s<'lt>(&'lt self) -> &'lt Matrix<T> {
    &self.s
  }

  pub fn get_v<'lt>(&'lt self) -> &'lt Matrix<T> {
    &self.v
  }

  pub fn rank(&self) -> uint {
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    let maxDim : T = num::cast(cmp::max(self.u.rows(), self.v.rows())).unwrap();
    let tol = maxDim * self.s.get(0, 0) * eps;
    let mut r = 0;
    for i in range(0u, self.s.rows()) {
      if self.s.get(i, i) > tol {
        r += 1;
      }
    }
    r
  }

  /// Calculates SVD using the direct method. Note that calculating it this way
  /// is not numerically stable, so it is mostly useful for testing purposes.
  pub fn direct(a : &Matrix<T>) -> SVD<T> {
    use EigenDecomposition;

    // A = USV'

    // A'A = VS'U'USV'
    //     = VS'SV'
    let ata = a.t().mul(a);
    let edc = EigenDecomposition::new(&ata);
    //let v = &Matrix::id(2, 2);
    let v = edc.get_v();
    //let temp = Matrix::zero_vector(2);
    //let eigs = temp.get_data(); //edc.get_real_eigenvalues();
    let eigs = edc.get_real_eigenvalues();
    let singular_values : Vec<T> = eigs.iter().map(|&e| e.sqrt()).collect();

    // U*S*V' = A
    // U*S*V'*V = A*V
    // U*S = A*V
    // U*S*Sinv = A*V*Sinv
    // U = A*V*Sinv
    let s_size = singular_values.len();
    let s = Matrix::block_diag(s_size, s_size, singular_values);
    let s_inv = s.inverse().unwrap();
    let u = a.mul(v).mul(&s_inv);

    SVD {
      u : u.clone(),
      s : s.clone(),
      v : v.clone()
    }
  }
}

#[test]
fn svd_test() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * *s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_m_over_n() {
  let a = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * *s * v.t()).approx_eq(&a));
}

#[test]
fn direct_test() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0);
  let svd = SVD::<f64>::direct(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * *s * v.t()).approx_eq(&a));
}

