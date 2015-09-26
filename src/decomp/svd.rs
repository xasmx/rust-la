use std::cmp;
use std::ops::Mul;
use num;
use num::{ Float, Signed };

use ApproxEq;
use Matrix;
use internalutil::{alloc_dirty_vec, hypot};

/// Singular Value Decomposition.
///
/// Ported from JAMA (with changes).
///
/// For an m-by-n matrix A, the singular value decomposition is
/// an m-by-m orthogonal matrix U, an m-by-n block diagonal matrix S, and
/// an n-by-n orthogonal matrix V so that A = U*S*V'.
///
/// The singular values, sigma[k] = S[k][k], are ordered so that
/// sigma[0] >= sigma[1] >= ... >= sigma[n-1].
///
/// The singular value decompostion always exists. The matrix condition number
/// and the effective numerical rank can be computed from this decomposition.
pub struct SVD<T> {
  u : Matrix<T>,
  s : Matrix<T>,
  v : Matrix<T>
}

impl<T : Float + Signed + ApproxEq<T>> SVD<T> {
  /// Calculates SVD.
  pub fn new(a : &Matrix<T>) -> SVD<T> {
    // A = USV'
    if a.rows() < a.cols() {
      // a' = (usv')' = vs'u'
      let svd = SVD::new(&a.t());
      return SVD {
        u : svd.v.clone(),
        s : svd.s.t(),
        v : svd.u.clone()
      }
    }

    // Derived from LINPACK code.
    // Initialize.
    let mut adata = a.get_data().clone();
    let m = a.rows();
    let n = a.cols();

    assert!(m >= n);

    let slen = cmp::min(m, n);
    let mut sdata : Vec<T> = alloc_dirty_vec(slen);

    let ulen = m * m;
    let mut udata = alloc_dirty_vec(ulen);

    let vlen = n * n;
    let mut vdata = alloc_dirty_vec(vlen);

    let mut edata = alloc_dirty_vec(n);
    let mut workdata : Vec<T> = alloc_dirty_vec(m);

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    let nct = cmp::min(m - 1, n);
    let nrt = cmp::max(0 as isize, cmp::min((n as isize) - 2, m as isize)) as usize;
    for k in 0..cmp::max(nct, nrt) {
      if k < nct {
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        sdata[k] = num::zero();
        for i in k..m {
          sdata[k] = hypot(sdata[k].clone(), adata[i * n + k].clone());
        }
        if sdata[k].clone() != num::zero() {
          if adata[k * n + k].clone() < num::zero() {
            sdata[k] = - sdata[k].clone();
          }
          for i in k..m {
            adata[i * n + k] = adata[i * n + k].clone() / sdata[k].clone();
          }
          adata[k * n + k] = adata[k * n + k].clone() + num::one();
        }
        sdata[k] = - sdata[k].clone();
      }
      for j in (k + 1)..n {
        if (k < nct) && (sdata[k].clone() != num::zero()) {
          // Apply the transformation.
          let mut t : T = num::zero();
          for i in k..m {
            t = t + adata[i * n + k].clone() * adata[i * n + j].clone();
          }
          t = - t / adata[k * n + k].clone();
          for i in k..m {
            adata[i * n + j] = adata[i * n + j].clone() + t * adata[i * n + k].clone();
          }
        }
        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.
        edata[j] = adata[k * n + j].clone();
      }

      if k < nct {
        // Place the transformation in U for subsequent back multiplication.
        for i in k..m {
          udata[i * m + k] = adata[i * n + k].clone();
        }
      }

      if k < nrt {
        // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        edata[k] = num::zero();
        for i in (k + 1)..n {
          edata[k] = hypot(edata[k].clone(), edata[i].clone());
        }
        if edata[k].clone() != num::zero() {
          if edata[k + 1].clone() < num::zero() {
            edata[k] = - edata[k].clone();
          }
          for i in (k + 1)..n {
            edata[i] = edata[i].clone() / edata[k].clone();
          }
          edata[k + 1] = edata[k + 1].clone() + num::one();
        }
        edata[k] = - edata[k].clone();
        if (k + 1 < m) && (edata[k].clone() != num::zero()) {
          // Apply the transformation.
          for i in (k + 1)..m {
            workdata[i] = num::zero();
          }
          for j in (k + 1)..n {
            for i in (k + 1)..m {
              workdata[i] = workdata[i].clone() + edata[j].clone() * adata[i * n + j].clone();
            }
          }
          for j in (k + 1)..n {
            let t = - edata[j].clone() / edata[k + 1].clone();
            for i in (k + 1)..m {
              adata[i * n + j] = adata[i * n + j].clone() + t * workdata[i].clone();
            }
          }
        }

        // Place the transformation in V for subsequent back multiplication.
        for i in (k + 1)..n {
          vdata[i * n + k] = edata[i].clone();
        }
      }
    }

    // Set up the final bidiagonal matrix or order p.
    let mut p = cmp::min(n, m + 1);
    if nct < n {
      sdata[nct] = adata[nct * n + nct].clone();
    }
    if m < p {
      sdata[p - 1] = num::zero();
    }
    if (nrt + 1) < p {
      edata[nrt] = adata[nrt * n + (p - 1)].clone();
    }
    edata[p - 1] = num::zero();

    // Generate U.
    for j in nct..m {
      for i in 0..m {
        udata[i * m + j] = num::zero();
      }
      udata[j * m + j] = num::one();
    }
    for k in (0..nct).rev() {
      if sdata[k].clone() != num::zero() {
        for j in (k + 1)..m {
          let mut t : T = num::zero();
          for i in k..m {
            t = t + udata[i * m + k].clone() * udata[i * m + j].clone();
          }
          t = - t / udata[k * m + k].clone();
          for i in k..m {
            udata[i * m + j] = udata[i * m + j].clone() + t * udata[i * m + k].clone();
          }
        }
        for i in k..m {
          udata[i * m + k] = - udata[i * m + k].clone();
        }
        udata[k * m + k] = num::one::<T>() + udata[k * m + k].clone();
        for i in 0..k {
          udata[(i as usize) * m + k] = num::zero();
        }
        //let mut i = 0;
        //while i < ((k as isize) - 1) {
        //  i -= 1;
        //}
      } else {
        for i in 0..m {
          udata[i * m + k] = num::zero();
        }
        udata[k * m + k] = num::one();
      }
    }

    // Generate V.
    for k in (0..n).rev() {
      if (k < nrt) && (edata[k].clone() != num::zero()) {
        for j in (k + 1)..n {
          let mut t : T = num::zero();
          for i in (k + 1)..n {
            t = t + vdata[i * n + k].clone() * vdata[i * n + j].clone();
          }
          t = - t / vdata[(k + 1) * n + k].clone();
          for i in (k + 1)..n {
            vdata[i * n + j] = vdata[i * n + j].clone() + t * vdata[i * n + k].clone();
          }
        }
      }
      for i in 0..n {
        vdata[i * n + k] = num::zero();
      }
      vdata[k * n + k] = num::one();
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
      let kase;
      let mut k = (p as isize) - 2;
      while k >= 0 {
        if num::abs(edata[k as usize].clone()) <= (tiny + eps * (num::abs(sdata[k as usize].clone()) + num::abs(sdata[(k + 1) as usize].clone()))) {
          edata[k as usize] = num::zero();
          break;
        }
        k -= 1;
      }

      if k == ((p as isize) - 2) {
        kase = 4;
      } else {
        let mut ks = (p as isize) - 1;
        while ks > k {
          let t = (if ks != (p as isize) { num::abs(edata[ks as usize].clone()) } else { num::zero() })
                  + (if ks != (k + 1) { num::abs(edata[(ks - 1) as usize].clone()) } else { num::zero() });
          if num::abs(sdata[ks as usize].clone()) <= (tiny + eps * t) {
            sdata[ks as usize] = num::zero();
            break;
          }
          ks -= 1;
        }
        if ks == k {
          kase = 3;
        } else if ks == ((p as isize) - 1) {
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
        let mut f = edata[p - 2].clone();
        edata[p - 2] = num::zero();
        let mut j = (p as isize) - 2;
        while j >= k {
          let mut t = hypot(sdata[j as usize].clone(), f.clone());
          let cs = sdata[j as usize].clone() / t;
          let sn = f / t;
          sdata[j as usize] = t;
          if j != k {
            f = - sn * edata[(j - 1) as usize].clone();
            edata[(j - 1) as usize] = cs * edata[(j - 1) as usize].clone();
          }

          for i in 0..n {
            t = cs * vdata[i * n + (j as usize)].clone() + sn * vdata[i * n + (p - 1)].clone();
            vdata[i * n + (p - 1)] = - sn * vdata[i * n + (j as usize)].clone() + cs * vdata[i * n + (p - 1)].clone();
            vdata[i * n + (j as usize)] = t;
          }
          j -= 1;
        }
      } else if kase == 2 {
        // Split at negligible s(k).
        let mut f = edata[(k - 1) as usize].clone();
        edata[(k - 1) as usize] = num::zero();
        for j in k..(p as isize) {
          let mut t = hypot(sdata[j as usize].clone(), f.clone());
          let cs = sdata[j as usize].clone() / t;
          let sn = f / t;
          sdata[j as usize] = t;
          f = - sn * edata[j as usize].clone();
          edata[j as usize] = cs * edata[j as usize].clone();

          for i in 0..m {
            t = cs * udata[i * m + (j as usize)].clone() + sn * udata[i * m + ((k as usize) - 1)].clone();
            udata[i * m + ((k as usize) - 1)] = - sn * udata[i * m + (j as usize)].clone() + cs * udata[i * m + ((k as usize) - 1)].clone();
            udata[i * m + (j as usize)] = t;
          }
        }
      } else if kase == 3 {
        // Perform one qr step.

        // Calculate the shift.
        let scale = num::abs(sdata[p - 1].clone())
                      .max(num::abs(sdata[p - 2].clone()))
                      .max(num::abs(edata[p - 2].clone()))
                      .max(num::abs(sdata[k as usize].clone()))
                      .max(num::abs(edata[k as usize].clone()));
        let sp = sdata[p - 1].clone() / scale;
        let spm1 = sdata[p - 2].clone() / scale;
        let epm1 = edata[p - 2].clone() / scale;
        let sk = sdata[k as usize].clone() / scale;
        let ek = edata[k as usize].clone() / scale;
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
        for j in k..((p as isize) - 1) {
          let mut t = hypot(f.clone(), g.clone());
          let mut cs = f / t;
          let mut sn = g / t;
          if j != k {
            edata[(j - 1) as usize] = t;
          }
          f = cs * sdata[j as usize].clone() + sn * edata[j as usize].clone();
          edata[j as usize] = cs * edata[j as usize].clone() - sn * sdata[j as usize].clone();
          g = sn * sdata[(j + 1) as usize].clone();
          sdata[(j + 1) as usize] = cs * sdata[(j + 1) as usize].clone();

          for i in 0..n {
            t = cs * vdata[i * n + (j as usize)].clone() + sn * vdata[i * n + ((j as usize) + 1)].clone();
            vdata[i * n + ((j as usize) + 1)] = - sn * vdata[i * n + (j as usize)].clone() + cs * vdata[i * n + ((j as usize) + 1)].clone();
            vdata[i * n + (j as usize)] = t;
          }

          t = hypot(f.clone(), g.clone());
          cs = f / t;
          sn = g / t;
          sdata[j as usize] = t;
          f = cs * edata[j as usize].clone() + sn * sdata[(j + 1) as usize].clone();
          sdata[(j + 1) as usize] = - sn * edata[j as usize].clone() + cs * sdata[(j + 1) as usize].clone();
          g = sn * edata[(j + 1) as usize].clone();
          edata[(j + 1) as usize] = cs * edata[(j + 1) as usize].clone();
          if j < ((m as isize) - 1) {
            for i in 0..m {
              t = cs * udata[i * m + (j as usize)].clone() + sn * udata[i * m + ((j as usize) + 1)].clone();
              udata[i * m + ((j as usize) + 1)] = - sn * udata[i * m + (j as usize)].clone() + cs * udata[i * m + ((j as usize) + 1)].clone();
              udata[i * m + (j as usize)] = t;
            }
          }
        }

        edata[p - 2] = f;
      } else if kase == 4 {
        // Convergence.

        // Make the singular values positive.
        if sdata[k as usize].clone() <= num::zero() {
          sdata[k as usize] = if sdata[k as usize].clone() < num::zero() { - sdata[k as usize].clone() } else { num::zero() };
          for i in 0..(pp + 1) {
            vdata[i * n + (k as usize)] = - vdata[i * n + (k as usize)].clone();
          }
        }

        // Order the singular values.
        while k < (pp as isize) {
          if sdata[k as usize].clone() >= sdata[(k + 1) as usize].clone() {
            break;
          }
          let mut t = sdata[k as usize].clone();
          sdata[k as usize] = sdata[(k + 1) as usize].clone();
          sdata[(k + 1) as usize] = t;
          if k < ((n as isize) - 1) {
            for i in 0..n {
              t = vdata[i * n + ((k as usize) + 1)].clone();
              vdata[i * n + ((k as usize) + 1)] = vdata[i * n + (k as usize)].clone();
              vdata[i * n + (k as usize)] = t;
            }
          }
          if k < ((m as isize) - 1) {
            for i in 0..m {
              t = udata[i * m + ((k as usize) + 1)].clone();
              udata[i * m + ((k as usize) + 1)] = udata[i * m + (k as usize)].clone();
              udata[i * m + (k as usize)] = t;
            }
          }
          k += 1;
        }

        p -= 1;
      }
    }

    SVD {
      u : Matrix::new(m, m, udata),
      s : Matrix::block_diag(m, n, sdata),
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

  pub fn rank(&self) -> usize {
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    let max_dim : T = num::cast(cmp::max(self.u.rows(), self.v.rows())).unwrap();
    let tol = max_dim * self.s.get(0, 0) * eps;
    let mut r = 0;
    for i in 0..self.s.rows() {
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
    if a.rows() < a.cols() {
      // a' = (usv')' = vs'u'
      let svd = SVD::direct(&a.t());
      return SVD {
        u : svd.v.clone(),
        s : svd.s.t(),
        v : svd.u.clone()
      }
    }

    // A'A = VS'U'USV'
    //     = VS'SV'
    let ata = a.t().mul(a);
    let edc = EigenDecomposition::new(&ata);
    let v = edc.get_v();
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
    let (s_aug, s_inv_aug) =
        if a.rows() == a.cols() { (s, s_inv) }
        else {
          (s.cb(&Matrix::zero(a.rows() - a.cols(), s.cols())),
           s_inv.cr(&Matrix::zero(s_inv.rows(), a.rows() - a.cols())))
        };
    let u = a.mul(v).mul(&s_inv_aug);

    SVD {
      u : u.clone(),
      s : s_aug.clone(),
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
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_m_over_n() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 10.0, 11.0, 12.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_n_over_m() {
  let a = m!(1.0, 2.0, 3.0, 4.0; 5.0, 6.0, 7.0, 8.0; 9.0, 10.0, 11.0, 12.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn direct_test() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0);
  let svd = SVD::<f64>::direct(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn direct_test_m_over_n() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 10.0, 11.0, 12.0);
  let svd = SVD::<f64>::direct(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn direct_test_n_over_m() {
  let a = m!(1.0, 2.0, 3.0, 4.0; 5.0, 6.0, 7.0, 8.0; 9.0, 10.0, 11.0, 12.0);
  let svd = SVD::<f64>::direct(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

