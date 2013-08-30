use std::num;
use std::num::{One, Zero, NumCast};

use super::super::matrix::*;
use super::super::util::{alloc_dirty_vec, hypot};

pub struct EigenDecomposition<T> {
  n : uint,
  d : ~[T],
  e : ~[T],
  v : Matrix<T>,
  h : Option<Matrix<T>>
}

// Ported from JAMA.
// Eigenvalues and eigenvectors of a real matrix. 
//
// If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is
// diagonal and the eigenvector matrix V is orthogonal.
// I.e. A = V * D * V' and V * V' = I.
//
// If A is not symmetric, then the eigenvalue matrix D is block diagonal
// with the real eigenvalues in 1-by-1 blocks and any complex eigenvalues,
// lambda + i*mu, in 2-by-2 blocks, [lambda, mu; -mu, lambda].  The
// columns of V represent the eigenvectors in the sense that A*V = V*D,
// The matrix V may be badly conditioned, or even singular, so the validity
// of the equation A = V * D * V^-1 depends upon V.cond().
impl<T : Num + NumCast + Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Algebraic + Signed + Orderable> EigenDecomposition<T> {
  // Symmetric Householder reduction to tridiagonal form.
  fn tred2(n : uint, ddata : &mut [T], vdata : &mut [T], edata : &mut [T]) {
    //  This is derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.    
    for j in range(0u, n) {
      ddata[j] = vdata[(n - 1) * n + j].clone();
    }

    // Householder reduction to tridiagonal form.
    for i in range(1, n).invert() {
      // Scale to avoid under/overflow.
      let mut scale : T = Zero::zero();
      let mut h : T = Zero::zero();
      for k in range(0u, i) {
        scale = scale + num::abs(ddata[k].clone());
      }
      if(scale == Zero::zero()) {
        edata[i] = ddata[i - 1].clone();
        for j in range(0u, i) {
          ddata[j] = vdata[(i - 1) * n + j].clone();
          vdata[i * n + j] = Zero::zero();
          vdata[j * n + i] = Zero::zero();
        }
      } else {
        // Generate Householder vector.
        for k in range(0u, i) {
          ddata[k] = ddata[k] / scale;
          h = h + ddata[k] * ddata[k];
        }
        let mut f = ddata[i - 1].clone();
        let mut g = num::sqrt(h.clone());
        if(f > Zero::zero()) {
          g = - g;
        }
        edata[i] = scale * g;
        h = h - f * g;
        ddata[i - 1] = f - g;
        for j in range(0u, i) {
          edata[j] = Zero::zero();
        }

        // Apply similarity transformation to remaining columns.
        for j in range(0u, i) {
          f = ddata[j].clone();
          vdata[j * n + i] = f.clone();
          g = edata[j] + vdata[j * n + j] * f;
          for k in range(j + 1, i) {
            g = g + vdata[k * n + j] * ddata[k];
            edata[k] = edata[k] + vdata[k * n + j] * f;
          }
          edata[j] = g;
        }
        f = Zero::zero();
        for j in range(0u, i) {
          edata[j] = edata[j] / h;
          f = f + edata[j] * ddata[j];
        }
        let hh = f / (h + h);
        for j in range(0u, i) {
          edata[j] = edata[j] - hh * ddata[j];
        }
        for j in range(0u, i) {
          f = ddata[j].clone();
          g = edata[j].clone();
          for k in range(j, i) {
            let orig_val = vdata[k * n + j].clone();
            vdata[k * n + j] = orig_val - (f * edata[k] + g * ddata[k]);
          }
          ddata[j] = vdata[(i - 1) * n + j].clone();
          vdata[i * n + j] = Zero::zero();
        }
      }
      ddata[i] = h;
    }

    // Accumulate transformations.
    for i in range(0u, n - 1) {
      let orig_val = vdata[i * n + i].clone();
      vdata[(n - 1) * n + i] = orig_val;
      vdata[i * n + i] = One::one();
      let h = ddata[i + 1].clone();
      if(h != Zero::zero()) {
        for k in range(0, i + 1) {
          ddata[k] = vdata[k * n + (i + 1)] / h;
        }
        for j in range(0u, i + 1) {
          let mut g : T = Zero::zero();
          for k in range(0u, i + 1) {
            g = g + vdata[k * n + (i + 1)] * vdata[k * n + j];
          }
          for k in range(0u, i + 1) {
            let orig_val = vdata[k * n + j].clone();
            vdata[k * n + j] = orig_val - g * ddata[k];
          }
        }
      }
      for k in range(0u, i + 1) {
        vdata[k * n + (i + 1)] = Zero::zero();
      }
    }
    for j in range(0u, n) {
      ddata[j] = vdata[(n - 1) * n + j].clone();
      vdata[(n - 1) * n + j] = Zero::zero();
    }
    vdata[(n - 1) * n + (n - 1)] = One::one();
    edata[0] = Zero::zero();
  }

  // Symmetric tridiagonal QL algorithm.
  fn tql2(n : uint, edata : &mut [T], ddata : &mut [T], vdata : &mut [T]) {
    // This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    // Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
    for i in range(1, n) {
      edata[i - 1] = edata[i].clone();
    }
    edata[n - 1] = Zero::zero();

    let mut f : T = Zero::zero();
    let mut tst1 : T = Zero::zero();
    let eps : T = num::cast(num::pow(2.0, -52.0));
    for l in range(0u, n) {
      // Find small subdiagonal element
      tst1 = num::max(tst1, num::abs(ddata[l].clone()) + num::abs(edata[l].clone()));
      let mut m = l;
      while(m < n) {
        if(num::abs(edata[m].clone()) <= (eps * tst1)) {
          break;
        }
        m += 1;
      }

      // If m == l, d[l] is an eigenvalue, otherwise, iterate.
      if(m > l) {
        loop {
          // Compute implicit shift
          let mut g = ddata[l].clone();
          let tmp : T = num::cast(2.0);
          let mut p = (ddata[l + 1] - g) / (tmp * edata[l]);
          let mut r = hypot::<T>(p.clone(), One::one());
          if(p < Zero::zero()) {
            r = -r;
          }
          ddata[l] = edata[l] / (p + r);
          ddata[l + 1] = edata[l] * (p + r);
          let dl1 = ddata[l + 1].clone();
          let mut h = g - ddata[l];
          for i in range(l + 2, n) {
            ddata[i] = ddata[i] - h;
          }
          f = f + h;

          // Implicit QL transformation.
          p = ddata[m].clone();
          let mut c : T = One::one();
          let mut c2 = c.clone();
          let mut c3 = c.clone();
          let el1 = edata[l + 1].clone();
          let mut s : T = Zero::zero();
          let mut s2 = Zero::zero();
          for i in range(l, m).invert() {
            c3 = c2.clone();
            c2 = c.clone();
            s2 = s.clone();
            g = c * edata[i];
            h = c * p;
            r = hypot::<T>(p.clone(), edata[i].clone());
            edata[i + 1] = s * r;
            s = edata[i] / r;
            c = p / r;
            p = c * ddata[i] - s * g;
            ddata[i + 1] = h + s * (c * g + s * ddata[i]);

            // Accumulate transformation.
            for k in range(0u, n) {
              h = vdata[k * n + (i + 1)].clone();
              vdata[k * n + (i + 1)] = s * vdata[k * n + i] + c * h;
              vdata[k * n + i] = c * vdata[k * n + i] - s * h;
            }
          }
          p = - s * s2 * c3 * el1 * edata[l] / dl1;
          edata[l] = s * p;
          ddata[l] = c * p;

          // Check for convergence.
          if(num::abs(edata[l].clone()) > (eps * tst1)) {
            break;
          }
        }
        ddata[l] = ddata[l] + f;
        edata[l] = Zero::zero();
      }

      // Sort eigenvalues and corresponding vectors.
      for i in range(0u, n - 1) {
        let mut k = i;
        let mut p = ddata[i].clone();
        for j in range(i + 1, n) {
          if(ddata[j] < p) {
            k = j;
            p = ddata[j].clone();
          }
        }
        if(k != i) {
          ddata[k] = ddata[i].clone();
          ddata[i] = p.clone();
          for j in range(0u, n) {
            p = vdata[j * n + i].clone();
            vdata[j * n + i] = vdata[j * n + k].clone();
            vdata[j * n + k] = p;
          }
        }
      }
    }
  }

  // Nonsymmetric reduction to Hessenberg form.
  fn orthes(n : uint, hdata : &mut [T], vdata : &mut[T]) {
    // This is derived from the Algol procedures orthes and ortran, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutines in EISPACK.

    let mut ort = alloc_dirty_vec(n);

    let low = 0;
    let high = n - 1;

    for m in range(low + 1, high) {
      // Scale column.
      let mut scale : T = Zero::zero();
      for i in range(m, high + 1) {
        scale = scale + num::abs(hdata[i * n + (m - 1)].clone());
      }
      if(scale != Zero::zero()) {
        // Compute Householder transformation.
        let mut h : T = Zero::zero();
        for i in range(m, high + 1).invert() {
          ort[i] = hdata[i * n + (m - 1)] / scale;
          h = h + ort[i] * ort[i];
        }
        let mut g = num::sqrt(h.clone());
        if(ort[m] > Zero::zero()) {
          g = -g;
        }
        h = h - ort[m] * g;
        ort[m] = ort[m] - g;

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)
        for j in range(m, n) {
          let mut f : T = Zero::zero();
          for i in range(m, high + 1).invert() {
            f = f + ort[i] * hdata[i * n + j];
          }
          f = f / h;
          for i in range(m, high + 1) {
            hdata[i * n + j] = hdata[i * n + j] - f * ort[i];
          }
        }

        for i in range(0u, high + 1) {
          let mut f : T = Zero::zero();
          for j in range(m, high + 1).invert() {
            f = f + ort[j] * hdata[i * n + j];
          }
          f = f / h;
          for j in range(m, high + 1) {
            hdata[i * n + j] = hdata[i * n + j] - f * ort[j];
          }
        }
        ort[m] = scale * ort[m];
        hdata[m * n + (m - 1)] = scale * g;
      }
    }

    // Accumulate transformations (Algol's ortran).
    for i in range(0u, n) {
      for j in range(0u, n) {
        vdata[i * n + j] = if(i == j) { One::one() } else { Zero::zero() };
      }
    }

    for m in range(low + 1, high).invert() {
      if(hdata[m * n + (m - 1)] != Zero::zero()) {
        for i in range(m + 1, high + 1) {
          ort[i] = hdata[i * n + (m - 1)].clone();
        }
        for j in range(m, high + 1) {
          let mut g : T = Zero::zero();
          for i in range(m, high + 1) {
            g = g + ort[i] * vdata[i * n + j];
          }
          // Double division avoids possible underflow
          g = (g / ort[m]) / hdata[m * n + (m - 1)];
          for i in range(m, high + 1) {
            vdata[i * n + j] = vdata[i * n + j] + g * ort[i];
          }
        }
      }
    }
  }

  // Complex scalar division.
  fn cdiv(xr : T, xi : T, yr : T, yi : T) -> (T, T) {
    if(num::abs(yr.clone()) > num::abs(yi.clone())) {
      let r = yi / yr;
      let d = yr + r * yi;
      ((xr + r * xi) / d, (xi - r * xr) / d)
    } else {
      let r = yr / yi;
      let d = yi + r * yr;
      ((r * xr + xi) / d, (r * xi - xr) / d)
    }
  }

  // Nonsymmetric reduction from Hessenberg to real Schur form.
  pub fn hqr2(n : uint, ddata : &mut [T], edata : &mut [T], hdata : &mut [T], vdata : &mut [T]) {
    // This is derived from the Algol procedure hqr2, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.

    // Initialize
    let nn = n as int;
    let mut n = nn - 1;
    let low : int = 0;
    let high = nn - 1;
    let eps : T = num::cast(num::pow(2.0, -52.0));
    let mut exshift = Zero::zero();
    let mut p = Zero::zero();
    let mut q = Zero::zero();
    let mut r = Zero::zero();
    let mut s = Zero::zero();
    let mut z = Zero::zero();
    let mut t;
    let mut w;
    let mut x;
    let mut y;

    // Store roots isolated by balanc and compute matrix norm
    let mut norm : T = Zero::zero();
    for i in range(0, nn) {
      if((i < low) || (i > high)) {
        ddata[i] = hdata[i * nn + i].clone();
        edata[i] = Zero::zero();
      }
      for j in range(num::max(i - 1, 0), nn) {
        norm = norm + num::abs(hdata[i * nn + j].clone());
      }
    }

    // Outer loop over eigenvalue index
    let mut iter = 0;
    while(n >= low) {

      // Look for single small sub-diagonal element
      let mut l = n;
      while(l > low) {
        s = num::abs(hdata[(l - 1) * nn + (l - 1)].clone()) + num::abs(hdata[l * nn + l].clone());
        if(s == Zero::zero()) {
          s = norm.clone();
        }
        if(num::abs(hdata[l * nn + (l - 1)].clone()) < (eps * s)) {
          break;
        }
        l -= 1;
      }

      // Check for convergence.
      if(l == n) {
        //One root found.
        hdata[n * nn + n] = hdata[n * nn + n] + exshift;
        ddata[n] = hdata[n * nn + n].clone();
        edata[n] = Zero::zero();
        n -= 1;
        iter = 0;
      } else if(l == (n - 1)) {
        // Two roots found
        w = hdata[n * nn + (n - 1)] * hdata[(n - 1) * nn + n];
        p = (hdata[(n - 1) * nn + (n - 1)] - hdata[n * nn + n]) / num::cast(2.0);
        q = p * p + w;
        z = num::sqrt(num::abs(q.clone()));
        hdata[n * nn + n] = hdata[n * nn + n] + exshift;
        hdata[(n - 1) * nn + (n - 1)] = hdata[(n - 1) * nn + (n - 1)] + exshift;
        x = hdata[n * nn + n].clone();

        // Real pair
        if(q >= Zero::zero()) {
          z = if(p >= Zero::zero()) { p + z } else { p - z };
          ddata[n - 1] = x + z;
          ddata[n] = ddata[n - 1].clone();
          if(z != Zero::zero()) {
            ddata[n] = x - w / z;
          }
          edata[n - 1] = Zero::zero();
          edata[n] = Zero::zero();
          x = hdata[n * nn + (n - 1)].clone();
          s = num::abs(x.clone()) + num::abs(z.clone());
          p = x / s;
          q = z / s;
          r = num::sqrt(p * p + q * q);
          p = p / r;
          q = q / r;

          // Row modification
          for j in range(n - 1, nn) {
            z = hdata[(n - 1) * nn + j].clone();
            hdata[(n - 1) * nn + j] = q * z + p * hdata[n * nn + j];
            hdata[n * nn + j] = q * hdata[n * nn + j] - p * z;
          }

          // Column modification
          for i in range(0, n + 1) {
            z = hdata[i * nn + (n - 1)].clone();
            hdata[i * nn + (n - 1)] = q * z + p * hdata[i * nn + n];
            hdata[i * nn + n] = q * hdata[i * nn + n] - p * z;
          }

          // Accumulate transformations
          for i in range(low, high + 1) {
            z = vdata[i * nn + (n - 1)].clone();
            vdata[i * nn + (n - 1)] = q * z + p * vdata[i * nn + n];
            vdata[i * nn + n] = q * vdata[i * nn + n] - p * z;
          }
        } else {
          // Complex pair
          ddata[n - 1] = x + p;
          ddata[n] = x + p;
          edata[n - 1] = z.clone();
          edata[n] = - z;
        }
        n = n - 2;
        iter = 0;
      } else {
        // No convergence yet

        // Form shift
        x = hdata[n * nn + n].clone();
        y = Zero::zero();
        w = Zero::zero();
        if(l < n) {
          y = hdata[(n - 1) * nn + (n - 1)].clone();
          w = hdata[n * nn + (n - 1)] * hdata[(n - 1) * nn + n];
        }

        // Wilkinson's original ad hoc shift
        if(iter == 10) {
          exshift = exshift + x;
          for i in range(low, n + 1) {
            hdata[i * nn + i] = hdata[i * nn + i] - x;
          }
          s = num::abs(hdata[n * nn + (n - 1)].clone()) + num::abs(hdata[(n - 1) * nn + (n - 2)].clone());
          let tmp : T = num::cast(0.75);
          y = tmp * s;
          x = y.clone();
          let tmp : T = num::cast(-0.4375);
          w = tmp * s * s;
        }

        // MATLAB's new ad hoc shift
        if(iter == 30) {
          s = (y - x) / num::cast(2.0);
          s = s * s + w;
          if(s > Zero::zero()) {
            s = num::sqrt(s.clone());
            if(y < x) {
              s = - s;
            }
            s = x - w / ((y - x) / num::cast(2.0) + s);
            for i in range(low, n + 1) {
              hdata[i * nn + i] = hdata[i * nn + i] - s;
            }
            exshift = exshift + s;
            w = num::cast(0.964);
            y = w.clone();
            x = y.clone();
          }
        }

        iter += 1;

        // Look for two consecutive small sub-diagonal elements
        let mut m = n - 2;
        while(m >= l) {
          z = hdata[m * nn + m].clone();
          r = x - z;
          s = y - z;
          p = (r * s - w) / hdata[(m + 1) * nn + m] + hdata[m * nn + (m + 1)];
          q = hdata[(m + 1) * nn + (m + 1)] - z - r - s;
          r = hdata[(m + 2) * nn + (m + 1)].clone();
          s = num::abs(p.clone()) + num::abs(q.clone()) + num::abs(r.clone());
          p = p / s;
          q = q / s;
          r = r / s;
          if(m == l) {
            break;
          }
          if((num::abs(hdata[m * nn + (m - 1)].clone()) * (num::abs(q.clone()) + num::abs(r.clone()))) <
             eps * (num::abs(p.clone()) * (num::abs(hdata[(m - 1) * nn + (m - 1)].clone()) + num::abs(z.clone()) + num::abs(hdata[(m + 1) * nn + (m + 1)].clone())))) {
            break;
          }
          m -= 1;
        }

        for i in range(m + 2, n + 1) {
          hdata[i * nn + (i - 2)] = Zero::zero();
          if(i > (m + 2)) {
            hdata[i * nn + (i - 3)] = Zero::zero();
          }
        }

//SAMI: OK this far.
        // Double QR step involving rows l:n and columns m:n
        for k in range(m, n) {
          let notlast = (k != (n - 1));
          if(k != m) {
            p = hdata[k * nn + (k - 1)].clone();
            q = hdata[(k + 1) * nn + (k - 1)].clone();
            r = if notlast { hdata[(k + 2) * nn + (k - 1)].clone() } else { Zero::zero() };
            x = num::abs(p.clone()) + num::abs(q.clone()) + num::abs(r.clone());
            if(x == Zero::zero()) {
              loop;
            }
            p = p / x;
            q = q / x;
            r = r / x;
          }

          s = num::sqrt(p * p + q * q + r * r);
          if(p < Zero::zero()) {
            s = - s;
          }
          if(s != Zero::zero()) {
            if(k != m) {
              hdata[k * nn + (k - 1)] = - s * x;
            } else if(l != m) {
              hdata[k * nn + (k - 1)] = - hdata[k * nn + (k - 1)];
            }
            p = p + s;
            x = p / s;
            y = q / s;
            z = r / s;
            q = q / p;
            r = r / p;

            // Row modification
            for j in range(k, nn) {
              p = hdata[k * nn + j] + q * hdata[(k + 1) * nn + j];
              if notlast {
                p = p + r * hdata[(k + 2) * nn + j];
                hdata[(k + 2) * nn + j] = hdata[(k + 2) * nn + j] - p * z;
              }
              hdata[k * nn + j] = hdata[k * nn + j] - p * x;
              hdata[(k + 1) * nn + j] = hdata[(k + 1) * nn + j] - p * y;
            }

            // Column modification
            for i in range(0, num::min(n, k + 3) + 1) {
              p = x * hdata[i * nn + k] + y * hdata[i * nn + (k + 1)];
              if notlast {
                p = p + z * hdata[i * nn + (k + 2)];
                hdata[i * nn + (k + 2)] = hdata[i * nn + (k + 2)] - p * r;
              }
              hdata[i * nn + k] = hdata[i * nn + k] - p;
              hdata[i * nn + (k + 1)] = hdata[i * nn + (k + 1)] - p * q;
            }

            // Accumulate transformations
            for i in range(low, high + 1) {
              p = x * vdata[i * nn + k] + y * vdata[i * nn + (k + 1)];
              if notlast {
                p = p + z * vdata[i * nn + (k + 2)];
                vdata[i * nn + (k + 2)] = vdata[i * nn + (k + 2)] - p * r;
              }
              vdata[i * nn + k] = vdata[i * nn + k] - p;
              vdata[i * nn + (k + 1)] = vdata[i * nn + (k + 1)] - p * q;
            }
          }
        }
      }
    }
// SAMI: broken here

    // Backsubstitute to find vectors of upper triangular form
    if(norm == Zero::zero()) {
      return;
    }

    for n in range(0, nn).invert() {
      p = ddata[n].clone();
      q = edata[n].clone();

      // Real vector
      if(q == Zero::zero()) {
        let mut l = n;
        hdata[n * nn + n] = One::one();
        for i in range(0, n).invert() {
          w = hdata[i * nn + i] - p;
          r = Zero::zero();
          for j in range(l, n + 1) {
            r = r + hdata[i * nn + j] * hdata[j * nn + n];
          }
          if(edata[i] < Zero::zero()) {
            z = w.clone();
            s = r.clone();
          } else {
            l = i;
            if(edata[i] == Zero::zero()) {
              if(w != Zero::zero()) {
                hdata[i * nn + n] = - r / w;
              } else {
                hdata[i * nn + n] = - r / (eps * norm);
              }
            } else {
              // Solve real equations
              x = hdata[i * nn + (i + 1)].clone();
              y = hdata[(i + 1) * nn + i].clone();
              q = (ddata[i] - p) * (ddata[i] - p) + edata[i] * edata[i];
              t = (x * s - z * r) / q;
              hdata[i * nn + n] = t.clone();
              if(num::abs(x.clone()) > num::abs(z.clone())) {
                hdata[(i + 1) * nn + n] = (-r - w * t) / x;
              } else {
                hdata[(i + 1) * nn + n] = (-s - y * t) / z;
              }
            }

            // Overflow control
            t = num::abs(hdata[i * nn + n].clone());
            if((eps * t) * t > One::one()) {
              for j in range(i, n + 1) {
                hdata[j * nn + n] = hdata[j * nn + n] / t;
              }
            }
          }
        }
      } else if(q < Zero::zero()) {
        // Complex vector
        let mut l = n - 1;

        // Last vector component imaginary so matrix is triangular
        if(num::abs(hdata[n * nn + (n - 1)].clone()) > num::abs(hdata[(n - 1) * nn + n].clone())) {
          hdata[(n - 1) * nn + (n - 1)] = q / hdata[n * nn + (n - 1)];
          hdata[(n - 1) * nn + n] = - (hdata[n * nn + n] - p) / hdata[n * nn + (n - 1)];
        } else {
          let (cdivr, cdivi) = EigenDecomposition::cdiv::<T>(Zero::zero(), - hdata[(n - 1) * nn + n], hdata[(n - 1) * nn + (n - 1)] - p, q.clone());
          hdata[(n - 1) * nn + (n - 1)] = cdivr;
          hdata[(n - 1) * nn + n] = cdivi;
        }
        hdata[n * nn + (n - 1)] = Zero::zero();
        hdata[n * nn + n] = One::one();
        for i in range(0, n - 1).invert() {
          let mut ra : T = Zero::zero();
          let mut sa : T = Zero::zero();
          let mut vr;
          let mut vi;
          for j in range(l, n + 1) {
            ra = ra + hdata[i * nn + j] * hdata[j * nn + (n - 1)];
            sa = sa + hdata[i * nn + j] * hdata[j * nn + n];
          }
          w = hdata[i * nn + i] - p;

          if(edata[i] < Zero::zero()) {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if(edata[i] == Zero::zero()) {
              let (cdivr, cdivi) = EigenDecomposition::cdiv(- ra, - sa, w.clone(), q.clone());
              hdata[i * nn + (n - 1)] = cdivr;
              hdata[i * nn + n] = cdivi;
            } else {
              // Solve complex equations
              x = hdata[i * nn + (i + 1)].clone();
              y = hdata[(i + 1) * nn + i].clone();
              vr = (ddata[i] - p) * (ddata[i] - p) + edata[i] * edata[i] - q * q;
              vi = (ddata[i] - p) * num::cast(2.0) * q;
              if((vr == Zero::zero()) && (vi == Zero::zero())) {
                vr = eps * norm * (num::abs(w.clone()) + num::abs(q.clone()) + num::abs(x.clone()) + num::abs(y.clone()) + num::abs(z.clone()));
              }
              let (cdivr, cdivi) = EigenDecomposition::cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              hdata[i * nn + (n - 1)] = cdivr;
              hdata[i * nn + n] = cdivi;
              if(num::abs(x.clone()) > (num::abs(z.clone()) + num::abs(q.clone()))) {
                hdata[(i + 1) * nn + (n - 1)] = (- ra - w * hdata[i * nn + (n - 1)] + q * hdata[i * nn + n]) / x;
                hdata[(i + 1) * nn + n] = (- sa - w * hdata[i * nn + n] - q * hdata[i * nn + (n - 1)]) / x;
              } else {
                let (cdivr, cdivi) = EigenDecomposition::cdiv(- r - y * hdata[i * nn + (n - 1)], - s - y * hdata[i * nn + n], z.clone(), q.clone());
                hdata[(i + 1) * nn + (n - 1)] = cdivr;
                hdata[(i + 1) * nn + n] = cdivi;
              }
            }

            // Overflow control
            t = num::max(num::abs(hdata[i * nn + (n - 1)].clone()), num::abs(hdata[i * nn + n].clone()));
            if((eps * t) * t > One::one()) {
              for j in range(i, n + 1) {
                hdata[j * nn + (n - 1)] = hdata[j * nn + (n - 1)] / t;
                hdata[j * nn + n] = hdata[j * nn + n] / t;
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots
    for i in range(0, nn) {
      if((i < low) || (i > high)) {
        for j in range(i, nn) {
          vdata[i * nn + j] = hdata[i * nn + j].clone();
        }
      }
    }

    // Back transformation to get eigenvectors of original matrix
    for j in range(low, nn).invert() {
      for i in range(low, high + 1) {
        z = Zero::zero();
        for k in range(low, num::min(j, high) + 1) {
          z = z + vdata[i * nn + k] * hdata[k * nn + j];
        }
        vdata[i * nn + j] = z;
      }
    }
  }

  pub fn new(a : &Matrix<T>) -> EigenDecomposition<T> {
    let n = a.noCols;

    let mut vdata = alloc_dirty_vec(n * n);
    let mut ddata = alloc_dirty_vec(n);
    let mut edata = alloc_dirty_vec(n);

    let mut issymmetric = true;
    let mut j = 0;
    while((j < n) && issymmetric) {
      let mut i = 0;
      while((i < n) && issymmetric) {
        issymmetric = (a.get(i, j) == a.get(j, i));
        i += 1;
      }
      j += 1;
    }

    if issymmetric {
      for i in range(0, n) {
        for j in range(0, n) {
          vdata[i * n + j] = a.get(i, j).clone();
        }
      }

      // Tridiagonalize.
      EigenDecomposition::tred2(n, ddata, vdata, edata);

      // Diagonalize.
      EigenDecomposition::tql2(n, edata, ddata, vdata);

      EigenDecomposition {
        n : n,
        d : ddata,
        e : edata,
        v : Matrix { noRows : n, noCols : n, data : vdata },
        h : None
      }
    } else {
      let mut hdata = alloc_dirty_vec(n * n);

      for j in range(0, n) {
        for i in range(0, n) {
          hdata[i * n + j] = a.get(i, j);
        }
      }

      // Reduce to Hessenberg form.
      EigenDecomposition::orthes(n, hdata, vdata);
   
      // Reduce Hessenberg to real Schur form.
      EigenDecomposition::hqr2(n, ddata, edata, hdata, vdata);

      EigenDecomposition {
        n : n,
        d : ddata,
        e : edata,
        v : Matrix { noRows : n, noCols : n, data : vdata },
        h : Some(Matrix { noRows : n, noCols : n, data : hdata })
      }
    }
  }

  pub fn get_v<'lt>(&'lt self) -> &'lt Matrix<T> { &self.v }

  pub fn get_real_eigenvalues<'lt>(&'lt self) -> &'lt ~[T] { &self.d }

  pub fn get_imag_eigenvalues<'lt>(&'lt self) -> &'lt ~[T] { &self.e }

  pub fn get_d(&self) -> Matrix<T> {
    let mut ddata = alloc_dirty_vec(self.n * self.n);

    for i in range(0u, self.n) {
      for j in range(0u, self.n) {
        ddata[i * self.n + j] = Zero::zero();
      }
      ddata[i * self.n + i] = self.d[i].clone();
      if(self.e[i] > Zero::zero()) {
        ddata[i * self.n + (i + 1)] = self.e[i].clone();
      } else if(self.e[i] < Zero::zero()) {
        ddata[i * self.n + (i - 1)] = self.e[i].clone();
      }
    }

    Matrix { noRows: self.n, noCols: self.n, data: ddata }
  }
}

#[test]
fn eigen_test() {
  let a = matrix(3, 3, ~[3.0, 1.0, 6.0, 2.0, 1.0, 0.0, -1.0, 0.0, -3.0]);
  let _eig = EigenDecomposition::new(&a);
  let r = _eig.get_real_eigenvalues();
  assert!(vector(r.clone()).approx_eq(&vector(~[3.0, -1.0, -1.0])));
}
