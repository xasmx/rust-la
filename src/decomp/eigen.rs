use std::cmp;
use std::num;
use std::num::FloatMath;

use ApproxEq;
use Matrix;
use internalutil::{alloc_dirty_vec, hypot};

pub struct EigenDecomposition<T> {
  n : uint,
  d : Vec<T>,
  e : Vec<T>,
  v : Matrix<T>
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
impl<T : FloatMath + ApproxEq<T>> EigenDecomposition<T> {
  // Symmetric Householder reduction to tridiagonal form.
  fn tred2(n : uint, ddata : &mut Vec<T>, vdata : &mut Vec<T>, edata : &mut Vec<T>) {
    //  This is derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.    
    for j in range(0u, n) {
      *ddata.get_mut(j as uint) = vdata.get(((n - 1) * n + j) as uint).clone();
    }

    // Householder reduction to tridiagonal form.
    for i in range(1, n).rev() {
      // Scale to avoid under/overflow.
      let mut scale : T = num::zero();
      let mut h : T = num::zero();
      for k in range(0u, i) {
        scale = scale + num::abs(ddata.get(k as uint).clone());
      }
      if scale == num::zero() {
        *edata.get_mut(i as uint) = ddata.get((i - 1) as uint).clone();
        for j in range(0u, i) {
          *ddata.get_mut(j as uint) = vdata.get(((i - 1) * n + j) as uint).clone();
          *vdata.get_mut((i * n + j) as uint) = num::zero();
          *vdata.get_mut((j * n + i) as uint) = num::zero();
        }
      } else {
        // Generate Householder vector.
        for k in range(0u, i) {
          *ddata.get_mut(k as uint) = *ddata.get(k as uint) / scale;
          h = h + *ddata.get(k as uint) * *ddata.get(k as uint);
        }
        let mut f = ddata.get((i - 1) as uint).clone();
        let mut g = h.sqrt();
        if f > num::zero() {
          g = - g;
        }
        *edata.get_mut(i as uint) = scale * g;
        h = h - f * g;
        *ddata.get_mut((i - 1) as uint) = f - g;
        for j in range(0u, i) {
          *edata.get_mut(j as uint) = num::zero();
        }

        // Apply similarity transformation to remaining columns.
        for j in range(0u, i) {
          f = ddata.get(j as uint).clone();
          *vdata.get_mut((j * n + i) as uint) = f.clone();
          g = *edata.get(j as uint) + *vdata.get((j * n + j) as uint) * f;
          for k in range(j + 1, i) {
            g = g + *vdata.get((k * n + j) as uint) * *ddata.get(k as uint);
            *edata.get_mut(k as uint) = *edata.get(k as uint) + *vdata.get((k * n + j) as uint) * f;
          }
          *edata.get_mut(j as uint) = g;
        }
        f = num::zero();
        for j in range(0u, i) {
          *edata.get_mut(j as uint) = *edata.get(j as uint) / h;
          f = f + *edata.get(j as uint) * *ddata.get(j as uint);
        }
        let hh = f / (h + h);
        for j in range(0u, i) {
          *edata.get_mut(j as uint) = *edata.get(j as uint) - hh * *ddata.get(j as uint);
        }
        for j in range(0u, i) {
          f = ddata.get(j as uint).clone();
          g = edata.get(j as uint).clone();
          for k in range(j, i) {
            let orig_val = vdata.get((k * n + j) as uint).clone();
            *vdata.get_mut((k * n + j) as uint) = orig_val - (f * *edata.get(k as uint) + g * *ddata.get(k as uint));
          }
          *ddata.get_mut(j as uint) = vdata.get(((i - 1) * n + j) as uint).clone();
          *vdata.get_mut((i * n + j) as uint) = num::zero();
        }
      }
      *ddata.get_mut(i as uint) = h;
    }

    // Accumulate transformations.
    for i in range(0u, n - 1) {
      let orig_val = vdata.get((i * n + i) as uint).clone();
      *vdata.get_mut(((n - 1) * n + i) as uint) = orig_val;
      *vdata.get_mut((i * n + i) as uint) = num::one();
      let h = ddata.get((i + 1) as uint).clone();
      if h != num::zero() {
        for k in range(0, i + 1) {
          *ddata.get_mut(k as uint) = *vdata.get((k * n + (i + 1)) as uint) / h;
        }
        for j in range(0u, i + 1) {
          let mut g : T = num::zero();
          for k in range(0u, i + 1) {
            g = g + *vdata.get((k * n + (i + 1)) as uint) * *vdata.get((k * n + j) as uint);
          }
          for k in range(0u, i + 1) {
            let orig_val = vdata.get((k * n + j) as uint).clone();
            *vdata.get_mut((k * n + j) as uint) = orig_val - g * *ddata.get(k as uint);
          }
        }
      }
      for k in range(0u, i + 1) {
        *vdata.get_mut((k * n + (i + 1)) as uint) = num::zero();
      }
    }
    for j in range(0u, n) {
      *ddata.get_mut(j as uint) = vdata.get(((n - 1) * n + j) as uint).clone();
      *vdata.get_mut(((n - 1) * n + j) as uint) = num::zero();
    }
    *vdata.get_mut(((n - 1) * n + (n - 1)) as uint) = num::one();
    *edata.get_mut(0) = num::zero();
  }

  // Symmetric tridiagonal QL algorithm.
  fn tql2(n : uint, edata : &mut Vec<T>, ddata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    // Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
    for i in range(1, n) {
      *edata.get_mut((i - 1) as uint) = edata.get(i as uint).clone();
    }
    *edata.get_mut((n - 1) as uint) = num::zero();

    let mut f : T = num::zero();
    let mut tst1 : T = num::zero();
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    for l in range(0u, n) {
      // Find small subdiagonal element
      tst1 = tst1.max(num::abs(ddata.get(l as uint).clone()) + num::abs(edata.get(l as uint).clone()));
      let mut m = l;
      while m < n {
        if num::abs(edata.get(m as uint).clone()) <= (eps * tst1) {
          break;
        }
        m += 1;
      }

      // If m == l, d[l] is an eigenvalue, otherwise, iterate.
      if m > l {
        loop {
          // Compute implicit shift
          let mut g = ddata.get(l as uint).clone();
          let tmp : T = num::cast(2.0).unwrap();
          let mut p = (*ddata.get((l + 1) as uint) - g) / (tmp * *edata.get(l as uint));
          let mut r = hypot::<T>(p.clone(), num::one());
          if p < num::zero() {
            r = -r;
          }
          *ddata.get_mut(l as uint) = *edata.get(l as uint) / (p + r);
          *ddata.get_mut((l + 1) as uint) = *edata.get(l as uint) * (p + r);
          let dl1 = ddata.get((l + 1) as uint).clone();
          let mut h = g - *ddata.get(l as uint);
          for i in range(l + 2, n) {
            *ddata.get_mut(i as uint) = *ddata.get(i as uint) - h;
          }
          f = f + h;

          // Implicit QL transformation.
          p = ddata.get(m as uint).clone();
          let mut c : T = num::one();
          let mut c2 = c.clone();
          let mut c3 = c.clone();
          let el1 = edata.get((l + 1) as uint).clone();
          let mut s : T = num::zero();
          let mut s2 = num::zero();
          for i in range(l, m).rev() {
            c3 = c2.clone();
            c2 = c.clone();
            s2 = s.clone();
            g = c * *edata.get(i as uint);
            h = c * p;
            r = hypot::<T>(p.clone(), edata.get(i as uint).clone());
            *edata.get_mut((i + 1) as uint) = s * r;
            s = *edata.get(i as uint) / r;
            c = p / r;
            p = c * *ddata.get(i as uint) - s * g;
            *ddata.get_mut((i + 1) as uint) = h + s * (c * g + s * *ddata.get(i as uint));

            // Accumulate transformation.
            for k in range(0u, n) {
              h = vdata.get((k * n + (i + 1)) as uint).clone();
              *vdata.get_mut((k * n + (i + 1)) as uint) = s * *vdata.get((k * n + i) as uint) + c * h;
              *vdata.get_mut((k * n + i) as uint) = c * *vdata.get((k * n + i) as uint) - s * h;
            }
          }
          p = - s * s2 * c3 * el1 * *edata.get(l as uint) / dl1;
          *edata.get_mut(l as uint) = s * p;
          *ddata.get_mut(l as uint) = c * p;

          // Check for convergence.
          if num::abs(edata.get(l as uint).clone()) > (eps * tst1) {
            break;
          }
        }
        *ddata.get_mut(l as uint) = *ddata.get(l as uint) + f;
        *edata.get_mut(l as uint) = num::zero();
      }

      // Sort eigenvalues and corresponding vectors.
      for i in range(0u, n - 1) {
        let mut k = i;
        let mut p = ddata.get(i as uint).clone();
        for j in range(i + 1, n) {
          if *ddata.get(j as uint) < p {
            k = j;
            p = ddata.get(j as uint).clone();
          }
        }
        if k != i {
          *ddata.get_mut(k as uint) = ddata.get(i as uint).clone();
          *ddata.get_mut(i as uint) = p.clone();
          for j in range(0u, n) {
            p = vdata.get((j * n + i) as uint).clone();
            *vdata.get_mut((j * n + i) as uint) = vdata.get((j * n + k) as uint).clone();
            *vdata.get_mut((j * n + k) as uint) = p;
          }
        }
      }
    }
  }

  // Nonsymmetric reduction to Hessenberg form.
  fn orthes(n : uint, hdata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedures orthes and ortran, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutines in EISPACK.

    let mut ort = alloc_dirty_vec(n);

    let low = 0;
    let high = n - 1;

    for m in range(low + 1, high) {
      // Scale column.
      let mut scale : T = num::zero();
      for i in range(m, high + 1) {
        scale = scale + num::abs(hdata.get((i * n + (m - 1)) as uint).clone());
      }
      if scale != num::zero() {
        // Compute Householder transformation.
        let mut h : T = num::zero();
        for i in range(m, high + 1).rev() {
          *ort.get_mut(i) = *hdata.get((i * n + (m - 1)) as uint) / scale;
          h = h + *ort.get(i) * *ort.get(i);
        }
        let mut g = h.sqrt();
        if *ort.get(m) > num::zero() {
          g = -g;
        }
        h = h - *ort.get(m) * g;
        *ort.get_mut(m) = *ort.get(m) - g;

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)
        for j in range(m, n) {
          let mut f : T = num::zero();
          for i in range(m, high + 1).rev() {
            f = f + *ort.get(i) * *hdata.get((i * n + j) as uint);
          }
          f = f / h;
          for i in range(m, high + 1) {
            *hdata.get_mut((i * n + j) as uint) = *hdata.get((i * n + j) as uint) - f * *ort.get(i);
          }
        }

        for i in range(0u, high + 1) {
          let mut f : T = num::zero();
          for j in range(m, high + 1).rev() {
            f = f + *ort.get(j) * *hdata.get((i * n + j) as uint);
          }
          f = f / h;
          for j in range(m, high + 1) {
            *hdata.get_mut((i * n + j) as uint) = *hdata.get((i * n + j) as uint) - f * *ort.get(j);
          }
        }
        *ort.get_mut(m) = scale * *ort.get(m);
        *hdata.get_mut((m * n + (m - 1)) as uint) = scale * g;
      }
    }

    // Accumulate transformations (Algol's ortran).
    for i in range(0u, n) {
      for j in range(0u, n) {
        *vdata.get_mut((i * n + j) as uint) = if i == j { num::one() } else { num::zero() };
      }
    }

    for m in range(low + 1, high).rev() {
      if *hdata.get((m * n + (m - 1)) as uint) != num::zero() {
        for i in range(m + 1, high + 1) {
          *ort.get_mut(i) = hdata.get((i * n + (m - 1)) as uint).clone();
        }
        for j in range(m, high + 1) {
          let mut g : T = num::zero();
          for i in range(m, high + 1) {
            g = g + *ort.get(i) * *vdata.get((i * n + j) as uint);
          }
          // Double division avoids possible underflow
          g = (g / *ort.get(m)) / *hdata.get((m * n + (m - 1)) as uint);
          for i in range(m, high + 1) {
            *vdata.get_mut((i * n + j) as uint) = *vdata.get((i * n + j) as uint) + g * *ort.get(i);
          }
        }
      }
    }
  }

  // Complex scalar division.
  fn cdiv(xr : T, xi : T, yr : T, yi : T) -> (T, T) {
    if num::abs(yr.clone()) > num::abs(yi.clone()) {
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
  fn hqr2(n : uint, ddata : &mut Vec<T>, edata : &mut Vec<T>, hdata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedure hqr2, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.

    // Initialize
    let nn = n as int;
    let mut n = nn - 1;
    let low : int = 0;
    let high = nn - 1;
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    let mut exshift = num::zero();
    let mut p = num::zero();
    let mut q = num::zero();
    let mut r = num::zero();
    let mut s = num::zero();
    let mut z = num::zero();
    let mut t;
    let mut w;
    let mut x;
    let mut y;

    // Store roots isolated by balanc and compute matrix norm
    let mut norm : T = num::zero();
    for i in range(0, nn) {
      if (i < low) || (i > high) {
        *ddata.get_mut(i as uint) = hdata.get((i * nn + i) as uint).clone();
        *edata.get_mut(i as uint) = num::zero();
      }
      for j in range(cmp::max(i - 1, 0), nn) {
        norm = norm + num::abs(hdata.get((i * nn + j) as uint).clone());
      }
    }

    // Outer loop over eigenvalue index
    let mut iter = 0;
    while n >= low {

      // Look for single small sub-diagonal element
      let mut l = n;
      while l > low {
        s = num::abs(hdata.get(((l - 1) * nn + (l - 1)) as uint).clone()) + num::abs(hdata.get((l * nn + l) as uint).clone());
        if s == num::zero() {
          s = norm.clone();
        }
        if num::abs(hdata.get((l * nn + (l - 1)) as uint).clone()) < (eps * s) {
          break;
        }
        l -= 1;
      }

      // Check for convergence.
      if l == n {
        //One root found.
        *hdata.get_mut((n * nn + n) as uint) = *hdata.get((n * nn + n) as uint) + exshift;
        *ddata.get_mut(n as uint) = hdata.get((n * nn + n) as uint).clone();
        *edata.get_mut(n as uint) = num::zero();
        n -= 1;
        iter = 0;
      } else if l == (n - 1) {
        // Two roots found
        w = *hdata.get((n * nn + (n - 1)) as uint) * *hdata.get(((n - 1) * nn + n) as uint);
        p = (*hdata.get(((n - 1) * nn + (n - 1)) as uint) - *hdata.get((n * nn + n) as uint)) / num::cast(2.0).unwrap();
        q = p * p + w;
        z = num::abs(q.clone()).sqrt();
        *hdata.get_mut((n * nn + n) as uint) = *hdata.get((n * nn + n) as uint) + exshift;
        *hdata.get_mut(((n - 1) * nn + (n - 1)) as uint) = *hdata.get(((n - 1) * nn + (n - 1)) as uint) + exshift;
        x = hdata.get((n * nn + n) as uint).clone();

        // Real pair
        if q >= num::zero() {
          z = if p >= num::zero() { p + z } else { p - z };
          *ddata.get_mut((n - 1) as uint) = x + z;
          *ddata.get_mut(n as uint) = ddata.get((n - 1) as uint).clone();
          if z != num::zero() {
            *ddata.get_mut(n as uint) = x - w / z;
          }
          *edata.get_mut((n - 1) as uint) = num::zero();
          *edata.get_mut(n as uint) = num::zero();
          x = hdata.get((n * nn + (n - 1)) as uint).clone();
          s = num::abs(x.clone()) + num::abs(z.clone());
          p = x / s;
          q = z / s;
          r = (p * p + q * q).sqrt();
          p = p / r;
          q = q / r;

          // Row modification
          for j in range(n - 1, nn) {
            z = hdata.get(((n - 1) * nn + j) as uint).clone();
            *hdata.get_mut(((n - 1) * nn + j) as uint) = q * z + p * *hdata.get((n * nn + j) as uint);
            *hdata.get_mut((n * nn + j) as uint) = q * *hdata.get((n * nn + j) as uint) - p * z;
          }

          // Column modification
          for i in range(0, n + 1) {
            z = hdata.get((i * nn + (n - 1)) as uint).clone();
            *hdata.get_mut((i * nn + (n - 1)) as uint) = q * z + p * *hdata.get((i * nn + n) as uint);
            *hdata.get_mut((i * nn + n) as uint) = q * *hdata.get((i * nn + n) as uint) - p * z;
          }

          // Accumulate transformations
          for i in range(low, high + 1) {
            z = vdata.get((i * nn + (n - 1)) as uint).clone();
            *vdata.get_mut((i * nn + (n - 1)) as uint) = q * z + p * *vdata.get((i * nn + n) as uint);
            *vdata.get_mut((i * nn + n) as uint) = q * *vdata.get((i * nn + n) as uint) - p * z;
          }
        } else {
          // Complex pair
          *ddata.get_mut((n - 1) as uint) = x + p;
          *ddata.get_mut(n as uint) = x + p;
          *edata.get_mut((n - 1) as uint) = z.clone();
          *edata.get_mut(n as uint) = - z;
        }
        n = n - 2;
        iter = 0;
      } else {
        // No convergence yet

        // Form shift
        x = hdata.get((n * nn + n) as uint).clone();
        y = num::zero();
        w = num::zero();
        if l < n {
          y = hdata.get(((n - 1) * nn + (n - 1)) as uint).clone();
          w = *hdata.get((n * nn + (n - 1)) as uint) * *hdata.get(((n - 1) * nn + n) as uint);
        }

        // Wilkinson's original ad hoc shift
        if iter == 10 {
          exshift = exshift + x;
          for i in range(low, n + 1) {
            *hdata.get_mut((i * nn + i) as uint) = *hdata.get((i * nn + i) as uint) - x;
          }
          s = num::abs(hdata.get((n * nn + (n - 1)) as uint).clone()) + num::abs(hdata.get(((n - 1) * nn + (n - 2)) as uint).clone());
          let tmp : T = num::cast(0.75).unwrap();
          y = tmp * s;
          x = y.clone();
          let tmp : T = num::cast(-0.4375).unwrap();
          w = tmp * s * s;
        }

        // MATLAB's new ad hoc shift
        if iter == 30 {
          s = (y - x) / num::cast(2.0).unwrap();
          s = s * s + w;
          if s > num::zero() {
            s = s.sqrt();
            if y < x {
              s = - s;
            }
            s = x - w / ((y - x) / num::cast(2.0).unwrap() + s);
            for i in range(low, n + 1) {
              *hdata.get_mut((i * nn + i) as uint) = *hdata.get((i * nn + i) as uint) - s;
            }
            exshift = exshift + s;
            w = num::cast(0.964).unwrap();
            y = w.clone();
            x = y.clone();
          }
        }

        iter += 1;

        // Look for two consecutive small sub-diagonal elements
        let mut m = n - 2;
        while m >= l {
          z = hdata.get((m * nn + m) as uint).clone();
          r = x - z;
          s = y - z;
          p = (r * s - w) / *hdata.get(((m + 1) * nn + m) as uint) + *hdata.get((m * nn + (m + 1)) as uint);
          q = *hdata.get(((m + 1) * nn + (m + 1)) as uint) - z - r - s;
          r = hdata.get(((m + 2) * nn + (m + 1)) as uint).clone();
          s = num::abs(p.clone()) + num::abs(q.clone()) + num::abs(r.clone());
          p = p / s;
          q = q / s;
          r = r / s;
          if m == l {
            break;
          }
          if (num::abs(hdata.get((m * nn + (m - 1)) as uint).clone()) * (num::abs(q.clone()) + num::abs(r.clone()))) <
             eps * (num::abs(p.clone()) * (num::abs(hdata.get(((m - 1) * nn + (m - 1)) as uint).clone()) + num::abs(z.clone()) + num::abs(hdata.get(((m + 1) * nn + (m + 1)) as uint).clone()))) {
            break;
          }
          m -= 1;
        }

        for i in range(m + 2, n + 1) {
          *hdata.get_mut((i * nn + (i - 2)) as uint) = num::zero();
          if i > (m + 2) {
            *hdata.get_mut((i * nn + (i - 3)) as uint) = num::zero();
          }
        }

        // Double QR step involving rows l:n and columns m:n
        for k in range(m, n) {
          let notlast = k != (n - 1);
          if k != m {
            p = hdata.get((k * nn + (k - 1)) as uint).clone();
            q = hdata.get(((k + 1) * nn + (k - 1)) as uint).clone();
            r = if notlast { hdata.get(((k + 2) * nn + (k - 1)) as uint).clone() } else { num::zero() };
            x = num::abs(p.clone()) + num::abs(q.clone()) + num::abs(r.clone());
            if x == num::zero() {
              continue;
            }
            p = p / x;
            q = q / x;
            r = r / x;
          }

          s = (p * p + q * q + r * r).sqrt();
          if p < num::zero() {
            s = - s;
          }
          if s != num::zero() {
            if k != m {
              *hdata.get_mut((k * nn + (k - 1)) as uint) = - s * x;
            } else if l != m {
              *hdata.get_mut((k * nn + (k - 1)) as uint) = - *hdata.get((k * nn + (k - 1)) as uint);
            }
            p = p + s;
            x = p / s;
            y = q / s;
            z = r / s;
            q = q / p;
            r = r / p;

            // Row modification
            for j in range(k, nn) {
              p = *hdata.get((k * nn + j) as uint) + q * *hdata.get(((k + 1) * nn + j) as uint);
              if notlast {
                p = p + r * *hdata.get(((k + 2) * nn + j) as uint);
                *hdata.get_mut(((k + 2) * nn + j) as uint) = *hdata.get(((k + 2) * nn + j) as uint) - p * z;
              }
              *hdata.get_mut((k * nn + j) as uint) = *hdata.get((k * nn + j) as uint) - p * x;
              *hdata.get_mut(((k + 1) * nn + j) as uint) = *hdata.get(((k + 1) * nn + j) as uint) - p * y;
            }

            // Column modification
            for i in range(0, cmp::min(n, k + 3) + 1) {
              p = x * *hdata.get((i * nn + k) as uint) + y * *hdata.get((i * nn + (k + 1)) as uint);
              if notlast {
                p = p + z * *hdata.get((i * nn + (k + 2)) as uint);
                *hdata.get_mut((i * nn + (k + 2)) as uint) = *hdata.get((i * nn + (k + 2)) as uint) - p * r;
              }
              *hdata.get_mut((i * nn + k) as uint) = *hdata.get((i * nn + k) as uint) - p;
              *hdata.get_mut((i * nn + (k + 1)) as uint) = *hdata.get((i * nn + (k + 1)) as uint) - p * q;
            }

            // Accumulate transformations
            for i in range(low, high + 1) {
              p = x * *vdata.get((i * nn + k) as uint) + y * *vdata.get((i * nn + (k + 1)) as uint);
              if notlast {
                p = p + z * *vdata.get((i * nn + (k + 2)) as uint);
                *vdata.get_mut((i * nn + (k + 2)) as uint) = *vdata.get((i * nn + (k + 2)) as uint) - p * r;
              }
              *vdata.get_mut((i * nn + k) as uint) = *vdata.get((i * nn + k) as uint) - p;
              *vdata.get_mut((i * nn + (k + 1)) as uint) = *vdata.get((i * nn + (k + 1)) as uint) - p * q;
            }
          }
        }
      }
    }

    // Backsubstitute to find vectors of upper triangular form
    if norm == num::zero() {
      return;
    }

    for n in range(0, nn).rev() {
      p = ddata.get(n as uint).clone();
      q = edata.get(n as uint).clone();

      // Real vector
      if q == num::zero() {
        let mut l = n;
        *hdata.get_mut((n * nn + n) as uint) = num::one();
        for i in range(0, n).rev() {
          w = *hdata.get((i * nn + i) as uint) - p;
          r = num::zero();
          for j in range(l, n + 1) {
            r = r + *hdata.get((i * nn + j) as uint) * *hdata.get((j * nn + n) as uint);
          }
          if *edata.get(i as uint) < num::zero() {
            z = w.clone();
            s = r.clone();
          } else {
            l = i;
            if *edata.get(i as uint) == num::zero() {
              if w != num::zero() {
                *hdata.get_mut((i * nn + n) as uint) = - r / w;
              } else {
                *hdata.get_mut((i * nn + n) as uint) = - r / (eps * norm);
              }
            } else {
              // Solve real equations
              x = hdata.get((i * nn + (i + 1)) as uint).clone();
              y = hdata.get(((i + 1) * nn + i) as uint).clone();
              q = (*ddata.get(i as uint) - p) * (*ddata.get(i as uint) - p) + *edata.get(i as uint) * *edata.get(i as uint);
              t = (x * s - z * r) / q;
              *hdata.get_mut((i * nn + n) as uint) = t.clone();
              if num::abs(x.clone()) > num::abs(z.clone()) {
                *hdata.get_mut(((i + 1) * nn + n) as uint) = (-r - w * t) / x;
              } else {
                *hdata.get_mut(((i + 1) * nn + n) as uint) = (-s - y * t) / z;
              }
            }

            // Overflow control
            t = num::abs(hdata.get((i * nn + n) as uint).clone());
            if (eps * t) * t > num::one() {
              for j in range(i, n + 1) {
                *hdata.get_mut((j * nn + n) as uint) = *hdata.get((j * nn + n) as uint) / t;
              }
            }
          }
        }
      } else if q < num::zero() {
        // Complex vector
        let mut l = n - 1;

        // Last vector component imaginary so matrix is triangular
        if num::abs(hdata.get((n * nn + (n - 1)) as uint).clone()) > num::abs(hdata.get(((n - 1) * nn + n) as uint).clone()) {
          *hdata.get_mut(((n - 1) * nn + (n - 1)) as uint) = q / *hdata.get((n * nn + (n - 1)) as uint);
          *hdata.get_mut(((n - 1) * nn + n) as uint) = - (*hdata.get((n * nn + n) as uint) - p) / *hdata.get((n * nn + (n - 1)) as uint);
        } else {
          let (cdivr, cdivi) = EigenDecomposition::<T>::cdiv(num::zero(), - *hdata.get(((n - 1) * nn + n) as uint), *hdata.get(((n - 1) * nn + (n - 1)) as uint) - p, q.clone());
          *hdata.get_mut(((n - 1) * nn + (n - 1)) as uint) = cdivr;
          *hdata.get_mut(((n - 1) * nn + n) as uint) = cdivi;
        }
        *hdata.get_mut((n * nn + (n - 1)) as uint) = num::zero();
        *hdata.get_mut((n * nn + n) as uint) = num::one();
        for i in range(0, n - 1).rev() {
          let mut ra : T = num::zero();
          let mut sa : T = num::zero();
          let mut vr;
          let mut vi;
          for j in range(l, n + 1) {
            ra = ra + *hdata.get((i * nn + j) as uint) * *hdata.get((j * nn + (n - 1)) as uint);
            sa = sa + *hdata.get((i * nn + j) as uint) * *hdata.get((j * nn + n) as uint);
          }
          w = *hdata.get((i * nn + i) as uint) - p;

          if *edata.get(i as uint) < num::zero() {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if *edata.get(i as uint) == num::zero() {
              let (cdivr, cdivi) = EigenDecomposition::cdiv(- ra, - sa, w.clone(), q.clone());
              *hdata.get_mut((i * nn + (n - 1)) as uint) = cdivr;
              *hdata.get_mut((i * nn + n) as uint) = cdivi;
            } else {
              // Solve complex equations
              x = hdata.get((i * nn + (i + 1)) as uint).clone();
              y = hdata.get(((i + 1) * nn + i) as uint).clone();
              vr = (*ddata.get(i as uint) - p) * (*ddata.get(i as uint) - p) + *edata.get(i as uint) * *edata.get(i as uint) - q * q;
              vi = (*ddata.get(i as uint) - p) * num::cast(2.0).unwrap() * q;
              if (vr == num::zero()) && (vi == num::zero()) {
                vr = eps * norm * (num::abs(w.clone()) + num::abs(q.clone()) + num::abs(x.clone()) + num::abs(y.clone()) + num::abs(z.clone()));
              }
              let (cdivr, cdivi) = EigenDecomposition::cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              *hdata.get_mut((i * nn + (n - 1)) as uint) = cdivr;
              *hdata.get_mut((i * nn + n) as uint) = cdivi;
              if num::abs(x.clone()) > (num::abs(z.clone()) + num::abs(q.clone())) {
                *hdata.get_mut(((i + 1) * nn + (n - 1)) as uint) = (- ra - w * *hdata.get((i * nn + (n - 1)) as uint) + q * *hdata.get((i * nn + n) as uint)) / x;
                *hdata.get_mut(((i + 1) * nn + n) as uint) = (- sa - w * *hdata.get((i * nn + n) as uint) - q * *hdata.get((i * nn + (n - 1)) as uint)) / x;
              } else {
                let (cdivr, cdivi) = EigenDecomposition::cdiv(- r - y * *hdata.get((i * nn + (n - 1)) as uint), - s - y * *hdata.get((i * nn + n) as uint), z.clone(), q.clone());
                *hdata.get_mut(((i + 1) * nn + (n - 1)) as uint) = cdivr;
                *hdata.get_mut(((i + 1) * nn + n) as uint) = cdivi;
              }
            }

            // Overflow control
            t = num::abs(hdata.get((i * nn + (n - 1)) as uint).clone()).max(num::abs(hdata.get((i * nn + n) as uint).clone()));
            if (eps * t) * t > num::one() {
              for j in range(i, n + 1) {
                *hdata.get_mut((j * nn + (n - 1)) as uint) = *hdata.get((j * nn + (n - 1)) as uint) / t;
                *hdata.get_mut((j * nn + n) as uint) = *hdata.get((j * nn + n) as uint) / t;
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots
    for i in range(0, nn) {
      if (i < low) || (i > high) {
        for j in range(i, nn) {
          *vdata.get_mut((i * nn + j) as uint) = hdata.get((i * nn + j) as uint).clone();
        }
      }
    }

    // Back transformation to get eigenvectors of original matrix
    for j in range(low, nn).rev() {
      for i in range(low, high + 1) {
        z = num::zero();
        for k in range(low, cmp::min(j, high) + 1) {
          z = z + *vdata.get((i * nn + k) as uint) * *hdata.get((k * nn + j) as uint);
        }
        *vdata.get_mut((i * nn + j) as uint) = z;
      }
    }
  }

  pub fn new(a : &Matrix<T>) -> EigenDecomposition<T> {
    let n = a.cols();

    let mut vdata = alloc_dirty_vec(n * n);
    let mut ddata = alloc_dirty_vec(n);
    let mut edata = alloc_dirty_vec(n);

    let mut issymmetric = true;
    let mut j = 0;
    while (j < n) && issymmetric {
      let mut i = 0;
      while (i < n) && issymmetric {
        issymmetric = a.get(i, j) == a.get(j, i);
        i += 1;
      }
      j += 1;
    }

    if issymmetric {
      for i in range(0, n) {
        for j in range(0, n) {
          *vdata.get_mut(i * n + j) = a.get(i as uint, j as uint).clone();
        }
      }

      // Tridiagonalize.
      EigenDecomposition::tred2(n, &mut ddata, &mut vdata, &mut edata);

      // Diagonalize.
      EigenDecomposition::tql2(n, &mut edata, &mut ddata, &mut vdata);

      EigenDecomposition {
        n : n,
        d : ddata,
        e : edata,
        v : Matrix::new(n, n, vdata)
      }
    } else {
      let mut hdata = alloc_dirty_vec(n * n);

      for j in range(0, n) {
        for i in range(0, n) {
          *hdata.get_mut(i * n + j) = a.get(i as uint, j as uint);
        }
      }

      // Reduce to Hessenberg form.
      EigenDecomposition::orthes(n, &mut hdata, &mut vdata);
   
      // Reduce Hessenberg to real Schur form.
      EigenDecomposition::hqr2(n, &mut ddata, &mut edata, &mut hdata, &mut vdata);

      EigenDecomposition {
        n : n,
        d : ddata,
        e : edata,
        v : Matrix::new(n, n, vdata)
      }
    }
  }

  pub fn get_v<'lt>(&'lt self) -> &'lt Matrix<T> { &self.v }

  pub fn get_real_eigenvalues<'lt>(&'lt self) -> &'lt Vec<T> { &self.d }

  pub fn get_imag_eigenvalues<'lt>(&'lt self) -> &'lt Vec<T> { &self.e }

  pub fn get_d(&self) -> Matrix<T> {
    let mut ddata = alloc_dirty_vec(self.n * self.n);

    for i in range(0u, self.n) {
      for j in range(0u, self.n) {
        *ddata.get_mut((i * self.n + j) as uint) = num::zero();
      }
      *ddata.get_mut((i * self.n + i) as uint) = self.d.get(i as uint).clone();
      if *self.e.get(i as uint) > num::zero() {
        *ddata.get_mut((i * self.n + (i + 1)) as uint) = self.e.get(i as uint).clone();
      } else if *self.e.get(i as uint) < num::zero() {
        *ddata.get_mut((i * self.n + (i - 1)) as uint) = self.e.get(i as uint).clone();
      }
    }

    Matrix::new(self.n, self.n, ddata)
  }
}

#[test]
fn eigen_test() {
  let a = m!(3.0, 1.0, 6.0; 2.0, 1.0, 0.0; -1.0, 0.0, -3.0);
  let _eig = EigenDecomposition::new(&a);
  let r = _eig.get_real_eigenvalues();
  assert!(Matrix::vector(r.clone()).approx_eq(&m!(3.0; -1.0; -1.0)));
}
