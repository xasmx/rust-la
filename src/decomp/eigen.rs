use std::cmp;
use num;
use num::traits::{Float, Signed};

use ApproxEq;
use Matrix;
use internalutil::{alloc_dirty_vec, hypot};

/// Eigenvalues and eigenvectors of a real matrix. 
///
/// Ported from JAMA.
///
/// If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is
/// diagonal and the eigenvector matrix V is orthogonal.
/// I.e. A = V * D * V' and V * V' = I.
///
/// If A is not symmetric, then the eigenvalue matrix D is block diagonal
/// with the real eigenvalues in 1-by-1 blocks and any complex eigenvalues,
/// lambda + i*mu, in 2-by-2 blocks, [lambda, mu; -mu, lambda].  The
/// columns of V represent the eigenvectors in the sense that A*V = V*D,
/// The matrix V may be badly conditioned, or even singular, so the validity
/// of the equation A = V * D * V^-1 depends upon V.cond().
pub struct EigenDecomposition<T> {
  n : usize,
  d : Vec<T>,
  e : Vec<T>,
  v : Matrix<T>
}

//impl<T : FloatMath + ApproxEq<T>> EigenDecomposition<T> {
impl<T : Float + ApproxEq<T> + Signed> EigenDecomposition<T> {
  // Symmetric Householder reduction to tridiagonal form.
  fn tred2(n : usize, ddata : &mut Vec<T>, vdata : &mut Vec<T>, edata : &mut Vec<T>) {
    //  This is derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.    
    for j in 0..n {
      ddata[j] = vdata[(n - 1) * n + j];
    }

    // Householder reduction to tridiagonal form.
    for i in (1..n).rev() {
      // Scale to avoid under/overflow.
      let mut scale : T = num::zero();
      let mut h : T = num::zero();
      for k in 0..i {
        scale = scale + num::abs(ddata[k]);
      }
      if scale == num::zero() {
        edata[i] = ddata[i - 1];
        for j in 0..i {
          ddata[j] = vdata[(i - 1) * n + j];
          vdata[i * n + j] = num::zero();
          vdata[j * n + i] = num::zero();
        }
      } else {
        // Generate Householder vector.
        for k in 0..i {
          ddata[k] = ddata[k] / scale;
          h = h + ddata[k] * ddata[k];
        }
        let mut f = ddata[i - 1];
        let mut g = h.sqrt();
        if f > num::zero() {
          g = - g;
        }
        edata[i] = scale * g;
        h = h - f * g;
        ddata[i - 1] = f - g;
        for j in 0..i {
          edata[j] = num::zero();
        }

        // Apply similarity transformation to remaining columns.
        for j in 0..i {
          f = ddata[j];
          vdata[j * n + i] = f;
          g = edata[j] + vdata[j * n + j] * f;
          for k in (j + 1)..i {
            g = g + vdata[k * n + j] * ddata[k];
            edata[k] = edata[k] + vdata[k * n + j] * f;
          }
          edata[j] = g;
        }
        f = num::zero();
        for j in 0..i {
          edata[j] = edata[j] / h;
          f = f + edata[j] * ddata[j];
        }
        let hh = f / (h + h);
        for j in 0..i {
          edata[j] = edata[j] - hh * ddata[j];
        }
        for j in 0..i {
          f = ddata[j];
          g = edata[j];
          for k in j..i {
            let orig_val = vdata[k * n + j];
            vdata[k * n + j] = orig_val - (f * edata[k] + g * ddata[k]);
          }
          ddata[j] = vdata[(i - 1) * n + j];
          vdata[i * n + j] = num::zero();
        }
      }
      ddata[i] = h;
    }

    // Accumulate transformations.
    for i in 0..(n - 1) {
      let orig_val = vdata[i * n + i];
      vdata[(n - 1) * n + i] = orig_val;
      vdata[i * n + i] = num::one();
      let h = ddata[i + 1];
      if h != num::zero() {
        for k in 0..(i + 1) {
          ddata[k] = vdata[k * n + (i + 1)] / h;
        }
        for j in 0..(i + 1) {
          let mut g : T = num::zero();
          for k in 0..(i + 1) {
            g = g + vdata[k * n + (i + 1)] * vdata[k * n + j];
          }
          for k in 0..(i + 1) {
            let orig_val = vdata[k * n + j];
            vdata[k * n + j] = orig_val - g * ddata[k];
          }
        }
      }
      for k in 0..(i + 1) {
        vdata[k * n + (i + 1)] = num::zero();
      }
    }
    for j in 0..n {
      ddata[j] = vdata[(n - 1) * n + j];
      vdata[(n - 1) * n + j] = num::zero();
    }
    vdata[(n - 1) * n + (n - 1)] = num::one();
    edata[0] = num::zero();
  }

  // Symmetric tridiagonal QL algorithm.
  fn tql2(n : usize, edata : &mut Vec<T>, ddata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    // Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
    for i in 1..n {
      edata[i - 1] = edata[i];
    }
    edata[n - 1] = num::zero();

    let mut f : T = num::zero();
    let mut tst1 : T = num::zero();
    let eps : T = num::cast(2.0f64.powf(-52.0)).unwrap();
    for l in 0..n {
      // Find small subdiagonal element
      tst1 = tst1.max(num::abs(ddata[l]) + num::abs(edata[l]));
      let mut m = l;
      while m < n {
        if num::abs(edata[m]) <= (eps * tst1) {
          break;
        }
        m += 1;
      }

      // If m == l, d[l] is an eigenvalue, otherwise, iterate.
      if m > l {
        loop {
          // Compute implicit shift
          let mut g = ddata[l];
          let tmp : T = num::cast(2.0).unwrap();
          let mut p = (ddata[l + 1] - g) / (tmp * edata[l]);
          let mut r = hypot::<T>(p, num::one());
          if p < num::zero() {
            r = -r;
          }
          ddata[l] = edata[l] / (p + r);
          ddata[l + 1] = edata[l] * (p + r);
          let dl1 = ddata[l + 1];
          let mut h = g - ddata[l];
          for i in (l + 2)..n {
            ddata[i] = ddata[i] - h;
          }
          f = f + h;

          // Implicit QL transformation.
          p = ddata[m];
          let mut c : T = num::one();
          let mut c2 = c;
          let mut c3 = c;
          let el1 = edata[l + 1];
          let mut s : T = num::zero();
          let mut s2 = num::zero();
          for i in (l..m).rev() {
            c3 = c2;
            c2 = c;
            s2 = s;
            g = c * edata[i];
            h = c * p;
            r = hypot::<T>(p, edata[i]);
            edata[i + 1] = s * r;
            s = edata[i] / r;
            c = p / r;
            p = c * ddata[i] - s * g;
            ddata[i + 1] = h + s * (c * g + s * ddata[i]);

            // Accumulate transformation.
            for k in 0..n {
              h = vdata[k * n + (i + 1)];
              vdata[k * n + (i + 1)] = s * vdata[k * n + i] + c * h;
              vdata[k * n + i] = c * vdata[k * n + i] - s * h;
            }
          }
          p = - s * s2 * c3 * el1 * edata[l] / dl1;
          edata[l] = s * p;
          ddata[l] = c * p;

          // Check for convergence.
          if num::abs(edata[l]) <= (eps * tst1) {
            break;
          }
        }
      }
      ddata[l] = ddata[l] + f;
      edata[l] = num::zero();
    }

    // Bubble sort eigenvalues and corresponding vectors.
    for i in 0..(n - 1) {
      let mut k = i;
      let mut p = ddata[i];
      for j in (i + 1)..n {
        if ddata[j] > p {
          k = j;
          p = ddata[j];
        }
      }
      if k != i {
        // Swap columns k and i of the diagonal and v.
        ddata[k] = ddata[i];
        ddata[i] = p;
        for j in 0..n {
          p = vdata[j * n + i];
          vdata[j * n + i] = vdata[j * n + k];
          vdata[j * n + k] = p;
        }
      }
    }
  }

  // Nonsymmetric reduction to Hessenberg form.
  fn orthes(n : usize, hdata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedures orthes and ortran, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutines in EISPACK.

    let mut ort = alloc_dirty_vec(n);

    let low = 0;
    let high = n - 1;

    for m in (low + 1)..high {
      // Scale column.
      let mut scale : T = num::zero();
      for i in m..(high + 1) {
        scale = scale + num::abs(hdata[i * n + (m - 1)]);
      }
      if scale != num::zero() {
        // Compute Householder transformation.
        let mut h : T = num::zero();
        for i in (m..(high + 1)).rev() {
          ort[i] = hdata[i * n + (m - 1)] / scale;
          h = h + ort[i] * ort[i];
        }
        let mut g = h.sqrt();
        if ort[m] > num::zero() {
          g = -g;
        }
        h = h - ort[m] * g;
        ort[m] = ort[m] - g;

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)
        for j in m..n {
          let mut f : T = num::zero();
          for i in (m..(high + 1)).rev() {
            f = f + ort[i] * hdata[i * n + j];
          }
          f = f / h;
          for i in m..(high + 1) {
            hdata[i * n + j] = hdata[i * n + j] - f * ort[i];
          }
        }

        for i in 0..(high + 1) {
          let mut f : T = num::zero();
          for j in (m..(high + 1)).rev() {
            f = f + ort[j] * hdata[i * n + j];
          }
          f = f / h;
          for j in m..(high + 1) {
            hdata[i * n + j] = hdata[i * n + j] - f * ort[j];
          }
        }
        ort[m] = scale * ort[m];
        hdata[m * n + (m - 1)] = scale * g;
      }
    }

    // Accumulate transformations (Algol's ortran).
    for i in 0..n {
      for j in 0..n {
        vdata[i * n + j] = if i == j { num::one() } else { num::zero() };
      }
    }

    for m in ((low + 1)..high).rev() {
      if hdata[m * n + (m - 1)] != num::zero() {
        for i in (m + 1)..(high + 1) {
          ort[i] = hdata[i * n + (m - 1)];
        }
        for j in m..(high + 1) {
          let mut g : T = num::zero();
          for i in m..(high + 1) {
            g = g + ort[i] * vdata[i * n + j];
          }
          // Double division avoids possible underflow
          g = (g / ort[m]) / hdata[m * n + (m - 1)];
          for i in m..(high + 1) {
            vdata[i * n + j] = vdata[i * n + j] + g * ort[i];
          }
        }
      }
    }
  }

  // Complex scalar division.
  fn cdiv(xr : T, xi : T, yr : T, yi : T) -> (T, T) {
    if num::abs(yr) > num::abs(yi) {
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
  fn hqr2(n : usize, ddata : &mut Vec<T>, edata : &mut Vec<T>, hdata : &mut Vec<T>, vdata : &mut Vec<T>) {
    // This is derived from the Algol procedure hqr2, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.

    // Initialize
    let nn = n as isize;
    let mut n = nn - 1;
    let low : isize = 0;
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
    for i in 0..nn {
      if (i < low) || (i > high) {
        ddata[i as usize] = hdata[(i * nn + i) as usize];
        edata[i as usize] = num::zero();
      }
      for j in cmp::max(i - 1, 0)..nn {
        norm = norm + num::abs(hdata[(i * nn + j) as usize]);
      }
    }

    // Outer loop over eigenvalue index
    let mut iter = 0;
    while n >= low {

      // Look for single small sub-diagonal element
      let mut l = n;
      while l > low {
        s = num::abs(hdata[((l - 1) * nn + (l - 1)) as usize]) + num::abs(hdata[(l * nn + l) as usize]);
        if s == num::zero() {
          s = norm;
        }
        if num::abs(hdata[(l * nn + (l - 1)) as usize]) < (eps * s) {
          break;
        }
        l -= 1;
      }

      // Check for convergence.
      if l == n {
        //One root found.
        hdata[(n * nn + n) as usize] = hdata[(n * nn + n) as usize] + exshift;
        ddata[n as usize] = hdata[(n * nn + n) as usize];
        edata[n as usize] = num::zero();
        n -= 1;
        iter = 0;
      } else if l == (n - 1) {
        // Two roots found
        w = hdata[(n * nn + (n - 1)) as usize] * hdata[((n - 1) * nn + n) as usize];
        p = (hdata[((n - 1) * nn + (n - 1)) as usize] - hdata[(n * nn + n) as usize]) / num::cast(2.0).unwrap();
        q = p * p + w;
        z = num::abs(q).sqrt();
        hdata[(n * nn + n) as usize] = hdata[(n * nn + n) as usize] + exshift;
        hdata[((n - 1) * nn + (n - 1)) as usize] = hdata[((n - 1) * nn + (n - 1)) as usize] + exshift;
        x = hdata[(n * nn + n) as usize];

        // Real pair
        if q >= num::zero() {
          z = if p >= num::zero() { p + z } else { p - z };
          ddata[(n - 1) as usize] = x + z;
          ddata[n as usize] = ddata[(n - 1) as usize];
          if z != num::zero() {
            ddata[n as usize] = x - w / z;
          }
          edata[(n - 1) as usize] = num::zero();
          edata[n as usize] = num::zero();
          x = hdata[(n * nn + (n - 1)) as usize];
          s = num::abs(x) + num::abs(z);
          p = x / s;
          q = z / s;
          r = (p * p + q * q).sqrt();
          p = p / r;
          q = q / r;

          // Row modification
          for j in (n - 1)..nn {
            z = hdata[((n - 1) * nn + j) as usize];
            hdata[((n - 1) * nn + j) as usize] = q * z + p * hdata[(n * nn + j) as usize];
            hdata[(n * nn + j) as usize] = q * hdata[(n * nn + j) as usize] - p * z;
          }

          // Column modification
          for i in 0..(n + 1) {
            z = hdata[(i * nn + (n - 1)) as usize];
            hdata[(i * nn + (n - 1)) as usize] = q * z + p * hdata[(i * nn + n) as usize];
            hdata[(i * nn + n) as usize] = q * hdata[(i * nn + n) as usize] - p * z;
          }

          // Accumulate transformations
          for i in low..(high + 1) {
            z = vdata[(i * nn + (n - 1)) as usize];
            vdata[(i * nn + (n - 1)) as usize] = q * z + p * vdata[(i * nn + n) as usize];
            vdata[(i * nn + n) as usize] = q * vdata[(i * nn + n) as usize] - p * z;
          }
        } else {
          // Complex pair
          ddata[(n - 1) as usize] = x + p;
          ddata[n as usize] = x + p;
          edata[(n - 1) as usize] = z;
          edata[n as usize] = - z;
        }
        n = n - 2;
        iter = 0;
      } else {
        // No convergence yet

        // Form shift
        x = hdata[(n * nn + n) as usize];
        y = num::zero();
        w = num::zero();
        if l < n {
          y = hdata[((n - 1) * nn + (n - 1)) as usize];
          w = hdata[(n * nn + (n - 1)) as usize] * hdata[((n - 1) * nn + n) as usize];
        }

        // Wilkinson's original ad hoc shift
        if iter == 10 {
          exshift = exshift + x;
          for i in low..(n + 1) {
            hdata[(i * nn + i) as usize] = hdata[(i * nn + i) as usize] - x;
          }
          s = num::abs(hdata[(n * nn + (n - 1)) as usize]) + num::abs(hdata[((n - 1) * nn + (n - 2)) as usize]);
          let tmp : T = num::cast(0.75).unwrap();
          y = tmp * s;
          x = y;
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
            for i in low..(n + 1) {
              hdata[(i * nn + i) as usize] = hdata[(i * nn + i) as usize] - s;
            }
            exshift = exshift + s;
            w = num::cast(0.964).unwrap();
            y = w;
            x = y;
          }
        }

        iter += 1;

        // Look for two consecutive small sub-diagonal elements
        let mut m = n - 2;
        while m >= l {
          z = hdata[(m * nn + m) as usize];
          r = x - z;
          s = y - z;
          p = (r * s - w) / hdata[((m + 1) * nn + m) as usize] + hdata[(m * nn + (m + 1)) as usize];
          q = hdata[((m + 1) * nn + (m + 1)) as usize] - z - r - s;
          r = hdata[((m + 2) * nn + (m + 1)) as usize];
          s = num::abs(p) + num::abs(q) + num::abs(r);
          p = p / s;
          q = q / s;
          r = r / s;
          if m == l {
            break;
          }
          if (num::abs(hdata[(m * nn + (m - 1)) as usize]) * (num::abs(q) + num::abs(r))) <
             eps * (num::abs(p) * (num::abs(hdata[((m - 1) * nn + (m - 1)) as usize]) + num::abs(z) + num::abs(hdata[((m + 1) * nn + (m + 1)) as usize]))) {
            break;
          }
          m -= 1;
        }

        for i in (m + 2)..(n + 1) {
          hdata[(i * nn + (i - 2)) as usize] = num::zero();
          if i > (m + 2) {
            hdata[(i * nn + (i - 3)) as usize] = num::zero();
          }
        }

        // Double QR step involving rows l:n and columns m:n
        for k in m..n {
          let notlast = k != (n - 1);
          if k != m {
            p = hdata[(k * nn + (k - 1)) as usize];
            q = hdata[((k + 1) * nn + (k - 1)) as usize];
            r = if notlast { hdata[((k + 2) * nn + (k - 1)) as usize] } else { num::zero() };
            x = num::abs(p) + num::abs(q) + num::abs(r);
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
              hdata[(k * nn + (k - 1)) as usize] = - s * x;
            } else if l != m {
              hdata[(k * nn + (k - 1)) as usize] = - hdata[(k * nn + (k - 1)) as usize];
            }
            p = p + s;
            x = p / s;
            y = q / s;
            z = r / s;
            q = q / p;
            r = r / p;

            // Row modification
            for j in k..nn {
              p = hdata[(k * nn + j) as usize] + q * hdata[((k + 1) * nn + j) as usize];
              if notlast {
                p = p + r * hdata[((k + 2) * nn + j) as usize];
                hdata[((k + 2) * nn + j) as usize] = hdata[((k + 2) * nn + j) as usize] - p * z;
              }
              hdata[(k * nn + j) as usize] = hdata[(k * nn + j) as usize] - p * x;
              hdata[((k + 1) * nn + j) as usize] = hdata[((k + 1) * nn + j) as usize] - p * y;
            }

            // Column modification
            for i in 0..(cmp::min(n, k + 3) + 1) {
              p = x * hdata[(i * nn + k) as usize] + y * hdata[(i * nn + (k + 1)) as usize];
              if notlast {
                p = p + z * hdata[(i * nn + (k + 2)) as usize];
                hdata[(i * nn + (k + 2)) as usize] = hdata[(i * nn + (k + 2)) as usize] - p * r;
              }
              hdata[(i * nn + k) as usize] = hdata[(i * nn + k) as usize] - p;
              hdata[(i * nn + (k + 1)) as usize] = hdata[(i * nn + (k + 1)) as usize] - p * q;
            }

            // Accumulate transformations
            for i in low..(high + 1) {
              p = x * vdata[(i * nn + k) as usize] + y * vdata[(i * nn + (k + 1)) as usize];
              if notlast {
                p = p + z * vdata[(i * nn + (k + 2)) as usize];
                vdata[(i * nn + (k + 2)) as usize] = vdata[(i * nn + (k + 2)) as usize] - p * r;
              }
              vdata[(i * nn + k) as usize] = vdata[(i * nn + k) as usize] - p;
              vdata[(i * nn + (k + 1)) as usize] = vdata[(i * nn + (k + 1)) as usize] - p * q;
            }
          }
        }
      }
    }

    // Backsubstitute to find vectors of upper triangular form
    if norm == num::zero() {
      return;
    }

    for n in (0..nn).rev() {
      p = ddata[n as usize];
      q = edata[n as usize];

      // Real vector
      if q == num::zero() {
        let mut l = n;
        hdata[(n * nn + n) as usize] = num::one();
        for i in (0..n).rev() {
          w = hdata[(i * nn + i) as usize] - p;
          r = num::zero();
          for j in l..(n + 1) {
            r = r + hdata[(i * nn + j) as usize] * hdata[(j * nn + n) as usize];
          }
          if edata[i as usize] < num::zero() {
            z = w;
            s = r;
          } else {
            l = i;
            if edata[i as usize] == num::zero() {
              if w != num::zero() {
                hdata[(i * nn + n) as usize] = - r / w;
              } else {
                hdata[(i * nn + n) as usize] = - r / (eps * norm);
              }
            } else {
              // Solve real equations
              x = hdata[(i * nn + (i + 1)) as usize];
              y = hdata[((i + 1) * nn + i) as usize];
              q = (ddata[i as usize] - p) * (ddata[i as usize] - p) + edata[i as usize] * edata[i as usize];
              t = (x * s - z * r) / q;
              hdata[(i * nn + n) as usize] = t;
              if num::abs(x) > num::abs(z) {
                hdata[((i + 1) * nn + n) as usize] = (-r - w * t) / x;
              } else {
                hdata[((i + 1) * nn + n) as usize] = (-s - y * t) / z;
              }
            }

            // Overflow control
            t = num::abs(hdata[(i * nn + n) as usize]);
            if (eps * t) * t > num::one() {
              for j in i..(n + 1) {
                hdata[(j * nn + n) as usize] = hdata[(j * nn + n) as usize] / t;
              }
            }
          }
        }
      } else if q < num::zero() {
        // Complex vector
        let mut l = n - 1;

        // Last vector component imaginary so matrix is triangular
        if num::abs(hdata[(n * nn + (n - 1)) as usize]) > num::abs(hdata[((n - 1) * nn + n) as usize]) {
          hdata[((n - 1) * nn + (n - 1)) as usize] = q / hdata[(n * nn + (n - 1)) as usize];
          hdata[((n - 1) * nn + n) as usize] = - (hdata[(n * nn + n) as usize] - p) / hdata[(n * nn + (n - 1)) as usize];
        } else {
          let (cdivr, cdivi) = EigenDecomposition::<T>::cdiv(num::zero(), - hdata[((n - 1) * nn + n) as usize], hdata[((n - 1) * nn + (n - 1)) as usize] - p, q);
          hdata[((n - 1) * nn + (n - 1)) as usize] = cdivr;
          hdata[((n - 1) * nn + n) as usize] = cdivi;
        }
        hdata[(n * nn + (n - 1)) as usize] = num::zero();
        hdata[(n * nn + n) as usize] = num::one();
        for i in (0..(n - 1)).rev() {
          let mut ra : T = num::zero();
          let mut sa : T = num::zero();
          let mut vr;
          let vi;
          for j in l..(n + 1) {
            ra = ra + hdata[(i * nn + j) as usize] * hdata[(j * nn + (n - 1)) as usize];
            sa = sa + hdata[(i * nn + j) as usize] * hdata[(j * nn + n) as usize];
          }
          w = hdata[(i * nn + i) as usize] - p;

          if edata[i as usize] < num::zero() {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if edata[i as usize] == num::zero() {
              let (cdivr, cdivi) = EigenDecomposition::cdiv(- ra, - sa, w, q);
              hdata[(i * nn + (n - 1)) as usize] = cdivr;
              hdata[(i * nn + n) as usize] = cdivi;
            } else {
              // Solve complex equations
              x = hdata[(i * nn + (i + 1)) as usize];
              y = hdata[((i + 1) * nn + i) as usize];
              vr = (ddata[i as usize] - p) * (ddata[i as usize] - p) + edata[i as usize] * edata[i as usize] - q * q;
              vi = (ddata[i as usize] - p) * num::cast(2.0).unwrap() * q;
              if (vr == num::zero()) && (vi == num::zero()) {
                vr = eps * norm * (num::abs(w) + num::abs(q) + num::abs(x) + num::abs(y) + num::abs(z));
              }
              let (cdivr, cdivi) = EigenDecomposition::cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              hdata[(i * nn + (n - 1)) as usize] = cdivr;
              hdata[(i * nn + n) as usize] = cdivi;
              if num::abs(x) > (num::abs(z) + num::abs(q)) {
                hdata[((i + 1) * nn + (n - 1)) as usize] = (- ra - w * hdata[(i * nn + (n - 1)) as usize] + q * hdata[(i * nn + n) as usize]) / x;
                hdata[((i + 1) * nn + n) as usize] = (- sa - w * hdata[(i * nn + n) as usize] - q * hdata[(i * nn + (n - 1)) as usize]) / x;
              } else {
                let (cdivr, cdivi) = EigenDecomposition::cdiv(- r - y * hdata[(i * nn + (n - 1)) as usize], - s - y * hdata[(i * nn + n) as usize], z, q);
                hdata[((i + 1) * nn + (n - 1)) as usize] = cdivr;
                hdata[((i + 1) * nn + n) as usize] = cdivi;
              }
            }

            // Overflow control
            t = num::abs(hdata[(i * nn + (n - 1)) as usize]).max(num::abs(hdata[(i * nn + n) as usize]));
            if (eps * t) * t > num::one() {
              for j in i..(n + 1) {
                hdata[(j * nn + (n - 1)) as usize] = hdata[(j * nn + (n - 1)) as usize] / t;
                hdata[(j * nn + n) as usize] = hdata[(j * nn + n) as usize] / t;
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots
    for i in 0..nn {
      if (i < low) || (i > high) {
        for j in i..nn {
          vdata[(i * nn + j) as usize] = hdata[(i * nn + j) as usize];
        }
      }
    }

    // Back transformation to get eigenvectors of original matrix
    for j in (low..nn).rev() {
      for i in low..(high + 1) {
        z = num::zero();
        for k in low..(cmp::min(j, high) + 1) {
          z = z + vdata[(i * nn + k) as usize] * hdata[(k * nn + j) as usize];
        }
        vdata[(i * nn + j) as usize] = z;
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
      for i in 0..n {
        for j in 0..n {
          vdata[i * n + j] = a.get(i, j);
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

      for j in 0..n {
        for i in 0..n {
          hdata[i * n + j] = a.get(i, j);
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

    for i in 0..self.n {
      for j in 0..self.n {
        ddata[i * self.n + j] = num::zero();
      }
      ddata[i * self.n + i] = self.d[i];
      if self.e[i] > num::zero() {
        ddata[i * self.n + (i + 1)] = self.e[i];
      } else if self.e[i] < num::zero() {
        ddata[i * self.n + (i - 1)] = self.e[i];
      }
    }

    Matrix::new(self.n, self.n, ddata)
  }
}

#[test]
fn eigen_test_symmetric() {
  let a = m!(3.0, 1.0, 6.0; 2.0, 1.0, 0.0; -1.0, 0.0, -3.0);
  let ata = a.t() * a;
  let _eig = EigenDecomposition::new(&ata);
  let r = _eig.get_real_eigenvalues();
  assert!(Matrix::vector(r.clone()).approx_eq(&m!(56.661209; 4.301868; 0.036923)));
}

#[test]
fn eigen_test_asymmetric() {
  let a = m!(3.0, 1.0, 6.0; 2.0, 1.0, 0.0; -1.0, 0.0, -3.0);
  let _eig = EigenDecomposition::new(&a);
  let r = _eig.get_real_eigenvalues();
  assert!(Matrix::vector(r.clone()).approx_eq(&m!(3.0; -1.0; -1.0)));
}

