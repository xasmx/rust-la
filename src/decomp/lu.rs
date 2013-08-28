use std::num;
use std::num::{One, Zero};

use super::super::matrix::*;
use super::super::util::{alloc_dirty_vec};

pub struct LUDecomposition<T> {
  lu : Matrix<T>,
  pospivsign : bool,
  piv : ~[uint]
}

// Ported from JAMA.
impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed> LUDecomposition<T> {
  pub fn new(a : &Matrix<T>) -> LUDecomposition<T> {
    // Use a "left-looking", dot-product, Crout/Doolittle algorithm.
    let mut ludata = a.data.clone();
    let m = a.noRows as int;
    let n = a.noCols as int;
    let mut pivdata = alloc_dirty_vec(m as uint);
    for i in range(0, m) {
      pivdata[i] = i as uint;
    }

    let mut pospivsign = true;
    let mut lucolj = alloc_dirty_vec(m as uint);

    // Outer loop.
    for j in range(0, n) {
      // Make a copy of the j-th column to localize references.
      for i in range(0, m) {
        lucolj[i] = ludata[i * n + j].clone();
      }

      // Apply previous transformations.
      for i in range(0, m) {
        let lurowiIdx = i * n;

        // Most of the time is spent in the following dot product.
        let kmax = num::min(i, j);
        let mut s : T = Zero::zero();
        for k in range(0, kmax) {
          s = s + ludata[lurowiIdx + k] * lucolj[k];
        }

        lucolj[i] = lucolj[i] - s;
        ludata[lurowiIdx + j] = lucolj[i].clone();
      }

      // Find pivot and exchange if necessary.
      let mut p = j;
      for i in range(j + 1, m) {
        if(num::abs(lucolj[i].clone()) > num::abs(lucolj[p].clone())) {
          p = i;
        }
      }

      if(p != j) {
        for k in range(0, n) {
          let t = ludata[p * n + k].clone();
          ludata[p * n + k] = ludata[j * n + k].clone();
          ludata[j * n + k] = t;
        }

        let k = pivdata[p];
        pivdata[p] = pivdata[j];
        pivdata[j] = k;
        pospivsign = !pospivsign;
      }

      // Compute multipliers.
      if((j < m) && (ludata[j * n + j] != Zero::zero())) {
        for i in range(j + 1, m) {
          ludata[i * n + j] = ludata[i * n + j] / ludata[j * n + j];
        }
      }
    }

    LUDecomposition { 
      lu : Matrix { noRows : m as uint, noCols : n as uint, data : ludata },
      pospivsign : pospivsign,
      piv : pivdata
    }
  }

  pub fn is_nonsingular(&self) -> bool {
    let n = self.lu.noCols;
    for j in range(0, n) {
      if(self.lu.data[j * n + j] == Zero::zero()) {
        return false;
      }
    }
    true
  }

  pub fn get_l(&self) -> Matrix<T> {
    let m = self.lu.noRows;
    let n = self.lu.noCols;
    let mut ldata = alloc_dirty_vec(m * n);
    for i in range(0, m) {
      for j in range(0, n) {
        ldata[i * n + j] = if(i > j) {
                             self.lu.data[i * n + j].clone()
                           } else if(i == j) {
                             One::one()
                           } else {
                             Zero::zero()
                           }
      }
    }
    Matrix { noRows : m, noCols : n, data : ldata }
  }

  pub fn get_u(&self) -> Matrix<T> {
    let n = self.lu.noCols as int;
    let mut udata = alloc_dirty_vec((n * n) as uint);
    for i in range(0, n) {
      for j in range(0, n) {
        udata[i * n + j] = if(i <= j) { self.lu.data[i * n + j].clone() } else { Zero::zero() };
      }
    }
    Matrix { noRows : n as uint, noCols : n as uint, data : udata }
  }

  pub fn det(&self) -> T {
    assert!(self.lu.noRows == self.lu.noCols);
    let n = self.lu.noCols as int;
    let mut d = if self.pospivsign { One::one::<T>() } else { - One::one::<T>() };
    for j in range(0, n) {
      d = d * self.lu.data[j * n + j];
    }
    d
  }

  // Solve A*X = B
  // B   A Matrix with as many rows as A and any number of columns.
  // Returns X so that L*U*X = B(piv,:)
  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    let m = self.lu.noRows as int;
    let n = self.lu.noCols as int;
    assert!(b.noRows == m as uint);
    if(!self.is_nonsingular()) {
      return None
    }

    // Copy right hand side with pivoting
    let nx = b.noCols as int;
    let mut xdata = alloc_dirty_vec((m * nx) as uint);
    let mut destIdx = 0;
    for i in range(0, self.piv.len()) {
      for j in range(0, nx) {
        xdata[destIdx] = b.data[(self.piv[i] as int) * (b.noCols as int) + j].clone();
        destIdx += 1;
      }
    }

    // Solve L*Y = B(piv,:)
    for k in range(0, n) {
      for i in range(k + 1, n) {
        for j in range(0, nx) {
          xdata[i * nx + j] = xdata[i * nx + j] - xdata[k * nx + j] * self.lu.data[i * (self.lu.noCols as int) + k];
        }
      }
    }

    // Solve U*X = Y;
    for k in range(0, n).invert() {
      for j in range(0, nx) {
        xdata[k * nx + j] = xdata[k * nx + j] / self.lu.data[k * (self.lu.noCols as int) + k];
      }
      for i in range(0, k) {
        for j in range(0, nx) {
          xdata[i * nx + j] = xdata[i * nx + j] - xdata[k * nx + j] * self.lu.data[i * (self.lu.noCols as int) + k];
        }
      }
    }

    Some(Matrix {
      noRows : self.lu.noRows,
      noCols : b.noCols,
      data : xdata
    })
  }
}

#[test]
fn test_lu() {
  let a = matrix(3, 3, ~[1.0, 2.0, 0.0, 3.0, 6.0, -1.0, 1.0, 2.0, 1.0]);
  let lu = LUDecomposition::new(&a);
  assert!((lu.get_l().permute_rows(lu.piv) * lu.get_u()).approx_eq(&a));
}
