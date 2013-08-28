use std::io;
use std::num;
use std::num::{One, Zero};
use std::rand;
use std::rand::{Rand};
use std::vec;

use super::decomp::lu;
use super::util::{alloc_dirty_vec};

#[deriving(Clone, Eq)]
pub struct Matrix<T> {
  noRows : uint,
  noCols : uint,
  data : ~[T]
}

pub fn matrix<T>(noRows : uint, noCols : uint, data : ~[T]) -> Matrix<T> {
  assert!(noRows * noCols == data.len());
  assert!(noRows > 0 && noCols > 0);
  Matrix { noRows : noRows, noCols : noCols, data : data }
}

pub fn random<T : Rand>(noRows : uint, noCols : uint) -> Matrix<T> {
  let elems = noRows * noCols;
  let mut d = alloc_dirty_vec(elems);
  for i in range(0u, elems) {
    d[i] = rand::random::<T>();
  }
  Matrix { noRows : noRows, noCols : noCols, data : d }
}

pub fn id<T : One + Zero + Clone>(n : uint) -> Matrix<T> {
  let mut d = vec::from_elem(n * n, Zero::zero());
  for i in range(0u, n) {
    d[i * n + i] = One::one();
  }
  Matrix { noRows : n, noCols : n, data : d }
}

pub fn zero<T : Zero + Clone>(noRows : uint, noCols : uint) -> Matrix<T> {
  Matrix {
    noRows : noRows,
    noCols : noCols,
    data : vec::from_elem(noRows * noCols, Zero::zero())
  }
}

pub fn vect<T>(data : ~[T]) -> Matrix<T> {
  assert!(data.len() > 0);
  Matrix { noRows : data.len(), noCols : 1, data : data }
}

pub fn zero_vect<T : Zero + Clone>(noRows : uint) -> Matrix<T> {
  Matrix { noRows : noRows, noCols : 1, data : vec::from_elem(noRows, Zero::zero()) }
}

pub fn one_vect<T : One + Clone>(noRows : uint) -> Matrix<T> {
  Matrix { noRows : noRows, noCols : 1, data : vec::from_elem(noRows, One::one()) }
}

pub fn row_vect<T>(data : ~[T]) -> Matrix<T> {
  assert!(data.len() > 0);
  Matrix { noRows : 1, noCols : data.len(), data : data }
}

impl<T> Matrix<T> {
  #[inline]
  pub fn rows(&self) -> uint { self.noRows }
}

impl<T> Matrix<T> {
  #[inline]
  pub fn cols(&self) -> uint { self.noCols }
}

impl<T : Clone> Matrix<T> {
  pub fn get(&self, row : uint, col : uint) -> T { self.data[row * self.noCols + col].clone() }
}

impl<T : Clone> Matrix<T> {
  pub fn get_ref<'lt>(&'lt self, row : uint, col : uint) -> &'lt T { &self.data[row * self.noCols + col] }
}

impl<T : Clone> Matrix<T> {
  pub fn get_mref<'lt>(&'lt mut self, row : uint, col : uint) -> &'lt mut T { &mut self.data[row * self.noCols + col] }
}

impl<T : Clone> Matrix<T> {
  pub fn set(&mut self, row : uint, col : uint, val : T) { self.data[row * self.noCols + col] = val.clone() }
}

impl<S, T> Matrix<S> {
  pub fn map(&self, f : &fn(&S) -> T) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = f(&self.data[i]);
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl<T> Matrix<T> {
  pub fn mmap(&mut self, f : &fn(&T) -> T) {
    for i in range(0u, self.data.len()) {
      self.data[i] = f(&self.data[i]);
    }
  }
}

impl<T : ApproxEq<T>> Matrix<T> {
  pub fn approx_eq(&self, m : &Matrix<T>) -> bool {
    if self.noRows != m.noRows || self.noCols != m.noCols { return false };
    for i in range(0u, self.data.len()) {
      if !self.data[i].approx_eq(&m.data[i]) { return false }
    }
    true
  }
}

impl<T : Clone> Matrix<T> {
  pub fn cr(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noRows == m.noRows);
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    let mut srcIdx1 = 0;
    let mut srcIdx2 = 0;
    let mut destIdx = 0;
    for _ in range(0u, self.noRows) {
      for _ in range(0u, self.noCols) {
        d[destIdx] = self.data[srcIdx1].clone();
        srcIdx1 += 1;
        destIdx += 1;
      }
      for _ in range(0u, m.noCols) {
        d[destIdx] = m.data[srcIdx2].clone();
        srcIdx2 += 1;
        destIdx += 1;
      }
    }
    Matrix {
      noRows : self.noRows,
      noCols : self.noCols + m.noCols,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn cb(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noCols == m.noCols);
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, self.data.len()) {
      d[i] = self.data[i].clone();
    }
    let offset = self.data.len();
    for i in range(0u, m.data.len()) {
      d[offset + i] = m.data[i].clone();
    }
    Matrix {
      noRows : self.noRows + m.noRows,
      noCols : self.noCols,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn t(&self) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    let mut srcIdx = 0;
    for i in range(0u, elems) {
      d[i] = self.data[srcIdx].clone();
      srcIdx += self.noCols;
      if(srcIdx >= elems) {
        srcIdx -= elems;
        srcIdx += 1;
      }
    }
    Matrix {
      noRows: self.noCols,
      noCols: self.noRows,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn mt(&mut self) {
    let mut visited = vec::from_elem(self.data.len(), false);

    for cycleIdx in range(1u, self.data.len() - 1) {
      if(visited[cycleIdx]) {
        loop;
      }

      let mut idx = cycleIdx;
      let mut prevValue = self.data[idx].clone();
      while(true) {
        idx = (self.noRows * idx) % (self.data.len() - 1);
        let currentValue = self.data[idx].clone();
        self.data[idx] = prevValue;
        if(idx == cycleIdx) {
          break;
        }

        prevValue = currentValue;
        visited[idx] = true;
      }
    }

    let rows = self.noRows;
    self.noRows = self.noCols;
    self.noCols = rows;
  }
}

impl<T : Clone> Matrix<T> {
  pub fn minor(&self, row : uint, col : uint) -> Matrix<T> {
    assert!(row < self.noRows && col < self.noCols && self.noRows > 1 && self.noCols > 1);
    let elems = (self.noCols - 1) * (self.noRows - 1);
    let mut d = alloc_dirty_vec(elems);
    let mut sourceRowIdx = 0u;
    let mut destIdx = 0u;
    for currentRow in range(0u, self.noRows - 1) {
      if currentRow != row {
        for currentCol in range(0u, self.noCols - 1) {
          if currentCol != col {
            d[destIdx] = self.data[sourceRowIdx + currentCol - (if currentCol < col { 0 } else { 1 })].clone();
            destIdx = destIdx + 1;
          }
        }
      }
      sourceRowIdx = sourceRowIdx + self.noCols;
    }
    Matrix {
      noRows : self.noRows - 1,
      noCols : self.noCols - 1,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn sub_matrix(&self, startRow : uint, startCol : uint, endRow : uint, endCol : uint) -> Matrix<T> {
    assert!(startRow < endRow);
    assert!(startCol < endCol);
    assert!(endRow < self.noRows && endCol < self.noCols && startRow != endRow && startCol != endCol);
    let rows = endRow - startRow;
    let cols = endCol - startCol;
    let elems = rows * cols;
    let mut d = alloc_dirty_vec(elems);
    let mut srcIdx = startRow * self.noCols + startCol;
    let mut destIdx = 0u;
    for _ in range(0u, rows) {
      for colOffset in range(0u, cols) {
        d[destIdx + colOffset] = self.data[srcIdx + colOffset].clone();
      }
      srcIdx += self.noCols;
      destIdx += cols;
    }
    Matrix {
      noRows : rows,
      noCols : cols,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn permute_rows(&self, rows : &[uint]) -> Matrix<T> {
    let no_rows = rows.len();
    let no_cols = self.noCols;
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    let mut destIdx = 0;
    for row in range(0u, no_rows) {
      let row_idx = rows[row] * no_cols;
      for col in range(0u, no_cols) {
        d[destIdx] = self.data[row_idx + col].clone();
        destIdx += 1;
      }
    }

    Matrix {
      noRows : no_rows,
      noCols : no_cols,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn print(&self) {
    io::print(fmt!("%10s ", ""));
    for col in range(0u, self.noCols) {
      io::print(fmt!("%10u ", col));
    }
    io::println("");
    for row in range(0u, self.noRows) {
      io::print(fmt!("%10u ", row));
      for col in range(0u, self.noCols) {
        io::print(fmt!("%10.10? ", self.get(row, col)))
      }
      io::println("")
    }
  }
}

impl<T : Neg<T>> Matrix<T> {
  pub fn neg(&self) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = - self.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl <T : Neg<T>> Neg<Matrix<T>> for Matrix<T> {
  fn neg(&self) -> Matrix<T> { self.neg() }
}

impl<T : Neg<T>> Matrix<T> {
  pub fn mneg(&mut self) {
    for i in range(0u, self.data.len()) {
      self.data[i] = - self.data[i];
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn scale(&self, factor : T) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = factor * self.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn mscale(&mut self, factor : T) {
    for i in range(0u, self.data.len()) {
      self.data[i] = factor * self.data[i];
    }
  }
}

impl<T : Add<T, T>> Matrix<T> {
  pub fn add(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = self.data[i] + m.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl <T : Add<T, T>> Add<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn add(&self, rhs: &Matrix<T>) -> Matrix<T> { self.add(rhs) }
}

impl<T : Add<T, T>> Matrix<T> {
  pub fn madd(&mut self, m : &Matrix<T>) {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    for i in range(0u, self.data.len()) {
      self.data[i] = self.data[i] + m.data[i];
    }
  }
}

impl<T : Sub<T, T>> Matrix<T> {
  pub fn sub(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = self.data[i] - m.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl <T : Sub<T, T>> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> { self.sub(rhs) }
}

impl<T : Sub<T, T>> Matrix<T> {
  pub fn msub(&mut self, m : &Matrix<T>) {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    for i in range(0u, self.data.len()) {
      self.data[i] = self.data[i] - m.data[i];
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn elem_mul(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = self.data[i] * m.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn melem_mul(&mut self, m : &Matrix<T>) {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    for i in range(0u, self.data.len()) {
      self.data[i] = self.data[i] * m.data[i];
    }
  }
}

impl<T : Div<T, T>> Matrix<T> {
  pub fn elem_div(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      d[i] = self.data[i] / m.data[i];
    }
    Matrix {
      noRows: self.noRows,
      noCols: self.noCols,
      data : d
    }
  }
}

impl<T : Div<T, T>> Matrix<T> {
  pub fn melem_div(&mut self, m : &Matrix<T>) {
    assert!(self.noRows == m.noRows);
    assert!(self.noCols == m.noCols);

    for i in range(0u, self.data.len()) {
      self.data[i] = self.data[i] / m.data[i];
    }
  }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Clone> Matrix<T> {
  pub fn mul(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.noCols == m.noRows);

    let elems = self.noRows * m.noCols;
    let mut d = alloc_dirty_vec(elems);
    for row in range(0u, self.noRows) {
      for col in range(0u, m.noCols) {
        let mut res : T = Zero::zero();
        for idx in range(0u, self.noCols) {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        d[row * m.noCols + col] = res;
      }
    }

    Matrix {
      noRows: self.noRows,
      noCols: m.noCols,
      data : d
    }
  }
}

impl<T : Mul<T, T> + Add<T, T> + Zero + Clone> Mul<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> { self.mul(rhs) }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Clone> Matrix<T> {
  pub fn mmul(&mut self, m : &Matrix<T>) {
    assert!(self.noCols == m.noRows);

    let elems = self.noRows * m.noCols;
    let mut d = alloc_dirty_vec(elems);
    for row in range(0u, self.noRows) {
      for col in range(0u, m.noCols) {
        let mut res : T = Zero::zero();
        for idx in range(0u, self.noCols) {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        d[row * m.noCols + col] = res;
      }
    }

    self.noCols = m.noCols;
    self.data = d
  }
}

impl<T : Add<T, T> + Zero> Matrix<T> {
  pub fn trace(&self) -> T {
    let mut sum : T = Zero::zero();
    let mut idx = 0;
    for _ in range(0u, num::min(self.noRows, self.noCols)) {
      sum = sum + self.data[idx];
      idx += self.noCols + 1;
    }
    sum
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic> Matrix<T> {
  pub fn det(&self) -> T {
    assert!(self.noCols == self.noRows);
    lu::LUDecomposition::new(self).det()
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic> Matrix<T> {
  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    lu::LUDecomposition::new(self).solve(b)
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic> Matrix<T> {
  pub fn inverse(&self) -> Option<Matrix<T>> {
    assert!(self.noRows == self.noCols);
    lu::LUDecomposition::new(self).solve(&id(self.noRows))
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic> Matrix<T> {
  pub fn is_nonsingular(&self) -> bool {
    assert!(self.noRows == self.noCols);
    lu::LUDecomposition::new(self).is_nonsingular()
  }
}
