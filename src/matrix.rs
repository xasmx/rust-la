use std::io;
use std::num;
use std::num::{One, Zero};
use std::rand;
use std::rand::{Rand};
use std::vec;

use decomp::lu;
use decomp::qr;
use util::{alloc_dirty_vec};

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

pub fn id<T : One + Zero + Clone>(m : uint, n : uint) -> Matrix<T> {
  let mut d = vec::from_elem(m * n, num::zero());
  for i in range(0u, num::min(m, n)) {
    d[i * n + i] = num::one();
  }
  Matrix { noRows : m, noCols : n, data : d }
}

pub fn zero<T : Zero + Clone>(noRows : uint, noCols : uint) -> Matrix<T> {
  Matrix {
    noRows : noRows,
    noCols : noCols,
    data : vec::from_elem(noRows * noCols, num::zero())
  }
}

pub fn vector<T>(data : ~[T]) -> Matrix<T> {
  assert!(data.len() > 0);
  Matrix { noRows : data.len(), noCols : 1, data : data }
}

pub fn zero_vector<T : Zero + Clone>(noRows : uint) -> Matrix<T> {
  Matrix { noRows : noRows, noCols : 1, data : vec::from_elem(noRows, num::zero()) }
}

pub fn one_vector<T : One + Clone>(noRows : uint) -> Matrix<T> {
  Matrix { noRows : noRows, noCols : 1, data : vec::from_elem(noRows, num::one()) }
}

pub fn row_vector<T>(data : ~[T]) -> Matrix<T> {
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
  pub fn get(&self, row : uint, col : uint) -> T {
    assert!(row < self.noRows && col < self.noCols);
    self.data[row * self.noCols + col].clone()
  }
}

impl<T : Clone> Matrix<T> {
  pub fn get_ref<'lt>(&'lt self, row : uint, col : uint) -> &'lt T {
    assert!(row < self.noRows && col < self.noCols);
    &self.data[row * self.noCols + col]
  }
}

impl<T : Clone> Matrix<T> {
  pub fn get_mref<'lt>(&'lt mut self, row : uint, col : uint) -> &'lt mut T {
    assert!(row < self.noRows && col < self.noCols);
    &mut self.data[row * self.noCols + col]
  }
}

impl<T : Clone> Matrix<T> {
  pub fn set(&mut self, row : uint, col : uint, val : T) {
    assert!(row < self.noRows && col < self.noCols);
    self.data[row * self.noCols + col] = val.clone()
  }
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
    for currentRow in range(0u, self.noRows) {
      if currentRow != row {
        for currentCol in range(0u, self.noCols) {
          if currentCol != col {
            d[destIdx] = self.data[sourceRowIdx + currentCol].clone();
            destIdx += 1;
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
    assert!((endRow - startRow) < self.noRows && (endCol - startCol) < self.noCols && startRow != endRow && startCol != endCol);
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

  pub fn get_column(&self, column : uint) -> Matrix<T> {
    assert!(column < self.noCols);
    let mut d = alloc_dirty_vec(self.noRows);
    let mut src_idx = column;
    for i in range(0, self.noRows) {
      d[i] = self.data[src_idx].clone();
      src_idx += self.noCols;
    }
    Matrix {
      noRows : self.noRows,
      noCols : 1,
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
        let mut res : T = num::zero();
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
        let mut res : T = num::zero();
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
    let mut sum : T = num::zero();
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
    lu::LUDecomposition::new(self).solve(&id(self.noRows, self.noRows))
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic + Orderable> Matrix<T> {
  pub fn pinverse(&self) -> Matrix<T> {
    // A+ = (A' A)^-1 A'
    //    = ((QR)' QR)^-1 A'
    //    = (R'Q'QR)^-1 A'
    //    = (R'R)^-1 A'
    let qr = qr::QRDecomposition::new(self);
    let r = qr.get_r();
    (r.t() * r).inverse().unwrap() * self.t()
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + Eq + Ord + ApproxEq<T> + One + Zero + Clone + Signed + Algebraic> Matrix<T> {
  #[inline]
  pub fn is_singular(&self) -> bool {
    !self.is_non_singular()
  }

  pub fn is_non_singular(&self) -> bool {
    assert!(self.noRows == self.noCols);
    lu::LUDecomposition::new(self).is_non_singular()
  }
}

impl<T> Matrix<T> {
  #[inline]
  pub fn is_square(&self) -> bool {
    self.noRows == self.noCols
  }

  #[inline]
  pub fn is_not_square(&self) -> bool {
    !self.is_square()
  }
}

impl<T : Eq + Clone> Matrix<T> {
  pub fn is_symmetric(&self) -> bool {
    if(self.noRows != self.noCols) { return false; }
    for row in range(1, self.noRows) {
      for col in range(0, row) {
        if(self.get(row, col) != self.get(col, row)) { return false; }
      }
    }

    true
  }

  #[inline]
  pub fn is_non_symmetric(&self) -> bool {
    !self.is_symmetric()
  }
}

impl<T : Add<T, T> + Mul<T, T> + Algebraic + Zero> Matrix<T> {
  pub fn vector_euclidean_norm(&self) -> T {
    assert!(self.noCols == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + self.data[i] * self.data[i];
    }

    num::sqrt(s)
  }

  #[inline]
  pub fn length(&self) -> T {
    self.vector_euclidean_norm()
  }

  #[inline]
  pub fn vector_2_norm(&self) -> T {
    self.vector_euclidean_norm()
  }
}

impl<T : Add<T, T> + Signed + Zero + Clone> Matrix<T> {
  pub fn vector_1_norm(&self) -> T {
    assert!(self.noCols == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + num::abs(self.data[i].clone());
    }

    s
  }
}

impl<T : Add<T, T> + Div<T, T> + Algebraic + Signed + Zero + Clone> Matrix<T> {
  pub fn vector_p_norm(&self, p : T) -> T {
    assert!(self.noCols == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + num::abs(num::pow(self.data[i].clone(), p.clone()));
    }

    num::pow(s, num::one::<T>() / p)
  }
}

impl<T : Signed + Ord + Clone> Matrix<T> {
  pub fn vector_inf_norm(&self) -> T {
    assert!(self.noCols == 1);

    let mut current_max : T = num::abs(self.data[0].clone());
    for i in range(1, self.data.len()) {
      let v = num::abs(self.data[i].clone());
      if(v > current_max) {
        current_max = v;
      }
    }

    current_max
  }
}

impl<T : Add<T, T> + Mul<T, T> + Algebraic + Zero> Matrix<T> {
  pub fn frobenius_norm(&self) -> T {
    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + self.data[i] * self.data[i];
    }

    num::sqrt(s)
  }
}

impl <S : Clone, T> Matrix<T> {
  pub fn reduce(&self, init: &[S], f: &fn(&S, &T) -> S) -> Matrix<S> {
    assert!(init.len() == self.noCols);

    let mut data = init.to_owned();
    let mut dataIdx = 0;
    for i in range(0, self.data.len()) {
      data[dataIdx] = f(&data[dataIdx], &self.data[i]);
      dataIdx += 1;
      dataIdx %= data.len();
    }

    Matrix {
      noRows : 1,
      noCols : self.noCols,
      data : data
    }
  }
}

#[test]
fn test_matrix() {
  let m = matrix(2, 2, ~[1, 2, 3, 4]);
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == ~[1, 2, 3, 4]);
}

#[test]
#[should_fail]
fn test_matrix__invalid_data() {
  matrix(1, 2, ~[1, 2, 3]);
}

#[test]
#[should_fail]
fn test_matrix__invalid_row_count() {
  matrix::<uint>(0, 2, ~[]);
}

#[test]
#[should_fail]
fn test_matrix__invalid_col_count() {
  matrix::<uint>(2, 0, ~[]);
}

#[test]
fn test_id__square() {
  let m = id::<uint>(2, 2);
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == ~[1, 0, 0, 1]);
}

#[test]
fn test_id__m_over_n() {
  let m = id::<uint>(3, 2);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == ~[1, 0, 0, 1, 0, 0]);
}

#[test]
fn test_id__n_over_m() {
  let m = id::<uint>(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == ~[1, 0, 0, 0, 1, 0]);
}

#[test]
fn test_zero() {
  let m = zero::<uint>(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == ~[0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_vector() {
  let v = vector::<uint>(~[1, 2, 3]);
  assert!(v.rows() == 3);
  assert!(v.cols() == 1);
  assert!(v.data == ~[1, 2, 3]);
}

#[test]
fn test_zero_vector() {
  let v = zero_vector::<uint>(2);
  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == ~[0, 0]);
}

#[test]
fn test_one_vector() {
  let v = one_vector::<uint>(2);

  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == ~[1, 1]);
}

#[test]
fn test_row_vector() {
  let v = row_vector::<uint>(~[1, 2, 3]);
  assert!(v.rows() == 1);
  assert!(v.cols() == 3);
  assert!(v.data == ~[1, 2, 3]);
}

#[test]
fn test_get_set() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  assert!(m.get(1, 0) == 3);
  assert!(m.get(0, 1) == 2);

  assert!(*m.get_ref(1, 1) == 4);

  *m.get_mref(0, 0) = 10;
  assert!(m.get(0, 0) == 10);

  m.set(1, 1, 5);
  assert!(m.get(1, 1) == 5);
}

#[test]
#[should_fail]
fn test_get__out_of_bounds_x() {
  let m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get(2, 0);
}

#[test]
#[should_fail]
fn test_get__out_of_bounds_y() {
  let m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get(0, 2);
}

#[test]
#[should_fail]
fn test_get_ref__out_of_bounds_x() {
  let m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get_ref(2, 0);
}

#[test]
#[should_fail]
fn test_get_ref__out_of_bounds_y() {
  let m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get_ref(0, 2);
}

#[test]
#[should_fail]
fn test_get_mref__out_of_bounds_x() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get_mref(2, 0);
}

#[test]
#[should_fail]
fn test_get_mref__out_of_bounds_y() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  let _ = m.get_mref(0, 2);
}

#[test]
#[should_fail]
fn test_set__out_of_bounds_x() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  m.set(2, 0, 0);
}

#[test]
#[should_fail]
fn test_set__out_of_bounds_y() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  m.set(0, 2, 0);
}

#[test]
fn test_map() {
  let mut m = matrix::<uint>(2, 2, ~[1, 2, 3, 4]);
  assert!(m.map(|x : &uint| -> uint { *x + 1 }).data == ~[2, 3, 4, 5]);

  m.mmap(|x : &uint| { *x + 2 });
  assert!(m.data == ~[3, 4, 5, 6]);
}

#[test]
fn test_cr() {
  let v = vector(~[1, 2, 3]);
  let m = v.cr(&v);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == ~[1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_cb() {
  let m = matrix(2, 2, ~[1, 2, 3, 4]);
  let m2 = m.cb(&m);
  assert!(m2.rows() == 4);
  assert!(m2.cols() == 2);
  assert!(m2.data == ~[1, 2, 3, 4, 1, 2, 3, 4]);
}

#[test]
fn test_t() {
  let mut m = matrix(2, 2, ~[1, 2, 3, 4]);
  assert!(m.t().data == ~[1, 3, 2, 4]);

  m.mt();
  assert!(m.data == ~[1, 3, 2, 4]);

  let mut m = matrix(2, 3, ~[1, 2, 3, 4, 5, 6]);
  let r = m.t();
  assert!(r.rows() == 3);
  assert!(r.cols() == 2);
  assert!(r.data == ~[1, 4, 2, 5, 3, 6]);

  m.mt();
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == ~[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_sub() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert!(m.minor(1, 1).data == ~[1, 3, 7, 9]);
  assert!(m.sub_matrix(1, 1, 3, 3).data == ~[5, 6, 8, 9]);
  assert!(m.get_column(1).data == ~[2, 5, 8]);
}

#[test]
#[should_fail]
fn test_minor__out_of_bounds() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  let _ = m.minor(1, 4);
}

#[test]
#[should_fail]
fn test_sub__out_of_bounds() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  let _ = m.sub_matrix(1, 1, 3, 4);
}

#[test]
#[should_fail]
fn test_get_column__out_of_bounds() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  let _ = m.get_column(3);
}

#[test]
fn test_permute_rows() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert!(m.permute_rows([1, 0, 2]).data == ~[4, 5, 6, 1, 2, 3, 7, 8, 9]);
}

#[test]
#[should_fail]
fn test_permute_rows__out_of_bounds() {
  let m = matrix(3, 3, ~[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  let _ = m.permute_rows([1, 0, 5]);
}

#[test]
fn test_algebra() {
  let a = matrix(2, 2, ~[1, 2, 3, 4]);
  let b = matrix(2, 2, ~[3, 4, 5, 6]);
  assert!(a.neg().data == ~[-1, -2, -3, -4]);
  assert!(a.scale(2).data == ~[2, 4, 6, 8]);
  assert!(a.add(&b).data == ~[4, 6, 8, 10]);
  assert!(b.sub(&a).data == ~[2, 2, 2, 2]);
  assert!(a.elem_mul(&b).data == ~[3, 8, 15, 24]);
  assert!(b.elem_div(&a).data == ~[3, 2, 1, 1]);

  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  a.mneg();
  assert!(a.data == ~[-1, -2, -3, -4]);

  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  a.mscale(2);
  assert!(a.data == ~[2, 4, 6, 8]);

  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  a.madd(&b);
  assert!(a.data == ~[4, 6, 8, 10]);

  let a = matrix(2, 2, ~[1, 2, 3, 4]);
  let mut b = matrix(2, 2, ~[3, 4, 5, 6]);
  b.msub(&a);
  assert!(b.data == ~[2, 2, 2, 2]);

  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  let b = matrix(2, 2, ~[3, 4, 5, 6]);
  a.melem_mul(&b);
  assert!(a.data == ~[3, 8, 15, 24]);

  let a = matrix(2, 2, ~[1, 2, 3, 4]);
  let mut b = matrix(2, 2, ~[3, 4, 5, 6]);
  b.melem_div(&a);
  assert!(b.data == ~[3, 2, 1, 1]);
}

#[test]
fn test_mul() {
  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  let b = matrix(2, 2, ~[3, 4, 5, 6]);
  assert!(a.mul(&b).data == ~[13, 16, 29, 36]);
  a.mmul(&b);
  assert!(a.data == ~[13, 16, 29, 36]);
}

#[test]
#[should_fail]
fn test_mul__incompatible() {
  let a = matrix(2, 2, ~[1, 2, 3, 4]);
  let b = matrix(3, 2, ~[1, 2, 3, 4, 5, 6]);
  let _ = a.mul(&b);
}

#[test]
#[should_fail]
fn test_mmul__incompatible() {
  let mut a = matrix(2, 2, ~[1, 2, 3, 4]);
  let b = matrix(3, 2, ~[1, 2, 3, 4, 5, 6]);
  a.mmul(&b);
}

#[test]
fn test_trace() {
  let a = matrix(2, 2, ~[1, 2, 3, 4]);
  assert!(a.trace() == 5);

  let a = matrix(3, 2, ~[1, 2, 3, 4, 5, 6]);
  assert!(a.trace() == 5);

  let a = matrix(2, 3, ~[1, 2, 3, 4, 5, 6]);
  assert!(a.trace() == 6);
}

#[test]
fn test_det() {
  let a = matrix(3, 3, ~[6.0, -7.0, 10.0, 0.0, 3.0, -1.0, 0.0, 5.0, -7.0]);
  assert!(a.det() == -96.0);
}

#[test]
#[should_fail]
fn test_det__not_square() {
  let _ = matrix(2, 3, ~[6.0, -7.0, 10.0, 0.0, 3.0, -1.0]).det();
}

#[test]
fn test_solve() {
  let a = matrix(3, 3, ~[1.0, 1.0, 1.0, 1.0, -1.0, 4.0, 2.0, 3.0, -5.0]);
  let b = vector(~[3.0, 4.0, 0.0]);
  assert!(a.solve(&b).unwrap().approx_eq(&vector(~[1.0, 1.0, 1.0])));
}

// TODO: Add more tests for solve

#[test]
fn test_inverse() {
  let a = matrix(3, 3, ~[6.0, -7.0, 10.0, 0.0, 3.0, -1.0, 0.0, 5.0, -7.0]);
  assert!(a.inverse().unwrap().approx_eq(&matrix(3, 3, [16.0, -1.0, 23.0, 0.0, 42.0, -6.0, 0.0, 30.0, -18.0].map(|x : &float| -> float { *x / 96.0 }))));
}

#[test]
#[should_fail]
fn test_inverse__not_square() {
  let a = matrix(2, 3, ~[6.0, -7.0, 10.0, 0.0, 3.0, -1.0]);
  let _ = a.inverse();
}

#[test]
fn test_inverse__singular() {
  let a = matrix(2, 2, ~[2.0, 6.0, 1.0, 3.0]);
  assert!(a.inverse() == None);
}

#[test]
fn test_pinverse() {
  let a = matrix(3, 2, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  assert!((a.pinverse() * a).approx_eq(&id::<float>(2, 2)));
}

#[test]
fn test_is_singular() {
  let m = matrix(2, 2, ~[2.0, 6.0, 1.0, 3.0]);
  assert!(m.is_singular());
}

#[test]
#[should_fail]
fn test_is_singular__non_square() {
  let m = matrix(2, 3, ~[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  assert!(m.is_singular());
}

#[test]
fn test_is_non_singular() {
  let m = matrix(2, 2, ~[2.0, 6.0, 6.0, 3.0]);
  assert!(m.is_non_singular());
}

#[test]
fn test_is_square() {
  let m = matrix(2, 2, ~[1, 2, 3, 4]);
  assert!(m.is_square());
  assert!(!m.is_not_square());

  let m = matrix(2, 3, ~[1, 2, 3, 4, 5, 6]);
  assert!(!m.is_square());
  assert!(m.is_not_square());

  let v = vector(~[1, 2, 3]);
  assert!(!v.is_square());
  assert!(v.is_not_square());
}

#[test]
fn test_is_symmetric() {
  let m = matrix(3, 3, ~[1, 2, 3, 2, 4, 5, 3, 5, 6]);
  assert!(m.is_symmetric());

  let m = matrix(2, 2, ~[1, 2, 3, 4]);
  assert!(!m.is_symmetric());

  let m = matrix(2, 3, ~[1, 2, 3, 2, 4, 5]);
  assert!(!m.is_symmetric());
}

#[test]
fn test_vector_euclidean_norm() {
  assert!(vector(~[1.0, 2.0, 2.0]).vector_euclidean_norm() == 3.0);
  assert!(vector(~[-2.0, 2.0, 2.0, 2.0]).vector_euclidean_norm() == 4.0);
}

#[test]
#[should_fail]
fn test_vector_euclidean_norm__not_vector() {
  let _ = matrix(2, 2, ~[1.0, 2.0, 3.0, 4.0]).vector_euclidean_norm();
}

#[test]
fn test_vector_1_norm() {
  assert!(vector(~[-3.0, 2.0, 2.5]).vector_1_norm() == 7.5);
  assert!(vector(~[6.0, 8.0, -2.0, 3.0]).vector_1_norm() == 19.0);
  assert!(vector(~[1.0]).vector_1_norm() == 1.0);
}

#[test]
#[should_fail]
fn test_vector_1_norm__not_vector() {
  let _ = matrix(2, 2, ~[1.0, 2.0, 3.0, 4.0]).vector_1_norm();
}

#[test]
fn test_vector_p_norm() {
  assert!(vector(~[-3.0, 2.0, 2.0]).vector_p_norm(3.0) == num::pow(43.0, 1.0 / 3.0));
  assert!(vector(~[6.0, 8.0, -2.0, 3.0]).vector_p_norm(5.0) == num::pow(40819.0, 1.0 / 5.0));
  assert!(vector(~[1.0]).vector_p_norm(2.0) == 1.0);
}

#[test]
#[should_fail]
fn test_vector_p_norm__not_vector() {
  let _ = matrix(2, 2, ~[1.0, 2.0, 3.0, 4.0]).vector_p_norm(1.0);
}

#[test]
fn test_vector_inf_norm() {
  assert!(vector(~[-3.0, 2.0, 2.5]).vector_inf_norm() == 3.0);
  assert!(vector(~[6.0, 8.0, -2.0, 3.0]).vector_inf_norm() == 8.0);
  assert!(vector(~[1.0]).vector_inf_norm() == 1.0);
}

#[test]
#[should_fail]
fn test_vector_inf_norm__not_vector() {
  let _ = matrix(2, 2, ~[1.0, 2.0, 3.0, 4.0]).vector_inf_norm();
}

#[test]
fn test_frobenius_norm() {
  assert!(matrix(2, 2, ~[1.0, 2.0, 3.0, 4.0]).frobenius_norm() == num::sqrt(30.0));
  assert!(vector(~[1.0, 2.0, 2.0]).frobenius_norm() == 3.0);
}

