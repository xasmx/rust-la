use std::cmp;
use std::fmt::{Show};
use std::io;
use std::num;
use std::num::{One, Zero};
use std::rand;
use std::rand::{Rand};
use std::vec;

use ApproxEq;
use decomp::lu;
use decomp::qr;
use internalutil::{alloc_dirty_vec};

#[deriving(PartialEq, Clone)]
pub struct Matrix<T> {
  no_rows : uint,
  pub data : Vec<T>
}

impl<T : ApproxEq<T>>  Matrix<T> {
  pub fn approx_eq(&self, m : &Matrix<T>) -> bool {
    if self.rows() != m.rows() || self.cols() != m.cols() { return false };
    for i in range(0u, self.data.len()) {
      if !self.data.get(i).approx_eq(m.data.get(i)) { return false }
    }
    true
  }
}

impl<T> Matrix<T> {
  pub fn new(no_rows : uint, no_cols : uint, data : Vec<T>) -> Matrix<T> {
    assert!(no_rows * no_cols == data.len());
    assert!(no_rows > 0 && no_cols > 0);
    Matrix { no_rows : no_rows, data : data }
  }

  pub fn vector(data : Vec<T>) -> Matrix<T> {
    assert!(data.len() > 0);
    Matrix { no_rows : data.len(), data : data }
  }

  pub fn row_vector(data : Vec<T>) -> Matrix<T> {
    assert!(data.len() > 0);
    Matrix { no_rows : 1, data : data }
  }

  #[inline]
  pub fn rows(&self) -> uint { self.no_rows }

  #[inline]
  pub fn cols(&self) -> uint { self.data.len() / self.no_rows }
}

impl<T : Rand> Matrix<T> {
  pub fn random(no_rows : uint, no_cols : uint) -> Matrix<T> {
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = rand::random::<T>();
    }
    Matrix { no_rows : no_rows, data : d }
  }
}

impl<T : Num> Matrix<T> {
  pub fn id(m : uint, n : uint) -> Matrix<T> {
    let elems = m * n;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = num::zero();
    }
    for i in range(0u, cmp::min(m, n)) {
      *d.get_mut(i * n + i) = num::one();
    }
    Matrix { no_rows : m, data : d }
  }

  pub fn zero(no_rows : uint, no_cols : uint) -> Matrix<T> {
    let elems = no_rows * no_cols;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = num::zero();
    }
    Matrix {
      no_rows : no_rows,
      data : d
    }
  }
}

impl<T : Zero + Clone> Matrix<T> {
  pub fn zero_vector(no_rows : uint) -> Matrix<T> {
    let mut d : Vec<T> = alloc_dirty_vec(no_rows);
    for i in range(0u, no_rows) {
      *d.get_mut(i) = num::zero();
    }
    Matrix { no_rows : no_rows, data : d }
  }
}

impl<T : One + Clone> Matrix<T> {
  pub fn one_vector(no_rows : uint) -> Matrix<T> {
    let mut d : Vec<T> = alloc_dirty_vec(no_rows);
    for i in range(0u, no_rows) {
      *d.get_mut(i) = num::one();
    }
    Matrix { no_rows : no_rows, data : d }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn get(&self, row : uint, col : uint) -> T {
    assert!(row < self.no_rows && col < self.cols());
    self.data.get(row * self.cols() + col).clone()
  }

  pub fn get_ref<'lt>(&'lt self, row : uint, col : uint) -> &'lt T {
    assert!(row < self.no_rows && col < self.cols());
    self.data.get(row * self.cols() + col)
  }

  pub fn get_mref<'lt>(&'lt mut self, row : uint, col : uint) -> &'lt mut T {
    assert!(row < self.no_rows && col < self.cols());
    let no_cols = self.cols();
    self.data.get_mut(row * no_cols + col)
  }

  pub fn set(&mut self, row : uint, col : uint, val : T) {
    assert!(row < self.no_rows && col < self.cols());
    let no_cols = self.cols();
    *self.data.get_mut(row * no_cols + col) = val.clone()
  }
}

impl<S, T> Matrix<S> {
  pub fn map(&self, f : |&S| -> T) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = f(self.data.get(i));
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T> Matrix<T> {
  pub fn mmap(&mut self, f : |&T| -> T) {
    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = f(self.data.get(i));
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn cr(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    let mut srcIdx1 = 0;
    let mut srcIdx2 = 0;
    let mut destIdx = 0;
    for _ in range(0u, self.no_rows) {
      for _ in range(0u, self.cols()) {
        *d.get_mut(destIdx) = self.data.get(srcIdx1).clone();
        srcIdx1 += 1;
        destIdx += 1;
      }
      for _ in range(0u, m.cols()) {
        *d.get_mut(destIdx) = m.data.get(srcIdx2).clone();
        srcIdx2 += 1;
        destIdx += 1;
      }
    }
    Matrix {
      no_rows : self.no_rows,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn cb(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.cols() == m.cols());
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, self.data.len()) {
      *d.get_mut(i) = self.data.get(i).clone();
    }
    let offset = self.data.len();
    for i in range(0u, m.data.len()) {
      *d.get_mut(offset + i) = m.data.get(i).clone();
    }
    Matrix {
      no_rows : self.no_rows + m.no_rows,
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
      *d.get_mut(i) = self.data.get(srcIdx).clone();
      srcIdx += self.cols();
      if srcIdx >= elems {
        srcIdx -= elems;
        srcIdx += 1;
      }
    }
    Matrix {
      no_rows: self.cols(),
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn mt(&mut self) {
    let mut visited = vec::Vec::from_elem(self.data.len(), false);

    for cycleIdx in range(1u, self.data.len() - 1) {
      if *visited.get(cycleIdx) {
        continue;
      }

      let mut idx = cycleIdx;
      let mut prevValue = self.data.get(idx).clone();
      loop {
        idx = (self.no_rows * idx) % (self.data.len() - 1);
        let currentValue = self.data.get(idx).clone();
        *self.data.get_mut(idx) = prevValue;
        if idx == cycleIdx {
          break;
        }

        prevValue = currentValue;
        *visited.get_mut(idx) = true;
      }
    }

    self.no_rows = self.cols();
  }
}

impl<T : Clone> Matrix<T> {
  pub fn minor(&self, row : uint, col : uint) -> Matrix<T> {
    assert!(row < self.no_rows && col < self.cols() && self.no_rows > 1 && self.cols() > 1);
    let elems = (self.cols() - 1) * (self.no_rows - 1);
    let mut d = alloc_dirty_vec(elems);
    let mut sourceRowIdx = 0u;
    let mut destIdx = 0u;
    for currentRow in range(0u, self.no_rows) {
      if currentRow != row {
        for currentCol in range(0u, self.cols()) {
          if currentCol != col {
            *d.get_mut(destIdx) = self.data.get(sourceRowIdx + currentCol).clone();
            destIdx += 1;
          }
        }
      }
      sourceRowIdx = sourceRowIdx + self.cols();
    }
    Matrix {
      no_rows : self.no_rows - 1,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn sub_matrix(&self, startRow : uint, startCol : uint, endRow : uint, endCol : uint) -> Matrix<T> {
    assert!(startRow < endRow);
    assert!(startCol < endCol);
    assert!((endRow - startRow) < self.no_rows && (endCol - startCol) < self.cols() && startRow != endRow && startCol != endCol);
    let rows = endRow - startRow;
    let cols = endCol - startCol;
    let elems = rows * cols;
    let mut d = alloc_dirty_vec(elems);
    let mut srcIdx = startRow * self.cols() + startCol;
    let mut destIdx = 0u;
    for _ in range(0u, rows) {
      for colOffset in range(0u, cols) {
        *d.get_mut(destIdx + colOffset) = self.data.get(srcIdx + colOffset).clone();
      }
      srcIdx += self.cols();
      destIdx += cols;
    }
    Matrix {
      no_rows : rows,
      data : d
    }
  }

  pub fn get_column(&self, column : uint) -> Matrix<T> {
    assert!(column < self.cols());
    let mut d = alloc_dirty_vec(self.no_rows);
    let mut src_idx = column;
    for i in range(0, self.no_rows) {
      *d.get_mut(i) = self.data.get(src_idx).clone();
      src_idx += self.cols();
    }
    Matrix {
      no_rows : self.no_rows,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn permute_rows(&self, rows : &Vec<uint>) -> Matrix<T> {
    let no_rows = rows.len();
    let no_cols = self.cols();
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    let mut destIdx = 0;
    for row in range(0u, no_rows) {
      let row_idx = *rows.get(row) * no_cols;
      for col in range(0u, no_cols) {
        *d.get_mut(destIdx) = self.data.get(row_idx + col).clone();
        destIdx += 1;
      }
    }

    Matrix {
      no_rows : no_rows,
      data : d
    }
  }

  pub fn permute_columns(&self, columns : &Vec<uint>) -> Matrix<T> {
    let no_rows = self.no_rows;
    let no_cols = columns.len();
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    let mut destIdx = 0;
    let mut row_idx = 0;
    for _ in range(0u, no_rows) {
      for col in range(0u, no_cols) {
        *d.get_mut(destIdx) = self.data.get(row_idx + *columns.get(col)).clone();
        destIdx += 1;
      }
      row_idx += self.cols();
    }

    Matrix {
      no_rows : no_rows,
      data : d
    }
  }
}

impl<T : Clone> Matrix<T> {
  pub fn filter_rows(&self, f : |m : &Matrix<T>, row : uint| -> bool) -> Matrix<T> {
    let mut rows = vec::Vec::with_capacity(self.rows());
    for row in range(0u, self.rows()) {
      if f(self, row) {
        rows.push(row);
      }
    }
    self.permute_rows(&rows)
  }

  pub fn filter_columns(&self, f : |m : &Matrix<T>, col : uint| -> bool) -> Matrix<T> {
    let mut cols = vec::Vec::with_capacity(self.cols());
    for col in range(0u, self.cols()) {
      if f(self, col) {
        cols.push(col);
      }
    }
    self.permute_columns(&cols)
  }

  pub fn select_rows(&self, selector : &[bool]) -> Matrix<T> {
    assert!(self.no_rows == selector.len());
    let mut rows = vec::Vec::with_capacity(self.no_rows);
    for i in range(0, selector.len()) {
      if selector[i] {
        rows.push(i);
      }
    }
    self.permute_rows(&rows)
  }

  pub fn select_columns(&self, selector : &[bool]) -> Matrix<T> {
    assert!(self.cols() == selector.len());
    let mut cols = vec::Vec::with_capacity(self.cols());
    for i in range(0, selector.len()) {
      if selector[i] {
        cols.push(i);
      }
    }
    self.permute_columns(&cols)
  }
}

impl<T : Clone + Show> Matrix<T> {
  pub fn print(&self) {
    print!("{:10s} ", "");
    for col in range(0u, self.cols()) {
      print!("{:10u} ", col);
    }
    println!("");
    for row in range(0u, self.no_rows) {
      print!("{:10u} ", row);
      for col in range(0u, self.cols()) {
        print!("{:10.10}? ", self.get(row, col))
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
      *d.get_mut(i) = - *self.data.get(i)
    }
    Matrix {
      no_rows: self.no_rows,
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
      *self.data.get_mut(i) = - *self.data.get(i);
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn scale(&self, factor : T) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = factor * *self.data.get(i);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn mscale(&mut self, factor : T) {
    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = factor * *self.data.get(i);
    }
  }
}

impl<T : Add<T, T>> Matrix<T> {
  pub fn add(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = *self.data.get(i) + *m.data.get(i);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl <T : Add<T, T>> Add<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn add(&self, rhs: &Matrix<T>) -> Matrix<T> { self.add(rhs) }
}

impl<T : Add<T, T>> Matrix<T> {
  pub fn madd(&mut self, m : &Matrix<T>) {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = *self.data.get(i) + *m.data.get(i);
    }
  }
}

impl<T : Sub<T, T>> Matrix<T> {
  pub fn sub(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = *self.data.get(i) - *m.data.get(i);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl <T : Sub<T, T>> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> { self.sub(rhs) }
}

impl<T : Sub<T, T>> Matrix<T> {
  pub fn msub(&mut self, m : &Matrix<T>) {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = *self.data.get(i) - *m.data.get(i);
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn elem_mul(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = *self.data.get(i) * *m.data.get(i);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T : Mul<T, T>> Matrix<T> {
  pub fn melem_mul(&mut self, m : &Matrix<T>) {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = *self.data.get(i) * *m.data.get(i);
    }
  }
}

impl<T : Div<T, T>> Matrix<T> {
  pub fn elem_div(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in range(0u, elems) {
      *d.get_mut(i) = *self.data.get(i) / *m.data.get(i);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T : Div<T, T>> Matrix<T> {
  pub fn melem_div(&mut self, m : &Matrix<T>) {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in range(0u, self.data.len()) {
      *self.data.get_mut(i) = *self.data.get(i) / *m.data.get(i);
    }
  }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Clone> Matrix<T> {
  pub fn mul(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.cols() == m.no_rows);

    let elems = self.no_rows * m.cols();
    let mut d = alloc_dirty_vec(elems);
    for row in range(0u, self.no_rows) {
      for col in range(0u, m.cols()) {
        let mut res : T = num::zero();
        for idx in range(0u, self.cols()) {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        *d.get_mut(row * m.cols() + col) = res;
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T : Mul<T, T> + Add<T, T> + Zero + Clone> Mul<Matrix<T>, Matrix<T>> for Matrix<T> {
  fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> { self.mul(rhs) }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Clone> Matrix<T> {
  pub fn mmul(&mut self, m : &Matrix<T>) {
    assert!(self.cols() == m.no_rows);

    let elems = self.no_rows * m.cols();
    let mut d = alloc_dirty_vec(elems);
    for row in range(0u, self.no_rows) {
      for col in range(0u, m.cols()) {
        let mut res : T = num::zero();
        for idx in range(0u, self.cols()) {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        *d.get_mut(row * m.cols() + col) = res;
      }
    }

    self.data = d
  }
}

impl<T : Add<T, T> + Zero> Matrix<T> {
  pub fn trace(&self) -> T {
    let mut sum : T = num::zero();
    let mut idx = 0;
    for _ in range(0u, cmp::min(self.no_rows, self.cols())) {
      sum = sum + *self.data.get(idx);
      idx += self.cols() + 1;
    }
    sum
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed> Matrix<T> {
  pub fn det(&self) -> T {
    assert!(self.cols() == self.no_rows);
    lu::LUDecomposition::new(self).det()
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed> Matrix<T> {
  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    lu::LUDecomposition::new(self).solve(b)
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed> Matrix<T> {
  pub fn inverse(&self) -> Option<Matrix<T>> {
    assert!(self.no_rows == self.cols());
    lu::LUDecomposition::new(self).solve(&Matrix::id(self.no_rows, self.no_rows))
  }
}

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed + Float> Matrix<T> {
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

impl<T : Add<T, T> + Sub<T, T> + Mul<T, T> + Div<T, T> + Neg<T> + ApproxEq<T> + PartialOrd + One + Zero + Clone + Signed> Matrix<T> {
  #[inline]
  pub fn is_singular(&self) -> bool {
    !self.is_non_singular()
  }

  pub fn is_non_singular(&self) -> bool {
    assert!(self.no_rows == self.cols());
    lu::LUDecomposition::new(self).is_non_singular()
  }
}

impl<T> Matrix<T> {
  #[inline]
  pub fn is_square(&self) -> bool {
    self.no_rows == self.cols()
  }

  #[inline]
  pub fn is_not_square(&self) -> bool {
    !self.is_square()
  }
}

impl<T : PartialEq + Clone> Matrix<T> {
  pub fn is_symmetric(&self) -> bool {
    if self.no_rows != self.cols() { return false; }
    for row in range(1, self.no_rows) {
      for col in range(0, row) {
        if self.get(row, col) != self.get(col, row) { return false; }
      }
    }

    true
  }

  #[inline]
  pub fn is_non_symmetric(&self) -> bool {
    !self.is_symmetric()
  }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Float> Matrix<T> {
  pub fn vector_euclidean_norm(&self) -> T {
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + *self.data.get(i) * *self.data.get(i);
    }

    s.sqrt()
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
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + num::abs(self.data.get(i).clone());
    }

    s
  }
}

impl<T : Add<T, T> + Div<T, T> + Signed + Zero + Clone + Float> Matrix<T> {
  pub fn vector_p_norm(&self, p : T) -> T {
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + num::abs(self.data.get(i).powf(p.clone()));
    }

    s.powf(num::one::<T>() / p)
  }
}

impl<T : Signed + PartialOrd + Clone> Matrix<T> {
  pub fn vector_inf_norm(&self) -> T {
    assert!(self.cols() == 1);

    let mut current_max : T = num::abs(self.data.get(0).clone());
    for i in range(1, self.data.len()) {
      let v = num::abs(self.data.get(i).clone());
      if v > current_max {
        current_max = v;
      }
    }

    current_max
  }
}

impl<T : Add<T, T> + Mul<T, T> + Zero + Float> Matrix<T> {
  pub fn frobenius_norm(&self) -> T {
    let mut s : T = num::zero();
    for i in range(0, self.data.len()) {
      s = s + *self.data.get(i) * *self.data.get(i);
    }

    s.sqrt()
  }
}

impl <S : Clone, T> Matrix<T> {
  pub fn reduce(&self, init: &Vec<S>, f: |&S, &T| -> S) -> Matrix<S> {
    assert!(init.len() == self.cols());

    let mut data = init.clone();
    let mut dataIdx = 0;
    for i in range(0, self.data.len()) {
      *data.get_mut(dataIdx) = f(data.get(dataIdx), self.data.get(i));
      dataIdx += 1;
      dataIdx %= data.len();
    }

    Matrix {
      no_rows : 1,
      data : data
    }
  }
}

#[test]
fn test_new() {
  let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
  m.print();
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 2, 3, 4]);
}

#[test]
#[should_fail]
fn test_new_invalid_data() {
  Matrix::new(1, 2, vec![1, 2, 3]);
}

#[test]
#[should_fail]
fn test_new_invalid_row_count() {
  Matrix::<uint>::new(0, 2, vec![]);
}

#[test]
#[should_fail]
fn test_new_invalid_col_count() {
  Matrix::<uint>::new(2, 0, vec![]);
}

#[test]
fn test_id_square() {
  let m = Matrix::<uint>::id(2, 2);
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 1]);
}

#[test]
fn test_id_m_over_n() {
  let m = Matrix::<uint>::id(3, 2);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 1, 0, 0]);
}

#[test]
fn test_id_n_over_m() {
  let m = Matrix::<uint>::id(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == vec![1, 0, 0, 0, 1, 0]);
}

#[test]
fn test_zero() {
  let m = Matrix::<uint>::zero(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_vector() {
  let v = Matrix::vector(vec![1, 2, 3]);
  assert!(v.rows() == 3);
  assert!(v.cols() == 1);
  assert!(v.data == vec![1, 2, 3]);
}

#[test]
fn test_zero_vector() {
  let v = Matrix::<uint>::zero_vector(2);
  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == vec![0, 0]);
}

#[test]
fn test_one_vector() {
  let v = Matrix::<uint>::one_vector(2);

  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == vec![1, 1]);
}

#[test]
fn test_row_vector() {
  let v = Matrix::row_vector(vec![1, 2, 3]);
  assert!(v.rows() == 1);
  assert!(v.cols() == 3);
  assert!(v.data == vec![1, 2, 3]);
}

#[test]
fn test_get_set() {
  let mut m = m!(1, 2; 3, 4);
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
fn test_get_out_of_bounds_x() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get(2, 0);
}

#[test]
#[should_fail]
fn test_get_out_of_bounds_y() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get(0, 2);
}

#[test]
#[should_fail]
fn test_get_ref_out_of_bounds_x() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get_ref(2, 0);
}

#[test]
#[should_fail]
fn test_get_ref_out_of_bounds_y() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get_ref(0, 2);
}

#[test]
#[should_fail]
fn test_get_mref_out_of_bounds_x() {
  let mut m = m!(1, 2; 3, 4);
  let _ = m.get_mref(2, 0);
}

#[test]
#[should_fail]
fn test_get_mref_out_of_bounds_y() {
  let mut m = m!(1, 2; 3, 4);
  let _ = m.get_mref(0, 2);
}

#[test]
#[should_fail]
fn test_set_out_of_bounds_x() {
  let mut m = m!(1, 2; 3, 4);
  m.set(2, 0, 0);
}

#[test]
#[should_fail]
fn test_set_out_of_bounds_y() {
  let mut m = m!(1, 2; 3, 4);
  m.set(0, 2, 0);
}

#[test]
fn test_map() {
  let mut m = m!(1, 2; 3, 4);
  assert!(m.map(|x : &uint| -> uint { *x + 1 }).data == vec![2, 3, 4, 5]);

  m.mmap(|x : &uint| { *x + 2 });
  assert!(m.data == vec![3, 4, 5, 6]);
}

#[test]
fn test_cr() {
  let v = m!(1; 2; 3);
  let m = v.cr(&v);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_cb() {
  let m = m!(1, 2; 3, 4);
  let m2 = m.cb(&m);
  assert!(m2.rows() == 4);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![1, 2, 3, 4, 1, 2, 3, 4]);
}

#[test]
fn test_t() {
  let mut m = m!(1, 2; 3, 4);
  assert!(m.t().data == vec![1, 3, 2, 4]);

  m.mt();
  assert!(m.data == vec![1, 3, 2, 4]);

  let mut m = m!(1, 2, 3; 4, 5, 6);
  let r = m.t();
  assert!(r.rows() == 3);
  assert!(r.cols() == 2);
  assert!(r.data == vec![1, 4, 2, 5, 3, 6]);

  m.mt();
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_sub() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.minor(1, 1).data == vec![1, 3, 7, 9]);
  assert!(m.sub_matrix(1, 1, 3, 3).data == vec![5, 6, 8, 9]);
  assert!(m.get_column(1).data == vec![2, 5, 8]);
}

#[test]
#[should_fail]
fn test_minor_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.minor(1, 4);
}

#[test]
#[should_fail]
fn test_sub_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.sub_matrix(1, 1, 3, 4);
}

#[test]
#[should_fail]
fn test_get_column_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.get_column(3);
}

#[test]
fn test_permute_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.permute_rows(&vec![1, 0, 2]).data == vec![4, 5, 6, 1, 2, 3, 7, 8, 9]);
  assert!(m.permute_rows(&vec![2, 1]).data == vec![7, 8, 9, 4, 5, 6]);
}

#[test]
#[should_fail]
fn test_permute_rows_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.permute_rows(&vec![1, 0, 5]);
}

#[test]
fn test_permute_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.permute_columns(&vec![1, 0, 2]).data == vec![2, 1, 3, 5, 4, 6, 8, 7, 9]);
  assert!(m.permute_columns(&vec![1, 2]).data == vec![2, 3, 5, 6, 8, 9]);
}

#[test]
#[should_fail]
fn test_permute_columns_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.permute_columns(&vec![1, 0, 5]);
}

#[test]
fn test_filter_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.filter_rows(|_, row| { ((row % 2) == 0) });
  assert!(m2.rows() == 2);
  assert!(m2.cols() == 3);
  assert!(m2.data == vec![1, 2, 3, 7, 8, 9]); 
}

#[test]
fn test_filter_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.filter_columns(|_, col| { (col >= 1) });
  m2.print();
  assert!(m2.rows() == 3);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![2, 3, 5, 6, 8, 9]); 
}

#[test]
fn test_select_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.select_rows([false, true, true]);
  assert!(m2.rows() == 2);
  assert!(m2.cols() == 3);
  assert!(m2.data == vec![4, 5, 6, 7, 8, 9]); 
}

#[test]
fn test_select_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.select_columns([true, false, true]);
  assert!(m2.rows() == 3);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![1, 3, 4, 6, 7, 9]); 
}

#[test]
fn test_algebra() {
  let a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  assert!(a.neg().data == vec![-1, -2, -3, -4]);
  assert!(a.scale(2).data == vec![2, 4, 6, 8]);
  assert!(a.add(&b).data == vec![4, 6, 8, 10]);
  assert!(b.sub(&a).data == vec![2, 2, 2, 2]);
  assert!(a.elem_mul(&b).data == vec![3, 8, 15, 24]);
  assert!(b.elem_div(&a).data == vec![3, 2, 1, 1]);

  let mut a = m!(1, 2; 3, 4);
  a.mneg();
  assert!(a.data == vec![-1, -2, -3, -4]);

  let mut a = m!(1, 2; 3, 4);
  a.mscale(2);
  assert!(a.data == vec![2, 4, 6, 8]);

  let mut a = m!(1, 2; 3, 4);
  a.madd(&b);
  assert!(a.data == vec![4, 6, 8, 10]);

  let a = m!(1, 2; 3, 4);
  let mut b = m!(3, 4; 5, 6);
  b.msub(&a);
  assert!(b.data == vec![2, 2, 2, 2]);

  let mut a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  a.melem_mul(&b);
  assert!(a.data == vec![3, 8, 15, 24]);

  let a = m!(1, 2; 3, 4);
  let mut b = m!(3, 4; 5, 6);
  b.melem_div(&a);
  assert!(b.data == vec![3, 2, 1, 1]);
}

#[test]
fn test_mul() {
  let mut a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  assert!(a.mul(&b).data == vec![13, 16, 29, 36]);
  a.mmul(&b);
  assert!(a.data == vec![13, 16, 29, 36]);
}

#[test]
#[should_fail]
fn test_mul_incompatible() {
  let a = m!(1, 2; 3, 4);
  let b = m!(1, 2; 3, 4; 5, 6);
  let _ = a.mul(&b);
}

#[test]
#[should_fail]
fn test_mmul_incompatible() {
  let mut a = m!(1, 2; 3, 4);
  let b = m!(1, 2; 3, 4; 5, 6);
  a.mmul(&b);
}

#[test]
fn test_trace() {
  let a = m!(1, 2; 3, 4);
  assert!(a.trace() == 5);

  let a = m!(1, 2; 3, 4; 5, 6);
  assert!(a.trace() == 5);

  let a = m!(1, 2, 3; 4, 5, 6);
  assert!(a.trace() == 6);
}

#[test]
fn test_det() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0; 0.0, 5.0, -7.0);
  assert!((a.det() - -96.0) <= Float::epsilon());
}

#[test]
#[should_fail]
fn test_det_not_square() {
  let _ = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0).det();
}

#[test]
fn test_solve() {
  let a = m!(1.0, 1.0, 1.0; 1.0, -1.0, 4.0; 2.0, 3.0, -5.0);
  let b = m!(3.0; 4.0; 0.0);
  assert!(a.solve(&b).unwrap().eq(&m!(1.0; 1.0; 1.0)));
}

// TODO: Add more tests for solve

#[test]
fn test_inverse() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0; 0.0, 5.0, -7.0);
  let data : Vec<f64> = vec![16.0, -1.0, 23.0, 0.0, 42.0, -6.0, 0.0, 30.0, -18.0].mut_iter().map(|x : &mut f64| -> f64 { *x / 96.0 }).collect();
  let a_inv = Matrix::new(3, 3, data);
  assert!(a.inverse().unwrap().approx_eq(&a_inv));
}

#[test]
#[should_fail]
fn test_inverse_not_square() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0);
  let _ = a.inverse();
}

#[test]
fn test_inverse_singular() {
  let a = m!(2.0, 6.0; 1.0, 3.0);
  assert!(a.inverse() == None);
}

#[test]
fn test_pinverse() {
  let a = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0);
  assert!((a.pinverse() * a).approx_eq(&Matrix::<f64>::id(2, 2)));
}

#[test]
fn test_is_singular() {
  let m = m!(2.0, 6.0; 1.0, 3.0);
  assert!(m.is_singular());
}

#[test]
#[should_fail]
fn test_is_singular_non_square() {
  let m = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0);
  assert!(m.is_singular());
}

#[test]
fn test_is_non_singular() {
  let m = m!(2.0, 6.0; 6.0, 3.0);
  assert!(m.is_non_singular());
}

#[test]
fn test_is_square() {
  let m = m!(1, 2; 3, 4);
  assert!(m.is_square());
  assert!(!m.is_not_square());

  let m = m!(1, 2, 3; 4, 5, 6);
  assert!(!m.is_square());
  assert!(m.is_not_square());

  let v = m!(1; 2; 3);
  assert!(!v.is_square());
  assert!(v.is_not_square());
}

#[test]
fn test_is_symmetric() {
  let m = m!(1, 2, 3; 2, 4, 5; 3, 5, 6);
  assert!(m.is_symmetric());

  let m = m!(1, 2; 3, 4);
  assert!(!m.is_symmetric());

  let m = m!(1, 2, 3; 2, 4, 5);
  assert!(!m.is_symmetric());
}

#[test]
fn test_vector_euclidean_norm() {
  assert!(m!(1.0; 2.0; 2.0).vector_euclidean_norm() == 3.0);
  assert!(m!(-2.0; 2.0; 2.0; 2.0).vector_euclidean_norm() == 4.0);
}

#[test]
#[should_fail]
fn test_vector_euclidean_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_euclidean_norm();
}

#[test]
fn test_vector_1_norm() {
  assert!(m!(-3.0; 2.0; 2.5).vector_1_norm() == 7.5);
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_1_norm() == 19.0);
  assert!(m!(1.0).vector_1_norm() == 1.0);
}

#[test]
#[should_fail]
fn test_vector_1_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_1_norm();
}

#[test]
fn test_vector_p_norm() {
  assert!(m!(-3.0; 2.0; 2.0).vector_p_norm(3.0) == 43.0f64.powf(1.0 / 3.0));
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_p_norm(5.0) == 40819.0f64.powf(1.0 / 5.0));
  assert!(m!(1.0).vector_p_norm(2.0) == 1.0);
}

#[test]
#[should_fail]
fn test_vector_p_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_p_norm(1.0);
}

#[test]
fn test_vector_inf_norm() {
  assert!(m!(-3.0; 2.0; 2.5).vector_inf_norm() == 3.0);
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_inf_norm() == 8.0);
  assert!(m!(1.0).vector_inf_norm() == 1.0);
}

#[test]
#[should_fail]
fn test_vector_inf_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_inf_norm();
}

#[test]
fn test_frobenius_norm() {
  assert!(m!(1.0, 2.0; 3.0, 4.0).frobenius_norm() == 30.0f64.sqrt());
  assert!(m!(1.0; 2.0; 2.0).frobenius_norm() == 3.0);
}

