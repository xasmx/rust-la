#![macro_use]

/// Helper macro for m!
#[macro_export]
macro_rules! m_one {
  ( $item:tt ) => ( 1 )
}

/// Helper macro for m!
#[macro_export]
macro_rules! m_rec(
  ( [ $($row:tt),* ] [$($i:expr),*]) => ({
     let _rows = 0 $(+ m_one!($row) )*;
     let _cols = (0 $(+ m_one!($i))*) / _rows;
     Matrix::new(
       _rows,
       _cols,
       vec![$($i),*]
     )
  })
);

/// Macro for building matrices. Use commas to separate columns, and semicolons
/// to separate rows.
///
/// # Example
///
/// ```
/// let a = m!(1, 2, 3; 4, 5, 6);
/// println!("{:?}", a);
/// // ->
/// // | 1 2 3 |
/// // | 4 5 6 |
/// ```
#[macro_export]
macro_rules! m {
  ( $( $( $i:expr ),* );* ) => ( m_rec!([$([$($i),*]),*] [$($($i),*),*]) )
}
