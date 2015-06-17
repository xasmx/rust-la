use std::fs;
use std::io;
use std::io::BufRead;
use std::path;
use std::vec;

use Matrix;

pub fn read_csv<T>(buf_reader_name : &str, parser : &Fn(&str) -> T) -> Matrix<T> {
  let mut data = vec::Vec::with_capacity(16384);
  let mut row_count = 0;
  let mut col_count = None;

  let path = path::Path::new(buf_reader_name);
  let buf_reader = io::BufReader::new(fs::File::open(&path).unwrap());
  for line in buf_reader.lines() {
    let element_count = data.len();
    for item in line.unwrap().split(",") {
      data.push(parser(item.trim()))
    }
    let line_col_count = data.len() - element_count;

    if col_count == None {
      col_count = Some(line_col_count);
    } else {
      assert!(col_count.unwrap() == line_col_count);
    }

    row_count += 1;
  }

  assert!(col_count != None);

  Matrix::new(row_count, col_count.unwrap(), data)
}
