extern crate la;

use la::gpu::GpuContext;

fn main() {
  let ctx = GpuContext::new();

  let a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
  let b = vec![-7isize, -6, 5, -4, 0, -1, 2, 3];
  let _ = ctx.add(a, b);
  println!("Hello, World!");
}

