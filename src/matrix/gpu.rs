use opencl;
use opencl::mem::CLBuffer;
use opencl::hl::{CommandQueue, Context, Device, Kernel};

pub struct GpuContext {
  device : Device,
  context : Context,
  command_queue : CommandQueue,
  add_kernel : Kernel
}

pub struct GpuReadOnlyMatrix {
  buffer : CLBuffer<isize>
}

pub struct GpuWriteOnlyMatrix {
  buffer : CLBuffer<isize>
}

impl GpuContext {
  pub fn new() -> GpuContext {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let add_ker = include_str!("add.ocl");
    let program = ctx.create_program_from_source(add_ker);
    program.build(&device).ok().expect("Couldn't build program.");
    let add_kernel = program.create_kernel("vector_add");

    GpuContext {
      device : device,
      context : ctx,
      command_queue : queue,
      add_kernel : add_kernel
    }
  }

  pub fn add(&self, vec_a : Vec<isize>, vec_b : Vec<isize>) -> Vec<isize> {
    assert!(vec_a.len() == vec_b.len());

    let len = vec_a.len();
    let a: CLBuffer<isize> = self.context.create_buffer(len, opencl::cl::CL_MEM_READ_ONLY);
    let b: CLBuffer<isize> = self.context.create_buffer(len, opencl::cl::CL_MEM_READ_ONLY);
    let c: CLBuffer<isize> = self.context.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);
    self.command_queue.write(&a, &&vec_a[..], ());
    self.command_queue.write(&b, &&vec_b[..], ());
    self.add_kernel.set_arg(0, &a);
    self.add_kernel.set_arg(1, &b);
    self.add_kernel.set_arg(2, &c);
    let event = self.command_queue.enqueue_async_kernel(&self.add_kernel, len, None, ());
    self.command_queue.get(&c, &event)
  }

  pub fn create_read_only(&self, size : usize) -> GpuReadOnlyMatrix {
    let buf : CLBuffer<isize> = self.context.create_buffer(size, opencl::cl::CL_MEM_READ_ONLY);
     GpuReadOnlyMatrix {
       buffer : buf
     }
  }

  pub fn create_write_only(&self, size : usize) -> GpuWriteOnlyMatrix {
    let buf : CLBuffer<isize> = self.context.create_buffer(size, opencl::cl::CL_MEM_WRITE_ONLY);
     GpuWriteOnlyMatrix {
       buffer : buf
     }
  }
}
