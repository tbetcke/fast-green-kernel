use ndarray;

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f64(
    target_ptr: *const f64,
    source_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::direct_evaluator::*;

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    println!("I am here.");

    make_laplace_evaluator(sources, targets).assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f32(
    target_ptr: *const f32,
    source_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::direct_evaluator::*;

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    println!("I am here.");


    make_laplace_evaluator(sources, targets).assemble_in_place(result, threading_type);
}
