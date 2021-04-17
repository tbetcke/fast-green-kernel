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
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

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
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    make_laplace_evaluator(sources, targets).assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f64(
    target_ptr: *const f64,
    source_ptr: *const f64,
    charge_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::direct_evaluator::*;
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe { ndarray::ArrayView2::from_shape_ptr((nsources, ncharge_vecs), charge_ptr) };
    let result = unsafe { ndarray::ArrayViewMut3::from_shape_ptr((ncharge_vecs, ntargets, ncols), result_ptr) };

    make_laplace_evaluator(sources, targets).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f32(
    target_ptr: *const f32,
    source_ptr: *const f32,
    charge_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::direct_evaluator::*;
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe { ndarray::ArrayView2::from_shape_ptr((nsources, ncharge_vecs), charge_ptr) };
    let result = unsafe { ndarray::ArrayViewMut3::from_shape_ptr((ncharge_vecs, ntargets, ncols), result_ptr) };


    make_laplace_evaluator(sources, targets).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );


}

