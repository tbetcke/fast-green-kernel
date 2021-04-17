use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fast_green_kernel::direct_evaluator::*;
use ndarray;
use rand::Rng;

fn benchmark_laplace_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f64>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f64>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f64>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f64>());
    targets.map_inplace(|item| *item = rng.gen::<f64>());

    c.bench_function("laplace single precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view())
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_laplace_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f32>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f32>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f32>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f32>());
    targets.map_inplace(|item| *item = rng.gen::<f32>());

    c.bench_function("laplace double precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view())
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = benchmark_laplace_single_precision, benchmark_laplace_double_precision,
}
    
criterion_main!(benches);
