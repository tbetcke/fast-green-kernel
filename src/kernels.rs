//! Definitions of the supported Greens function kernels.
use crate::RealType;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use num;

pub enum EvalMode {
    Value,
    ValueGrad,
}

/// Implementation of the Laplace kernel
pub fn laplace_kernel<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    result: &mut ArrayViewMut2<T>,
    _eval_mode: &EvalMode,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();

    let m_inv_4pi: T =
        num::traits::cast::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    result.fill(zero);

    Zip::from(target.view())
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .for_each(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    result.mapv_inplace(|item| m_inv_4pi / item.sqrt());
    result
        .iter_mut()
        .filter(|item| item.is_infinite())
        .for_each(|item| *item = zero);
}

/// Implementation of the Helmholtz kernel
pub fn helmholtz_kernel<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    result: &mut ArrayViewMut2<num::complex::Complex<T>>,
    wavenumber: num::complex::Complex<f64>,
    _eval_mode: &EvalMode,
) {
    use ndarray::Zip;

    let real_zero: T = num::traits::zero();
    let complex_zero: num::complex::Complex<T> =
        num::Complex::new(num::traits::zero(), num::traits::zero());
    let mut diff = Array1::<T>::zeros(sources.len_of(Axis(1)));
    let m_inv_4pi: T =
        num::traits::cast::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let wavenumber = num::complex::Complex::new(
        num::traits::cast::cast::<f64, T>(wavenumber.re).unwrap(),
        num::traits::cast::cast::<f64, T>(wavenumber.im).unwrap(),
    );

    let exp_factor: T = (-wavenumber.re).exp();

    Zip::from(target.view())
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(&mut diff)
                .for_each(|&source_value, diff_value| {
                    *diff_value += (target_value - source_value) * (target_value - source_value)
                })
        });

    diff.mapv_inplace(|item| item.sqrt());

    Zip::from(result.index_axis_mut(Axis(0), 0))
        .and(diff.view())
        .for_each(|result_value, diff_value| {
            *result_value = num::Complex::new(
                exp_factor * m_inv_4pi * (wavenumber.re * *diff_value).cos() / *diff_value,
                exp_factor * m_inv_4pi * (wavenumber.re * *diff_value).sin() / *diff_value,
            );
        });
    Zip::from(result.index_axis_mut(Axis(0), 0))
        .and(diff.view())
        .for_each(|res_value, diff_value| {
            if *diff_value == real_zero {
                *res_value = complex_zero
            }
        });
}

