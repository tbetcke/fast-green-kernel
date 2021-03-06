"""Unit tests for direct assembly and evaluation of kernels."""
import numpy as np
import pytest


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_assemble(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import assemble_laplace_kernel

    nsources = 10
    ntargets = 20

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target

    actual = assemble_laplace_kernel(targets, sources, dtype=dtype, parallel=parallel)

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((ntargets, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        expected[index, :] = 1.0 / (
            4 * np.pi * np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        )

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_only_values(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import evaluate_laplace_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_laplace_kernel(
        targets, sources, charges, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((nsources, ntargets), dtype=dtype)

    for index, target in enumerate(targets.T):
        expected[:, index] = 1.0 / (
            4 * np.pi * np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        )

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    expected = np.expand_dims(charges @ expected, -1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_values_and_deriv(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import evaluate_laplace_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_laplace_kernel(
        targets, sources, charges, dtype=dtype, return_gradients=True, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets, 4), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = sources - target.reshape(3, 1)
        dist = np.linalg.norm(diff, axis=0)
        expected[:, index, 0] = 1.0 / (4 * np.pi * dist)
        expected[:, index, 1:] = diff.T / (4 * np.pi * dist.reshape(nsources, 1) ** 3)
        expected[dist == 0, index, :] = 0

    # Reset the warnings
    np.seterr(**old_params)

    expected = np.tensordot(charges, expected, 1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_assemble(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import assemble_helmholtz_kernel

    wavenumber = 2.5

    nsources = 10
    ntargets = 20

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target

    actual = assemble_helmholtz_kernel(
        targets, sources, wavenumber, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((ntargets, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[index, :] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[index, dist == 0] = 0

    # Reset the warnings
    np.seterr(**old_params)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_evaluate_only_values(dtype, rtol):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import evaluate_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 2.5

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=real_type) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=real_type
    )

    actual = evaluate_helmholtz_kernel(
        targets, sources, charges, wavenumber, dtype=dtype, parallel=False
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[:, index] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[dist == 0, index] = 0

    # Reset the warnings
    np.seterr(**old_param)

    expected = np.expand_dims(np.tensordot(charges, expected, 1), -1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_evaluate_values_and_deriv(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from fast_green_kernel.direct_evaluator import evaluate_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 2.5

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=real_type) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=real_type
    )

    actual = evaluate_helmholtz_kernel(
        targets,
        sources,
        charges,
        wavenumber,
        dtype=dtype,
        return_gradients=True,
        parallel=parallel,
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets, 4), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = target.reshape(3, 1) - sources
        dist = np.linalg.norm(diff, axis=0)
        expected[:, index, 0] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[:, index, 1:] = (
            diff.T * expected[:, index, 0].reshape(nsources, 1) / dist.reshape(nsources, 1)**2
            * (1j * wavenumber * dist.reshape(nsources, 1) - 1)
        )
        expected[dist == 0, index, :] = 0

    # Reset the warnings
    np.seterr(**old_params)

    expected = np.tensordot(charges, expected, 1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)
