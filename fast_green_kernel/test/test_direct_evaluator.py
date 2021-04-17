"""Unit tests for direct assembly and evaluation of kernels."""
import numpy as np
import pytest


@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_assemble(dtype, rtol):
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

    actual = assemble_laplace_kernel(targets, sources, dtype=dtype)

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

@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_only_values(dtype, rtol):
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
    charges = rng.random((nsources, ncharge_vecs), dtype=dtype)

    actual = evaluate_laplace_kernel(targets, sources, charges, dtype=dtype)

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

    expected = (expected @ charges).reshape(ntargets, 1, ncharge_vecs)

    np.testing.assert_allclose(actual, expected, rtol=rtol)

@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_values_and_deriv(dtype, rtol):
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
    charges = rng.random((nsources, ncharge_vecs), dtype=dtype)

    actual = evaluate_laplace_kernel(targets, sources, charges, dtype=dtype, return_gradients=True, parallel=True)

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((ntargets, 4, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = target.reshape(3, 1) - sources
        dist = np.linalg.norm(diff, axis=0)
        expected[index, 0, :] = 1.0 / ( 4 * np.pi * dist)
        expected[index, 1:, :] = diff / (4 * np.pi * dist**3)
        expected[index, :, dist == 0] = 0


    # Reset the warnings
    np.seterr(**old_params)

    expected = (expected @ charges)

    np.testing.assert_allclose(actual, expected, rtol=rtol)
