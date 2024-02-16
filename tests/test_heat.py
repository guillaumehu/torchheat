import pytest
import torch
from torchheat.heat_kernel import HeatKernelGaussian, laplacian_from_data


def gt_heat_kernel_knn(
    data,
    t,
    sigma,
    alpha=20,
):
    L = laplacian_from_data(data, sigma, alpha=alpha)
    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L)
    # compute the heat kernel
    heat_kernel = eigvecs @ torch.diag(torch.exp(-t * eigvals)) @ eigvecs.T
    return heat_kernel


def test_laplacian():
    data = torch.randn(100, 5)
    sigma = 1.0
    L = laplacian_from_data(data, sigma)
    assert torch.allclose(L, L.T)
    # compute the largest eigenvalue
    eigvals = torch.linalg.eigvals(L).real
    max_eigval = eigvals.max()
    min_eigval = eigvals.min()
    assert max_eigval <= 2.0
    torch.testing.assert_allclose(min_eigval, 0.0)


@pytest.mark.parametrize("t", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("order", [10, 30, 50])
def test_heat_kernel_gaussian(t, order):
    data = torch.randn(100, 5)
    heat_op = HeatKernelGaussian(sigma=1.0, t=t, order=order)
    heat_kernel = heat_op(data)

    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_knn(data, t=t, sigma=1.0)
    assert torch.allclose(heat_kernel, gt_heat_kernel, atol=1e-3)


def test_heat_gauss_differentiable():
    data = torch.randn(100, 5, requires_grad=True)
    heat_op = HeatKernelGaussian(sigma=1.0, t=1.0, order=10)
    heat_kernel = heat_op(data)
    heat_kernel.sum().backward()
    assert data.grad is not None
    assert torch.all(torch.isfinite(data.grad))


if __name__ == "__main__":
    pytest.main([__file__])