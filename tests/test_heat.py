import pytest
import torch
from torchheat.heat_kernel import HeatKernelGaussian, laplacian_from_data, HeatKernelKNN, torch_knn_from_data

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

def gt_heat_kernel_gaussian(
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
    heat_kernel = (heat_kernel + heat_kernel.T) / 2
    heat_kernel[heat_kernel < 0] = 0.0
    return heat_kernel

def gt_heat_kernel_knn(
    data,
    t,
    k,
):
    L = torch_knn_from_data(data, k=k, projection=False, proj_dim=10)
    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L)
    # compute the heat kernel
    heat_kernel = eigvecs @ torch.diag(torch.exp(-t * eigvals)) @ eigvecs.T
    heat_kernel = (heat_kernel + heat_kernel.T) / 2
    heat_kernel[heat_kernel < 0] = 0.0
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


@pytest.mark.parametrize("t", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("device", DEVICES)
def test_heat_kernel_gaussian(t, order, device):
    data = torch.randn(100, 5)
    data = data.to(device)
    heat_op = HeatKernelGaussian(sigma=1.0, t=t, order=order, alpha=20)
    heat_kernel = heat_op(data)

    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_gaussian(data, t=t, sigma=1.0)
    assert torch.allclose(heat_kernel, gt_heat_kernel, atol=1e-1, rtol=1e-1)

@pytest.mark.parametrize("t", [0.1, 1.0, 4.0])
@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("k", [10, 20])
def test_heat_kernel_knn(t, order, k):
    tol = 2e-1 if t > 1.0 else 1e-1
    data = torch.randn(100, 5)
    heat_op = HeatKernelKNN(k=k, t=t, order=order, graph_type="scanpy")
    heat_kernel = heat_op(data)
    
    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_knn(data, t=t, k=k)
    assert torch.allclose(heat_kernel, gt_heat_kernel, atol=tol, rtol=tol)



def test_heat_gauss_differentiable():
    data = torch.randn(100, 5, requires_grad=True)
    heat_op = HeatKernelGaussian(sigma=1.0, t=1.0, order=10, alpha=20)
    heat_kernel = heat_op(data)
    heat_kernel.sum().backward()
    assert data.grad is not None
    assert torch.all(torch.isfinite(data.grad))


if __name__ == "__main__":
    pytest.main([__file__])