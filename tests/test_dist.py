import pytest
import torch
from torchheat.heat_kernel import HeatKernelGaussian, laplacian_from_data

@pytest.mark.parametrize("t", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("dist_type", ["var", "phate", "diff"])
def test_heat_kernel_gaussian(t, order, dist_type):
    data = torch.randn(100, 5)
    heat_op = HeatKernelGaussian(sigma=1.0, t=t, order=order)
    dist = heat_op.fit(data, dist_type=dist_type)

    # test if symmetric
    assert torch.allclose(dist, dist.T, rtol=1e-3)

    # test if positive
    assert torch.all(dist >= 0)

    # assert that all value on the diagonal are smaller then the non-diagonal
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i != j:
                assert dist[i, i] <= dist[i, j]

if __name__ == "__main__":
    pytest.main([__file__])