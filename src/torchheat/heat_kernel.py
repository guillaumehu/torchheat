import torch

from torchheat.approx import compute_chebychev_coeff_all, expm_multiply

EPS_LOG = 1e-6


class HeatKernelGaussian:
    """Approximation of the heat kernel with a graph from a gaussian affinity matrix.
    Uses Chebyshev polynomial approximation.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        alpha: int = 20,
        order: int = 30,
        t: float = 1.0,
        lap_type: str = "sym",
    ):
        assert lap_type in ["sym", "comb"]
        self.sigma = sigma
        self.order = order
        self.t = t
        self.alpha = alpha if alpha % 2 == 0 else alpha + 1
        self.dist_fn = {
            "var": lambda x: -torch.log(x + EPS_LOG),
            "phate": lambda x: torch.cdist(
                -torch.log(x + EPS_LOG), -torch.log(x + EPS_LOG)
            ),
            "diff": lambda x: torch.cdist(x, x),
        }

    def __call__(self, data: torch.Tensor):
        L = laplacian_from_data(data, self.sigma, alpha=self.alpha)
        val = torch.linalg.eigvals(L).real
        max_eigval = val.max()
        cheb_coeff = compute_chebychev_coeff_all(
            0.5 * max_eigval, self.t, self.order
        )
        heat_kernel = expm_multiply(
            L, torch.eye(data.shape[0]), cheb_coeff, 0.5 * max_eigval
        )
        # symmetrize the heat kernel, for larger t it may not be symmetric
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        return heat_kernel

    def get_distances(self, data: torch.Tensor, dist_type: str = "var"):
        assert dist_type in self.dist_fn
        heat_kernel = self(data)
        return self.dist_fn[dist_type](heat_kernel)


def laplacian_from_data(data: torch.Tensor, sigma: float, alpha: int = 20):
    affinity = torch.exp(
        -torch.cdist(data, data) ** alpha / (2 * sigma**alpha)
    )
    degree = affinity.sum(dim=1)
    inv_deg_sqrt = 1.0 / torch.sqrt(degree)
    D = torch.diag(inv_deg_sqrt)
    L = torch.eye(data.shape[0]) - D @ affinity @ D
    return L
