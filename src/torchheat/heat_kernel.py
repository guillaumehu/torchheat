import torch
from torchheat.approx import compute_chebychev_coeff_all, expm_multiply

try:
    import scanpy as sc
except ImportError:
    pass


EPS_LOG = 1e-6
EPS_HEAT = 1e-4


def norm_sym_laplacian(A: torch.Tensor):
    deg = A.sum(dim=1)
    deg_sqrt_inv = torch.diag(1.0 / torch.sqrt(deg + EPS_LOG))
    id = torch.eye(A.shape[0], device=A.device)
    return id - deg_sqrt_inv @ A @ deg_sqrt_inv


def laplacian_from_data(data: torch.Tensor, sigma: float, alpha: int = 20):
    affinity = torch.exp(-(torch.cdist(data, data) / (2 * sigma)).pow(alpha))
    return norm_sym_laplacian(affinity)


def torch_knn_from_data(
    data: torch.Tensor, k: int, projection: bool = False, proj_dim: int = 100
):
    if projection:
        _, _, V = torch.pca_lowrank(data, q=proj_dim, center=True)
        data = data @ V
    dist = torch.cdist(data, data)
    _, indices = torch.topk(dist, k, largest=False)
    affinity = torch.zeros(data.shape[0], data.shape[0], device=data.device)
    affinity.scatter_(1, indices, 1)
    return norm_sym_laplacian(affinity)


def scanpy_knn_from_data(
    data: torch.Tensor, k: int, projection: bool = False, proj_dim: int = 100
):
    adata = sc.AnnData(data.numpy())
    if projection:
        sc.pp.pca(adata, n_comps=proj_dim)
    sc.pp.neighbors(
        adata, n_neighbors=k, use_rep="X_pca" if projection else None
    )
    return norm_sym_laplacian(
        torch.tensor(adata.obsp["connectivities"].toarray())
    )


def var_fn(x, t):
    outer = torch.outer(torch.diag(x), torch.ones(x.shape[0]))
    vol_approx = (outer + outer.T) * 0.5
    return -t * torch.log(x + EPS_LOG) + t * torch.log(vol_approx + EPS_LOG)


class BaseHeatKernel:
    def __init__(self, t: float = 1.0, order: int = 30):
        self.t = t
        self.order = order
        self.dist_fn = {
            "var": var_fn,
            "phate": lambda x, t: torch.cdist(
                -torch.log(x + EPS_LOG), -torch.log(x + EPS_LOG)
            ),
            "diff": lambda x, t: torch.cdist(x, x),
        }
        self.graph_fn = None

    def __call__(self, data: torch.Tensor):
        if self.graph_fn is None:
            raise NotImplementedError("graph_fn is not implemented")
        L = self.graph_fn(data)
        heat_kernel = self.compute_heat_from_laplacian(L)
        heat_kernel = self.sym_clip(heat_kernel)
        return heat_kernel

    def compute_heat_from_laplacian(self, L: torch.Tensor):
        n = L.shape[0]
        val = torch.linalg.eigvals(L).real
        max_eigval = val.max()
        cheb_coeff = compute_chebychev_coeff_all(
            0.5 * max_eigval, self.t, self.order
        )
        heat_kernel = expm_multiply(
            L, torch.eye(n, device=L.device), cheb_coeff, 0.5 * max_eigval
        )
        return heat_kernel

    def sym_clip(self, heat_kernel: torch.Tensor):
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        heat_kernel[heat_kernel < 0] = 0.0 + EPS_HEAT
        return heat_kernel

    def fit(self, data: torch.Tensor, dist_type: str = "var"):
        assert dist_type in self.dist_fn
        heat_kernel = self(data)
        return self.dist_fn[dist_type](heat_kernel, self.t)


class HeatKernelGaussian(BaseHeatKernel):
    """Approximation of the heat kernel with a graph from a gaussian affinity matrix.
    Uses Chebyshev polynomial approximation.
    """

    _is_differentiable = True

    def __init__(
        self,
        sigma: float = 1.0,
        alpha: int = 20,
        order: int = 30,
        t: float = 1.0,
    ):
        super().__init__(t=t, order=order)
        self.sigma = sigma
        self.alpha = alpha if alpha % 2 == 0 else alpha + 1
        self.graph_fn = lambda x: laplacian_from_data(
            x, self.sigma, alpha=self.alpha
        )


class HeatKernelKNN(BaseHeatKernel):
    """Approximation of the heat kernel with a graph from a k-nearest neighbors affinity matrix.
    Uses Chebyshev polynomial approximation.
    """

    _is_differentiable = False
    _implemented_graph = {
        "torch": torch_knn_from_data,
        "scanpy": scanpy_knn_from_data,
    }

    def __init__(
        self,
        k: int = 10,
        order: int = 30,
        t: float = 1.0,
        projection: bool = False,
        proj_dim: int = 100,
        graph_type: str = "torch",
    ):
        super().__init__(t=t, order=order)
        assert (
            graph_type in self._implemented_graph
        ), f"Type must be in {self._implemented_graph}"
        self.k = k
        self.projection = projection
        self.proj_dim = proj_dim
        self.graph_fn = lambda x: self._implemented_graph[graph_type](
            x, self.k, projection=self.projection, proj_dim=self.proj_dim
        )
