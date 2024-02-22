# heatdist
Implementation of diffusion-based distances in torch.

```python
from torchheat.heat_kernel import HeatKernelGaussian, HeatKernelKNN
import torch    

data = torch.randn(100, 5)
# Heat kernel for a gaussian affinity matrix
heat_op = HeatKernelGaussian(sigma=1.0, t=1.0)
dist = heat_op.fit(data, dist_type="var") # ["var", "phate", "diff"]
# Heat kernel for a k-nearest neighbor affinity matrix
heat_op = HeatKernelKNN(k=5, t=1.0)
dist = heat_op.fit(data, dist_type="var") # ["var", "phate", "diff"]
```
