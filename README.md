# heatdist
Implementation of diffusion-based distances in torch.

```python
from torchheat.heat_kernel import HeatKernelGaussian
import torch    

data = torch.randn(100, 5)
heat_op = HeatKernelGaussian(sigma=1.0, t=1.0)
dist = heat_op.fit(data, dist_type="var") # ["var", "phate", "diff"]
```
