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

Below is an example of distance matrices from a line embedded in two dimensions. The Euclidean distance between the two sets of points highlighted in green does not reflect the true distances on the one dimensional line.
![image](https://github.com/guillaumehu/torchheat/assets/57917099/89b845a1-1625-4f36-9e8c-d3db62281e2c)
