"""Independent implementation of Qiu et al. (AAAI 2025), Eq. (6)."""
import torch
from torch import nn
from torch.nn import functional as F

class PowerLayer(nn.Module):
    def __init__(self, in_features, out_features, k=3, basis=True):
        super().__init__()
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        self.k = k
        self.affine = nn.Linear(in_features, out_features)
        self.basis = nn.Linear(in_features, out_features, bias=False) if basis else None

    def forward(self, x):
        out = F.relu(self.affine(x)).pow(self.k)
        return out + self.basis(F.silu(x)) if self.basis is not None else out

class PowerMLP(nn.Module):
    def __init__(self, widths=(2, 32, 8, 1), k=3, basis=True):
        super().__init__()
        if len(widths) < 2 or any(w < 1 for w in widths):
            raise ValueError('widths must contain at least two positive sizes')
        self.layers = nn.Sequential(*[
            PowerLayer(a, b, k, basis) for a, b in zip(widths[:-2], widths[1:-1])
        ], nn.Linear(widths[-2], widths[-1]))

    def forward(self, x):
        return self.layers(x)

class MLP(nn.Module):
    def __init__(self, widths=(2, 32, 18, 1)):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths)-2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SplineKANLayer(nn.Module):
    """Fixed uniform grid KAN, Eqs. (1)-(4); no adaptive grid or symbolic branch.

    Each edge has independent base and spline scales plus G+k coefficients.
    Vectorizes basis evaluation across edges. This is NOT pykan's runtime.
    """
    def __init__(self, in_features, out_features, grid_size=3, k=3, grid_range=(-1., 1.)):
        super().__init__()
        if grid_size < 1 or k < 1 or grid_range[0] >= grid_range[1]:
            raise ValueError('invalid grid settings')
        self.k = k
        step = (grid_range[1]-grid_range[0])/grid_size
        grid = torch.arange(-k, grid_size+k+1)*step+grid_range[0]
        self.register_buffer('grid', grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_scale = nn.Parameter(torch.ones(out_features, in_features))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, grid_size+k))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.coefficients, std=0.05)

    def bsplines(self, x):
        t = self.grid
        z = x.unsqueeze(-1)
        b = ((z >= t[:-1]) & (z < t[1:])).to(x.dtype)
        for degree in range(1, self.k+1):
            b = ((z-t[:-(degree+1)]) / (t[degree:-1]-t[:-(degree+1)]) * b[..., :-1]
                 + (t[degree+1:]-z) / (t[degree+1:]-t[1:-degree]) * b[..., 1:])
        return b

    def forward(self, x):
        spline = torch.einsum('bik,oik->bo', self.bsplines(x),
                              self.coefficients*self.spline_scale.unsqueeze(-1))
        return F.linear(F.silu(x), self.base_weight) + spline

class SplineKAN(nn.Module):
    def __init__(self, widths=(2, 2, 1, 1), grid_size=100, k=3):
        super().__init__()
        self.layers = nn.Sequential(*[SplineKANLayer(a,b,grid_size,k) for a,b in zip(widths[:-1],widths[1:])])

    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
