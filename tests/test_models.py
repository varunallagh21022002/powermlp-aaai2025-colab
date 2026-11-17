import unittest
import torch
import numpy as np
from scipy.interpolate import BSpline
from src.models import PowerLayer, PowerMLP, MLP, SplineKAN, SplineKANLayer, count_parameters

class ArchitectureTests(unittest.TestCase):
    def test_exact_equation(self):
        torch.manual_seed(42)
        layer = PowerLayer(2,3).double()
        x = torch.randn(5,2,dtype=torch.float64)
        expected = torch.relu(x @ layer.affine.weight.T + layer.affine.bias)**3
        expected += (x*torch.sigmoid(x)) @ layer.basis.weight.T
        torch.testing.assert_close(layer(x),expected)

    def test_gradients_parameters_and_input(self):
        from torch.func import functional_call
        layer = PowerLayer(2,3).double()
        x = torch.tensor([[0.2,-0.4],[0.7,0.3]],dtype=torch.float64,requires_grad=True)
        params = dict(layer.named_parameters())
        names = list(params)
        def fn(x,*values):
            return functional_call(layer,dict(zip(names,values)),(x,))
        self.assertTrue(torch.autograd.gradcheck(fn,(x,*params.values()),eps=1e-6,atol=1e-5))

    def test_parameter_counts(self):
        self.assertEqual(count_parameters(PowerMLP((2,4,1))),25)
        self.assertEqual(count_parameters(MLP((2,6,1))),25)
        self.assertEqual(count_parameters(SplineKAN((2,1,1),3)),24)
        self.assertEqual(count_parameters(PowerMLP()),689)
        self.assertEqual(count_parameters(MLP()),709)
        self.assertEqual(count_parameters(SplineKAN()),735)

    def test_splines_against_scipy(self):
        layer = SplineKANLayer(2,3).double()
        x = torch.linspace(-.99,.99,17,dtype=torch.float64).reshape(-1,1).repeat(1,2)
        t = layer.grid.numpy()
        n = len(t)-layer.k-1
        reference = BSpline(t,np.eye(n),layer.k)(x.numpy())
        np.testing.assert_allclose(layer.bsplines(x).numpy(),reference,atol=1e-12)
        torch.testing.assert_close(layer.bsplines(x).sum(-1),torch.ones_like(x))

    def test_linear_output_and_backward(self):
        for model in [PowerMLP(),MLP(),SplineKAN()]:
            y=model(torch.randn(7,2))
            self.assertEqual(y.shape,(7,1))
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsInstance(PowerMLP().layers[-1],torch.nn.Linear)

if __name__ == '__main__':
    unittest.main()
