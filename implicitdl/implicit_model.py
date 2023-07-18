import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .implicit_function import ImplicitFunction, ImplicitFunctionInf
from .utils import transpose

class ImplicitModel(nn.Module):
    def __init__(self, n: int, p: int, q: int,
                 f: Optional[ImplicitFunction] = ImplicitFunctionInf,
                 no_D: Optional[bool] = False,
                 bias: Optional[bool] = False):
        """
        Create a new Implicit Model:
            A: n*n  B: n*p  C: q*n  D: q*p
            X: n*m  U: p*m, m = batch size
            Note that for X and U, the batch size comes first when inputting into the model.
            These sizes reflect that the model internally transposes them so that their sizes line up with ABCD.
        
        Args:
            n: the number of hidden features.
            p: the number of input features.
            q: the number of output classes.
            f: the implicit function to use.
            no_D: whether or not to use the D matrix (i. e. whether the prediction equation should explicitly depend on the input U).
            bias: whether or not to use a bias.
        """
        super(ImplicitModel, self).__init__()

        if bias:
            p += 1

        self.n = n
        self.p = p
        self.q = q

        self.A = nn.Parameter(torch.randn(n, n)/n)
        self.B = nn.Parameter(torch.randn(n, p)/n)
        self.C = nn.Parameter(torch.randn(q, n)/n)
        self.D = nn.Parameter(torch.randn(q, p)/n) if not no_D else torch.zeros((q, p), requires_grad=False)

        self.f = f
        self.bias = bias

    def forward(self, U: torch.Tensor, X0: Optional[torch.Tensor] = None):
        U = transpose(U)
        if self.bias:
            U = F.pad(U, (0, 0, 0, 1), value=1)
        assert U.shape[0] == self.p, f'Given input size {U.shape[0]} does not match expected input size {self.p}.'

        m = U.shape[1]
        X_shape = torch.Size([self.n, m])

        if X0 is not None:
            X0 = transpose(X0)
            assert X0.shape == X_shape
        else:
            X0 = torch.zeros(X_shape, dtype=U.dtype, device=U.device)

        X = self.f.apply(self.A, self.B, X0, U)
        return transpose(self.C @ X + self.D @ U)
