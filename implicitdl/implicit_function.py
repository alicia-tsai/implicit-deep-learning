import torch
from torch.autograd import Function
import numpy as np

class ImplicitFunction(Function):
    mitr = grad_mitr = 300
    tol = grad_tol = 3e-6

    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        with torch.no_grad():
            X, err, status = cls.inn_pred(A, B @ U, X0, cls.mitr, cls.tol)
        ctx.save_for_backward(A, B, X, U)
        if status not in "converged":
            print(f"Picard iterations did not converge: err={err.item():.4e}, status={status}")
        return X

    @classmethod
    def backward(cls, ctx, *grad_outputs):
        A, B, X, U = ctx.saved_tensors

        grad_output = grad_outputs[0]
        assert grad_output.size() == X.size()

        DPhi = cls.dphi(A @ X + B @ U)
        V, err, status = cls.inn_pred_grad(A.T, DPhi * grad_output, DPhi, cls.grad_mitr, cls.grad_tol)
        if status not in "converged":
            print(f"Gradient iterations did not converge: err={err.item():.4e}, status={status}")
        grad_A = V @ X.T
        grad_B = V @ U.T
        grad_U = B.T @ V

        return grad_A, grad_B, torch.zeros_like(X), grad_U

    @staticmethod
    def phi(X):
        return torch.clamp(X, min=0)

    @staticmethod
    def dphi(X):
        grad = X.clone().detach()
        grad[X <= 0] = 0
        grad[X > 0] = 1

        return grad

    @classmethod
    def inn_pred(cls, A, Z, X, mitr, tol):
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = cls.phi(A @ X + Z)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def inn_pred_grad(AT, Z, DPhi, mitr, tol):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = DPhi * (AT @ X) + Z
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status


class ImplicitFunctionTriu(ImplicitFunction):
    """
    Constrains the A matrix to be upper triangular. Only allows for the implicit model to learn feed-forward architectures.
    """

    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        A = A.triu_(1)
        return super(ImplicitFunctionTriu, cls).forward(ctx, A, B, X0, U)

    @classmethod
    def backward(cls, ctx, *grad_outputs):
        grad_A, grad_B, grad_X, grad_U = super(ImplicitFunctionTriu, cls).backward(ctx, *grad_outputs)
        return grad_A.triu(1), grad_B, grad_X, grad_U


class ImplicitFunctionInf(ImplicitFunction):
    """
    Implicit function which projects A onto the infinity norm ball. Allows for the model to learn closed-loop feedback.
    """

    @classmethod
    def forward(cls, ctx, A, B, X0, U):

        # project A on |A|_inf=v
        v = 0.95

        A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v)
            # write back
            A_np[idx, :] = a

        A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))

        return super(ImplicitFunctionInf, cls).forward(ctx, A, B, X0, U)
