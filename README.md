## Implicit Deep Learning Models

This repository contains an implementation of implicit deep learning models, first introduced in [Implicit Deep Learning](https://epubs.siam.org/doi/abs/10.1137/20M1358517) published in *SIAM Journal on Mathematics of Data Science*, 2021, as well as future work highlighting the advantages of implicit models in robustness, sparsity and OOD generalization.

Implicit Deep Learning is an alternative to classical deep neural networks defined via a fixed-point equation rather than explicit features. The outputs are determined only implicitly through this equilibrium. 

### Brief definition
Given a dataset with input matrix $U \in \mathbb{R}^{p\times m}$ and output matrix $Y \in \mathbb{R}^{q\times m}$, where each column represents an input or output vector and m is the batch size, an implicit model consists of an equilibrium equation in a "state matrix" $X \in \mathbb{R}^{n\times m}$ and a prediction equation:


$X = \phi (AX + BU)$ (equilibrium equation)

$\hat{Y}(U) = CX + DU$ (prediction equation)

where $\phi: \mathbb{R}^{n\times m} \to \mathbb{R}^{n\times m}$ is a nonlinear activation that is strictly increasing and component-wise non-expansive, such as ReLU, tanh or sigmoid. While the above model seems very specific, it covers as special case most known architectures arising in deep learning. Matrices $A\in \mathbb{R}^{n\times n}$, $B\in \mathbb{R}^{n\times p}$, $C\in \mathbb{R}^{q\times n}$ and $D\in \mathbb{R}^{q\times p}$ are model parameters.

For illustration, below is an implicit model equivalent to a 2-layer feedforward neural network: 
![feedforward-implicit-illus](https://github.com/Implicit-DL/implicit-deep-learning/blob/main/figures/ff-illus.jpg)

As opposed to the above network, the typical implicit model does not have a clear hierachical, layered structure:
![feedforward-implicit-illus](https://github.com/Implicit-DL/implicit-deep-learning/blob/main/figures/im-illus.jpg)

Journal article: https://epubs.siam.org/doi/abs/10.1137/20M1358517

Press article: https://medium.com/analytics-vidhya/what-is-implicit-deep-learning-9d94c67ec7b4
## Getting started

Requisites: Python version 3.9

Clone the repo by running:

`git clone https://github.com/Implicit-DL/implicit-model-archive.git`

Required packages are detailed in `requirements.txt`

Install required packages by running:

`pip install -r requirements.txt`

## Data
We provide examples of loading several existing datasets, namely [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](http://yann.lecun.com/exdb/mnist/), in [examples](https://github.com/Implicit-DL/implicit-deep-learning/tree/main/examples). 

## Usage
Here we provide one example of training and using an imlicit model (hyper-parameters are to be adapted to each use).
### Step 1: Load data and define hyper-parameters

The hyperparameters of an implicit model determine the size of the state matrices and the fixed point:
- n: corresponds to the size of the hidden state matrix $X$
- p: flattened input size
- q: output size

We will have $A\in \mathbb{R}^{n\times n}$, $B\in \mathbb{R}^{n\times p}$, $C\in \mathbb{R}^{q\times n}$ and $D\in \mathbb{R}^{q\times p}$.

```
epochs = 10
bs = 100
lr = 5e-4

train_ds, train_dl, valid_ds, valid_dl = cifar_load(bs)

n = 300 # the main parameter of an implicit model, determining the size of the hidden state matrix X 
p = 3072 # the flattened input size, in this case 32 x 32 (pixels) x 3 (channels) for CIFAR
q = 10 # the output size

model = ImplicitModel(n, p, q, f=ImplicitFunctionInf, no_D=False)
opt = optim.Adam(ImplicitModel.parameters(model), lr=lr)
loss_fn = F.cross_entropy
```

### Step 2: **Train** the model

A generic training routine is provided in implicitdl.utils and can be invoked as follows:

```
model, _ = train(model, train_dl, valid_dl, opt, loss_fn, epochs, "CIFAR_Implicit_300_Inf")
```

The training routine can also be custom-defined. For reference, here is the default training routine:
```
def train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, dirname, device=None):
    # load the model to GPU / CPU device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc"],
                dir_name=dirname)

    for i in range(epochs):
        j = 0
        for xs, ys in train_dl:
            # forward step
            pred = model(xs.to(device))
            loss = loss_fn(pred, ys.to(device))

            # backward step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log the performance and the model
            valid_res = get_valid_accuracy(model, loss_fn, valid_dl, device)
            log_dict = {
                "batch": j,
                "loss": loss,
                "valid_loss": valid_res[0],
                "valid_acc":valid_res[1]
            }
            logger.log(log_dict, model, "valid_acc")

            j+=1
        print("--------------epoch: {}. loss: {}".format(i, loss))
        pass
    return model, logger
```
### Step 3: Use the model for **inference**
```
pred = model(xs)
```
### A Note on Convergence
You may run into a warning like "Picard iterations did not converge: err=1.5259e-05, status=max itrs reached." For the most part, these warnings can be safely ignored if the err term is relatively small (i. e. less than around 1e-3). They are simply there to indicate that the fixed point was not found within the specified tolerance within the specified number of iterations. This can happen with "more difficult" inputs to the implicit model, especially those outside of the training distribution, where more iterations are required to find the fixed point.

As of now, all implicit functions use the Picard method to solve the forward pass and gradient pass implicit equations. The default convergence parameters are given at the top of implicit_function.py. ```mitr``` is the max allowed number of forward iterations, ```grad_mitr``` is the max allowed number of gradient iterations, ```tol``` is the forward solver tolerance, and ```grad_tol``` is the gradient solver tolerance. To override these defaults, subclass the implicit function you wish to use as follows:
```
class CustomInf(ImplicitFunctionInf):
    """
    Change the default convergence parameters.
    """
    mitr = grad_mitr = 500
    tol = grad_tol = 1e-6
```
See examples/custom_nonlinearity.ipynb for a full tutorial.

### Custom Nonlinearities
By default, the implicit model uses the ReLU nonlinearity. All of our experiments have used this nonlinearity, but if you wish to customize this feature of the model, you must subclass your desired implicit function and override the phi(...) (nonlinearity) and dphi(...) (nonlinearity gradient) static methods, as follows:
```
class ImplicitFunctionInfSiLU(ImplicitFunctionInf):
    """
    An implicit function that uses the SiLU nonlinearity.
    """
    
    @staticmethod
    def phi(X):
        return X * torch.sigmoid(X)

    @staticmethod
    def dphi(X):
        grad = X.clone().detach()
        sigmoid = torch.sigmoid(grad)
        return sigmoid * (1 + grad * (1 - sigmoid))
```
See examples/custom_nonlinearity.ipynb for a full tutorial.

### Stacking Implicit Models
No extra expressive power comes from stacking implicit models like standard neural network layers, since the stack can be compressed into a single implicit model. To increase model complexity, the only necessary step is to increase the hidden size of the model, $n$.

### Visualisation of Implicit Models
Currently, Implicit Models can be visualised by plotting the non-zero parameters in the M matrix as follows:
```
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.figure(figsize=(15,15), dpi=300)
plt.spy(sp.bmat([[model.C, model.D], [model.A, model.B]]), markersize=0.01, color='black')
plt.show()
```
The results will look something like this:
![feedforward-implicit](https://github.com/Implicit-DL/implicit-deep-learning/blob/main/figures/ff.png)

### Further Examples
More examples can be found in [examples](https://github.com/Implicit-DL/implicit-deep-learning/tree/main/examples)

## Collaboration & Contribution
Please contact `aliciatsai@berkeley.edu` and `laurent.eg@vinuni.edu.vn` if you would like to collaborate with us on Implicit Deep Learning research.

To contribute, please clone or fork the repo and later create a pull request to merge your addition into the main branch.

## Bibliography
Source code contributors: Alicia Tsai, Max Emerling, Juliette Decugis, Ashwin Ganesh, Fangda Gu.

If you use or extend our work on implicit models, please cite the following paper.

    @article{ghaoui2021implicitdl,
    title={Implicit deep learning},
    author={El Ghaoui, Laurent and Gu, Fangda and Travacca, Bertrand and Askari, Armin and Tsai, Alicia},
    journal={SIAM Journal on Mathematics of Data Science},
    volume={3},
    number={3},
    pages={930--958},
    year={2021},
    publisher={SIAM}
    }
