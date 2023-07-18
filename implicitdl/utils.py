import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from .logger import Logger


def transpose(X):
    """
    Convenient function to transpose a matrix.
    """
    assert len(X.size()) == 2, "data must be 2D"
    X = torch.transpose(X, -1, -2)
    return X


def get_valid_accuracy(model, loss_fn, valid_dl, device):
    """
    Run the model with the validation dataset and compute the loss and accuracy.
    """
    for xs, ys in valid_dl:
        xs, ys = xs.to(device), ys.to(device)
        # if isinstance(model, (ImplicitRobustModel, ImplicitRobustModelRank1FT)):
        #     pred, _ = model(xs, 0.0)
        # else:
        pred = model(xs)
        pred = pred if isinstance(pred, torch.Tensor) else torch.from_numpy(pred).to(device)
        loss = loss_fn(pred, ys)
        pred_i = np.argmax(pred.cpu().detach().numpy(), axis=-1)
        correct = np.sum([1 if ys[i] == pred_i[i] else 0 for i in range(len(ys))])
        return loss, correct/len(ys)


def train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, dirname, device=None):
    """
    A pre-built convenient routine to train a model and print and log the results.
    
    Args:
        model: the model (torch.nn.Module) to be trained.
        train_dl: train data loader.
        valid_dl: validation data loader.
        optimizer: optimizer from the torch.optim package.
        loss_fn: loss function to optimise the model against.
        epochs: the number of epochs to train the model.
        dirname: folder to save the log file.
        device: the device to train the model with (default cuda if available, else cpu).
    
    Returns:
        The trained model and the logger object.
    """
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