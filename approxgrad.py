"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.ODIN
    :members:

.. autofunction:: pytorch_ood.detector.odin_preprocessing

"""

import faiss
import logging
import warnings
from typing import Callable, List, Optional, TypeVar
from functorch import make_functional_with_buffers, vmap, grad

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from detector import Detector
import annoy

log = logging.getLogger(__name__)

Self = TypeVar("Self")

loss_fn = torch.nn.CrossEntropyLoss()

def compute_grad(sample, target, model, required_params):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)
    prediction = model(sample)
    loss = loss_fn(prediction, target) / 1000
    return torch.autograd.grad(loss, required_params)


def compute_sample_grads(data, targets, model, required_params, batch_size=125):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i], model, required_params) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads[0]



def shadow_backprop(model, x, bp_layer, criterion, return_predictions=False):
    x.requires_grad = True
    # freeze all layers except the one we want to backprop through
    for param in model.parameters():
        param.requires_grad = False
    bp_layer.weight.requires_grad = True
    bp_layer.bias.requires_grad = True

    required_params = []
    for param in model.parameters():
        if param.requires_grad:
            required_params.append(param)

    o = model(x)

    _, ground_truth = torch.max(o, 1)
    if return_predictions:
        return compute_sample_grads(x, ground_truth, model, required_params, batch_size=x.shape[0]), ground_truth
    else:
        return compute_sample_grads(x, ground_truth, model, required_params, batch_size=x.shape[0])




class ApproxGrad(Detector):

    def __init__(self, model, criterion, layer):
        super(ApproxGrad, self).__init__()
        self.model = model
        self.criterion = criterion
        self.layer = layer




    def predict(self, x: torch.Tensor, return_predictions=False, y=None):

        gradients, predictions = shadow_backprop(self.model, x, self.layer, self.criterion, return_predictions=True)
        if return_predictions:
            return gradients.detach().cpu().numpy(), predictions.detach().cpu().numpy()
        else:
            return gradients.detach().cpu().numpy()



    def fit(self, vectors) -> Self:
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self
