from typing import NoReturn

import torch
from torch import FloatTensor, LongTensor, nn
from torch.nn import functional as F

from .metric_conv import MetricConv


class LinearMetricNet(nn.Module):
    r"""
    This model adapts the architecture of the correspondence model used in SpiralNet++ (https://leichen2018.github.io/files/spiralnet_plusplus.pdf). We replace the SpiralNet++ convolutional operators with ``MetricConv`` operators.

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * info (``str``) -- type of ``MetricConv`` to use
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(LinearMetricNet, self).__init__()
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True

        self.fc0 = nn.Linear(3, 16)
        self.conv1 = MetricConv(16, 32, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.conv2 = MetricConv(32, 64, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.conv3 = MetricConv(64, 128, info=info, embedding_dim=embedding_dim,symmetric=symmetric)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, out_feats)

        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        """
        Sets initial parameters of weights follow a Uniform Xavier distribution for fully connected matrices and a uniform distribution for biases.
        """
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, features: FloatTensor, vertices: FloatTensor, edges: LongTensor, faces: LongTensor) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices

        :return: Returns tensor contains ``out_feats`` features for each vertex.
        """
        x = features
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, vertices, edges, faces))
        x = F.elu(self.conv2(x, vertices, edges, faces))
        x = F.elu(self.conv3(x, vertices, edges, faces))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out