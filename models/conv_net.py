from typing import NoReturn

import torch
from torch import FloatTensor, LongTensor, nn
from torch.nn import functional as F

from .metric_conv import MetricConv
from .blocks import MetricResBlock


class MetricConvNet(nn.Module):
    r"""
    All (``MetricConv``) convolutional network using a progressively wider network as described below:

                                in_feats->16->32->64->128->256->out_feats

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * classification (``bool``) -- Bool indicating whether model is end-to-end or classification
        * info (``str``) -- type of ``MetricConv`` to use
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
        * layers (``List[int]``) -- Sequence of conv layers to use``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(MetricConvNet, self).__init__()
        classification = kwargs["classification"] if "classification" in kwargs.keys() else False
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True
        layers = kwargs["layers"] if "layers" in kwargs.keys() else [16, 32, 64, 128, 256]
        layers = [in_feats] + layers

        convs = []
        for i in range(len(layers) - 2):
            if layers[i] == layers[i + 1]:
                conv = MetricResBlock(layers[i], info, embedding_dim=embedding_dim,symmetric=symmetric)
            else:
                conv = MetricConv(layers[i], layers[i + 1], info=info, embedding_dim=embedding_dim,symmetric=symmetric)
            convs.append(conv)

        if classification:
            assert kwargs["num_vertices"] is not None, "To use the classification layer you must specify num_vertices."
            num_vertices = kwargs["num_vertices"]
            # If classification network then instantiate two linear layers used at end of network to convert to logits
            self.fc1 = nn.Linear(layers[-2] * num_vertices, layers[-1])
            self.fc2 = nn.Linear(layers[-1], out_feats)

            self.reset_parameters()
        else:
            if layers[-2] == layers[-1]:
                conv = MetricResBlock(layers[i], info, embedding_dim=embedding_dim,symmetric=symmetric)
            else:
                conv = MetricConv(layers[-2], layers[-1], info=info, embedding_dim=embedding_dim,symmetric=symmetric)
            convs.append(conv)
            convs.append(MetricConv(layers[-1], out_feats, info=info, embedding_dim=embedding_dim,symmetric=symmetric))
        self.convs = nn.ModuleList(convs)

        self.classification = classification
        self.nonlinear = nn.ELU()

    def reset_parameters(self) -> NoReturn:
        """
        Sets initial parameters of weights follow a Uniform Xavier distribution for fully connected matrices and
        a uniform distribution for biases.
        """
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(
        self,
        features: FloatTensor,
        vertices: FloatTensor,
        edges: LongTensor,
        faces: LongTensor,
        eps: float = 1e-5,
    ) -> FloatTensor:
        r"""
        :param features: Input features per vertex
        :param vertices: Positions of vectors in \mathbf{R}^3
        :param edges: Edge connectivity of vertices
        :param faces: Face indices of vertices
        :param eps: Small epsilon used to avoid division by 0

        :return: Returns tensor with ``out_feats`` features for each vertex
        """
        self.metric_per_vertex = []

        x = features

        for conv in self.convs:
            x = conv(x, vertices, edges, faces)
            x = self.nonlinear(x)
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
            self.metric_per_vertex.append(conv.metric_per_vertex)

        if self.classification:
            x = x.flatten()
            x = self.fc1(x)
            x = self.nonlinear(x)
            x = F.dropout(x, training=self.training)

            out = self.fc2(x)
        else:
            out = self.conv6(x, vertices, edges, faces)
            self.metric_per_vertex.append(self.conv6.metric_per_vertex)

        return out