from typing import NoReturn

import torch
from torch import FloatTensor, LongTensor, nn
from torch.nn import functional as F

from .metric_conv import MetricConv, DeformableMetricConv


class MetricConvBlock(nn.Module):
    r"""
    Residual block using ``MetricConv`` layer for forward propagation. The computation may be expressed as:

                                        :math:y=ELU(MetricConv(x)

    :param in_channels: Number of hidden units used in residual layer
    :param out_channels: Number of hidden units used in residual layer
    :param info: Which metric tensor to use
    :param metric_in_feats: Number of input features for ``MetricConv``
    :param metric_n_hidden: Number of hidden parameters for ``MetricConv``
    :param embedding_dim: Embedding dimension of metric tensor used in ``MetricConv``
    :param symmetric: Boolean indicating symmetry of metric used in ``MetricConv``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        info: str = "tangent",
        metric_n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
        deformable: bool = False
    ):
        super(MetricConvBlock, self).__init__()
        # Initialize MetricConv CNN operator
        if not deformable:
            self.conv = MetricConv(
                in_channels,
                out_channels,
                info=info,
                metric_n_hidden=metric_n_hidden,
                embedding_dim=embedding_dim,
                symmetric=symmetric,
            )
        else:
            self.conv = DeformableMetricConv(
                in_channels,
                out_channels,
                info=info,
                metric_n_hidden=metric_n_hidden,
                embedding_dim=embedding_dim,
                symmetric=symmetric,
            )

        self.nonlinear = nn.ELU()

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

        :return: Returns tensor with ``n_hidden`` features for each vertex
        """
        x, delta_vert = self.conv(features, vertices, edges, faces)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        self.metric_per_vertex = self.conv.metric_per_vertex
        return x, delta_vert


class MetricResBlock(nn.Module):
    r"""
    Residual block using ``MetricConv`` layer for forward propagation. The computation may be expressed as:

                                        :math:y=0.5*(ELU(MetricConv(x))+x)

    :param n_hidden: Number of hidden units used in residual layer
    :param info: Which metric tensor to use
    :param metric_in_feats: Number of input features for ``MetricConv``
    :param metric_n_hidden: Number of hidden parameters for ``MetricConv``
    :param embedding_dim: Embedding dimension of metric tensor used in ``MetricConv``
    :param symmetric: Boolean indicating symmetry of metric used in ``MetricConv``
    """

    def __init__(
        self,
        n_hidden: int,
        info: str = "tangent",
        metric_n_hidden: int = 32,
        embedding_dim: int = 3,
        symmetric: bool = True,
        deformable: bool = True
    ):
        super(MetricResBlock, self).__init__()
        # Initialize MetricConv CNN operator
        if not deformable:
            self.conv = MetricConv(
                n_hidden,
                n_hidden,
                info=info,
                metric_n_hidden=metric_n_hidden,
                embedding_dim=embedding_dim,
                symmetric=symmetric,
            )
        else:
            self.conv = DeformableMetricConv(
                n_hidden,
                n_hidden,
                info=info,
                metric_n_hidden=metric_n_hidden,
                embedding_dim=embedding_dim,
                symmetric=symmetric,
            )
        self.nonlinear = nn.ELU()

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

        :return: Returns tensor with ``n_hidden`` features for each vertex
        """
        residual = features.clone()  # Store original features to be added back as the residual
        x, vertex_delta = self.conv(features, vertices, edges, faces)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        out = (x + residual) / 2  # Add back residual and divide by 2 for average
        self.metric_per_vertex = self.conv.metric_per_vertex
        return out, vertex_delta