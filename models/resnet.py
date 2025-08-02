from torch import FloatTensor, LongTensor, nn

from .metric_conv import DeformableMetricConv
from .blocks import MetricResBlock


class MetricResNet(nn.Module):
    r"""
    Simple network with kwargs['n_layers'] ``MetricResBlock`` residual blocks preceded and followed by ``MetricConv`` layers.

    :param in_channels: Number of input features per vertex
    :param out_channels: Number of output features per vertex
    :param \**kwargs: See below
    :Keyword Arguments:
        * n_layers (``int``) -- Number of residual blocks
        * info (``str``) -- type of ``MetricConv`` to use
        * n_hidden (``int``) -- Number of hidden layers used for for residual blocks
        * embedding_dim (``int``) -- Embedding dimension of metric tensor used in ``MetricConv``
    """

    def __init__(self, in_feats: int, out_feats: int, **kwargs):
        super(MetricResNet, self).__init__()
        n_layers = kwargs["n_layers"] if "n_layers" in kwargs.keys() else 8
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        n_hidden = kwargs["n_hidden"] if "n_hidden" in kwargs.keys() else 64
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True

        self.conv1 = DeformableMetricConv(in_feats, n_hidden, info=info, embedding_dim=embedding_dim,symmetric=symmetric)

        # Instantiate the residual blocks
        res_blocks = []
        for _ in range(n_layers):
            res_blocks.append(MetricResBlock(n_hidden, info, embedding_dim=embedding_dim,symmetric=symmetric, deformable=True))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.conv2 = DeformableMetricConv(n_hidden, out_feats, info=info, embedding_dim=embedding_dim,symmetric=symmetric)

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

        :return: Returns tensor with ``out_feats`` features for each vertex
        """

        self.metric_per_vertex = []
        self.vertex_deltas = []

        x = features

        x, vertex_delta = self.conv1(x, vertices, edges, faces)
        vertices = vertices + vertex_delta
        self.vertex_deltas.append(vertex_delta)

        x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
        x = self.nonlinear(x)
        self.metric_per_vertex.append(self.conv1.metric_per_vertex)

        for i in range(len(self.res_blocks)):
            x, vertex_delta = self.res_blocks[i](x, vertices, edges, faces)
            vertices = vertices + vertex_delta
            self.vertex_deltas.append(vertex_delta)
            self.metric_per_vertex.append(self.res_blocks[i].metric_per_vertex)

        out, vertex_delta = self.conv2(x, vertices, edges, faces)
        self.metric_per_vertex.append(self.conv2.metric_per_vertex)

        return out