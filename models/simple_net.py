from torch import FloatTensor, LongTensor, nn

from .blocks import MetricConvBlock


class SimpleNet(nn.Module):
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
        super(SimpleNet, self).__init__()
        info = kwargs["info"] if "info" in kwargs.keys() else "vanilla"
        embedding_dim = kwargs["embedding_dim"] if "embedding_dim" in kwargs.keys() else 8
        symmetric = kwargs["symmetric"] if "symmetric" in kwargs.keys() else True
        layers = kwargs["layers"] if "layers" in kwargs.keys() else [16, 32, 64, 128, 256]
        layers = [in_feats] + layers

        self.conv1 = MetricConvBlock(3, 256, info=info, embedding_dim=embedding_dim,symmetric=symmetric, deformable=True)
        self.conv2 = MetricConvBlock(256, 256, info=info, embedding_dim=embedding_dim,symmetric=symmetric, deformable=True)
        self.conv3 = MetricConvBlock(256, 256, info=info, embedding_dim=embedding_dim,symmetric=symmetric, deformable=True)
        self.conv4 = MetricConvBlock(256, out_feats, info=info, embedding_dim=embedding_dim,symmetric=symmetric, deformable=True)

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

        conv1_output, vertex_delta1 = self.conv1(features, vertices, edges, faces)
        vertices1 = vertices + vertex_delta1
        self.vertex_deltas.append(vertex_delta1)
        self.metric_per_vertex.append(self.conv1.metric_per_vertex)

        conv2_output, vertex_delta2 = self.conv2(conv1_output, vertices1, edges, faces)
        vertices2 = vertices1 + vertex_delta2
        self.vertex_deltas.append(vertex_delta2)
        self.metric_per_vertex.append(self.conv2.metric_per_vertex)

        conv3_output, vertex_delta3 = self.conv3(conv2_output, vertices2, edges, faces)
        vertices3 = vertices2 + vertex_delta3
        self.vertex_deltas.append(vertex_delta3)
        self.metric_per_vertex.append(self.conv3.metric_per_vertex)

        out, _ = self.conv4(conv3_output, vertices3, edges, faces)

        return out