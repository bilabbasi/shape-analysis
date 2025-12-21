# MetricConv: An adaptive convolutional neural network for graphs and meshes
<img src="imgs/all_metrics01.png" align="center">

## Setup with `uv`

This project is managed with `uv`, a fast Python package installer and resolver. The following steps will guide you through setting up the environment and installing dependencies.

**1. Clone the repo and navigate into it.**
```bash
git clone https://github.com/eidosmontreal/shape-analysis.git
cd shape-analysis
```

**2. Install `uv` (if not already on your system).**
We recommend installing `uv` via its standalone installation script:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Make sure `uv` is available in your `PATH`. You may need to source your shell profile (`source ~/.bashrc`, `source ~/.zshrc`, etc.) or add `~/.local/bin` to your `PATH` manually.

**3. Create the virtual environment.**
This command creates a virtual environment in a `.venv` directory.
```bash
uv venv --seed
```

**4. Install dependencies.**
This command installs the project in "editable" mode (`-e`) and installs all dependencies from `pyproject.toml`. This makes local modules like `train` and `models` importable without any `PYTHONPATH` manipulation.
```bash
uv pip install -e .
```

**Note on `torch-scatter`:** This project requires `torch-scatter`, which needs `torch` to be available during its build process. This is handled automatically by the configuration in `pyproject.toml` under the `[tool.uv.extra-build-dependencies]` section.


## MetricConv
<img src="imgs/meshes.png" align="center">

MetricConv builds metric tensors at each vertex depending on local geometric statistics (refer to picture above). The type of local geometric statistics may be specified via the `info` parameter upon initialization of a `MetricConv` module. These metric tensors are used to determine local distances which are then used to construct attention matrices for graph/mesh convolution.  

------------------------------------------------------------------------
### Using MetricConv
After installing the required dependencies, it's easy to start using `MetricConv`. Initialization of a `MetricConv` module requires the number of input and output features, and most notably the type of metric to use (chosen among `vanilla`, `tangent`, `face`, `feature`), specified by the `info` parameter. A typical forward pass requires the input features, positions, edges, and faces. 

We refer the user to the example below.

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.io import read_off
from torch_geometric.transforms import FaceToEdge

from models import MetricConv

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.cnn1 = MetricConv(3,32,info='face')
        self.cnn2 = MetricConv(32,3,info='face')
    def forward(self,feats,verts,edges,faces):
        x = self.cnn1(feats,verts,edges,faces)
        x = F.elu(x)
        out = self.cnn2(x,verts,edges,faces)
        return out

face_to_edge = FaceToEdge(False)
mesh = face_to_edge(read_off('meshes/chair.off'))
net = Net()
features = torch.ones(len(mesh.pos),3)
out = net(features,mesh.pos,mesh.edge_index,mesh.face.t())
```

Also included in `models/` is `architectures.py` which contains predefined architectures that comprise ``MetricConv`` blocks.

### Building your own Metric
It is also possible to include your own metric (see `models/metric.py`) to be used in `MetricConv`. Create a class inheriting the `Metric` class, and override the `compute_features` method with your own desired mesh features (to be used to construct the local tensor). Make sure to update the `info_to_metric` dictionary in `metric_conv.py` with your new metric.

## Training/Experiments

### Data
We test our model on the segmentation and correspondence task. Below are some details regarding which datasets were used.

* *FAUST.* For the correspondence task, we used the Fine Alignment Using Scan Texture ([FAUST](http://faust.is.tue.mpg.de/)) dataset. We rely on the [FAUST dataset module](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/faust.py) in *PyTorch Geometric* for handling the dataset.
Download the dataset from [here](http://faust.is.tue.mpg.de/) and move it to `data/FAUST/raw` (which may need to be manually created).

* *COSEG.* For the segmentation task, we use the [Shape COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm) dataset.To download and install the COSEG dataset, run the following command, which will download, split, and process labels of the COSEG dataset (respectively):
    ```
    python preprocess/install_coseg.py --data-dir data/COSEG --ratio 0.85
    ```
To use the exact train/test split used in our experiments, add the flag `--splits preprocess/coseg/splits.yaml` to the above command.
### Training
One can find training scripts for both the correspondence and segmentation tasks in `train`, with the appropriately labeled files. Details on the arguments used for the training scripts can be found in `train/train_args.py`, or by running, for example, the following command in the console:

```
uv run python -m train.segmentation -h
```

### Experiments
After installing the data as described in above, one can run sample training experiments found in `experiments/`. For example, to train a model for the correspondence task on the FAUST dataset, one can run: 
```
uv run python -m train.correspondence --yaml experiments/faust_correspondence.yml
```

### Demos
If you would like to visualize the results of a trained model, you can do so with the `demo.py` script in `scripts/`. For example, if you have saved a model whose weights are stored in `path/to/log` and the datasets are stored in `data`, you can run:
```
uv run -m scripts.demo --root path/to/log --data-dir data
```
which will store comparisons between ground truth and predictions (from the trained model) in `path/to/log/samples`.

## Tests
One may test basic functionalities of this repo using [pytest](https://docs.pytest.org/en/stable/). In particular, after installing `pytest` (`$ pip install -U pytest
`), run the following:
```
uv run -m pytest tests
```
