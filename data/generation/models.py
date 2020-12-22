from torch import Tensor
import torch.nn as nn

from .experiment_config import DatasetType


class ExperimentBaseModel(nn.Module):
  def __init__(self, dataset_type: DatasetType):
    super().__init__()
    self.dataset_type = dataset_type

  def forward(self, x) -> Tensor:
    raise NotImplementedError


class NiNBlock(nn.Module):
  def __init__(self, inplanes: int, planes: int) -> None:
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn3 = nn.BatchNorm2d(planes)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    return x


class NiN(ExperimentBaseModel):
  def __init__(self, depth: int, width: int, base_width: int, dataset_type: DatasetType) -> None:
    super().__init__(dataset_type)

    self.base_width = base_width

    blocks = []
    blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width*width))
    for _ in range(depth-1):
      blocks.append(NiNBlock(self.base_width*width,self.base_width*width))
    self.blocks = nn.Sequential(*blocks)

    self.conv = nn.Conv2d(self.base_width*width, self.dataset_type.K, kernel_size=1, stride=1)
    self.bn = nn.BatchNorm2d(self.dataset_type.K)
    self.relu = nn.ReLU(inplace=True)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x):
    x = self.blocks(x)

    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.avgpool(x)
    
    return x.squeeze()
