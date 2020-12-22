from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
from pathlib import Path
import time
from typing import Dict, List, NamedTuple, Optional, Tuple


class DatasetType(Enum):
  CIFAR10 = (1, (3, 32, 32), 10)
  SVHN = (2, (3, 32, 32), 10)
  
  def __init__(self, id: int, image_shape: Tuple[int, int, int], num_classes: int):
    self.D = image_shape
    self.K = num_classes

class DatasetSubsetType(IntEnum):
  TRAIN = 0
  TEST = 1

class ComplexityType(Enum):
  # Measures from Fantastic Generalization Measures (equation numbers)
  PARAMS = 20
  INVERSE_MARGIN = 22
  LOG_SPEC_INIT_MAIN = 29
  LOG_SPEC_ORIG_MAIN = 30
  LOG_PROD_OF_SPEC_OVER_MARGIN = 31
  LOG_PROD_OF_SPEC = 32
  FRO_OVER_SPEC = 33
  LOG_SUM_OF_SPEC_OVER_MARGIN = 34
  LOG_SUM_OF_SPEC = 35
  LOG_PROD_OF_FRO_OVER_MARGIN = 36
  LOG_PROD_OF_FRO = 37
  LOG_SUM_OF_FRO_OVER_MARGIN = 38
  LOG_SUM_OF_FRO = 39
  FRO_DIST = 40
  DIST_SPEC_INIT = 41
  PARAM_NORM = 42
  PATH_NORM_OVER_MARGIN = 43
  PATH_NORM = 44
  PACBAYES_INIT = 48
  PACBAYES_ORIG = 49
  PACBAYES_FLATNESS = 53
  PACBAYES_MAG_INIT = 56
  PACBAYES_MAG_ORIG = 57
  PACBAYES_MAG_FLATNESS = 61
  # Other Measures
  L2 = 100
  L2_DIST = 101
  # FFT Spectral Measures
  LOG_SPEC_INIT_MAIN_FFT = 129
  LOG_SPEC_ORIG_MAIN_FFT = 130
  LOG_PROD_OF_SPEC_OVER_MARGIN_FFT = 131
  LOG_PROD_OF_SPEC_FFT = 132
  FRO_OVER_SPEC_FFT = 133
  LOG_SUM_OF_SPEC_OVER_MARGIN_FFT = 134
  LOG_SUM_OF_SPEC_FFT = 135
  DIST_SPEC_INIT_FFT = 141

class OptimizerType(Enum):
  SGD = 1
  SGD_MOMENTUM = 2
  ADAM = 3

class Verbosity(IntEnum):
  NONE = 1
  RUN = 2
  EPOCH = 3
  BATCH = 4

@dataclass(frozen=False)
class State:
  epoch: int = 1
  batch: int = 1
  global_batch: int = 1
  converged: bool = False
  ce_check_freq: int = 0
  ce_check_milestones: Optional[List[float]] = None

# Hyperparameters that uniquely determine the experiment
@dataclass(frozen=True)
class HParams:
  seed: int = 0
  use_cuda: bool = True
  # Model
  model_depth: int = 2
  model_width: int = 8
  base_width: int = 25
  # Dataset
  dataset_type: DatasetType = DatasetType.CIFAR10
  data_seed: Optional[int] = 42
  train_dataset_size: Optional[int] = None
  test_dataset_size: Optional[int] = None
  # Training
  batch_size: int = 32
  epochs: int = 300
  optimizer_type: OptimizerType = OptimizerType.SGD_MOMENTUM
  lr: float = 0.01
  # Cross-entropy stopping criterion
  ce_target: Optional[float] = 0.01
  ce_target_milestones: Optional[List[float]] = field(default_factory=lambda: [0.05, 0.025, 0.015])

  def to_tensorboard_dict(self) -> dict:
    d = asdict(self)
    d = {x: y for (x,y) in d.items() if y is not None}
    d = {x:(y.name if isinstance(y, Enum) else y) for x,y in d.items()}
    return d
  
  @property
  def md5(self):
    return hashlib.md5(str(self).encode('utf-8')).hexdigest()
  
  @property
  def wandb_md5(self):
    dictionary = self.to_tensorboard_dict()
    dictionary['seed'] = 0
    return hashlib.md5(str(dictionary).encode('utf-8')).hexdigest()

# Configuration which doesn't affect experiment results
@dataclass(frozen=True)
class Config:
  id: int = field(default_factory=lambda: time.time_ns())
  log_batch_freq: Optional[int] = None
  log_epoch_freq: Optional[int] = 10
  save_epoch_freq: Optional[int] = 1
  root_dir: Path = Path('./temp')
  data_dir: Path = Path('./temp/data')
  verbosity: Verbosity = Verbosity.EPOCH
  use_tqdm: bool = False

  def setup_dirs(self) -> None:
    # Set up directories
    for directory in ('results', 'checkpoints'):
      (self.root_dir / directory).mkdir(parents=True, exist_ok=True)
    self.data_dir.mkdir(parents=True, exist_ok=True)

  @property
  def checkpoint_dir(self):
    return self.root_dir / 'checkpoints'
  
  @property
  def results_dir(self):
    return self.root_dir / 'results'

class EvaluationMetrics(NamedTuple):
  acc: float
  avg_loss: float
  num_correct: int
  num_to_evaluate_on: int
  all_complexities: Dict[ComplexityType, float]
