import argparse
from collections import ChainMap
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import autograd, nn, optim
import wandb


os.environ["WANDB_SILENT"] = "true"


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Robust Regression')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--steps', type=int, default=1)

parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--rex', type=str2bool, default=False)
parser.add_argument('--weight_init', type=float, default=0.001)

# Pertinent params
parser.add_argument('--only_bias__ignore_input', type=str2bool, default=False)
parser.add_argument('--bias', type=str2bool, default=True)
parser.add_argument('--nonnegative_weights_only', type=str2bool, default=True)
parser.add_argument('--selected_single_measure', type=str, default="path_norm")
parser.add_argument('--env_split', type=str, default="all")
parser.add_argument('--exp_type', type=str, default="v1")
parser.add_argument('--wandb_tag', type=str, default="default")
parser.add_argument('--datafile', type=str, default='nin.cifar10_svhn')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--datasize', type=int, default=None)

flags = parser.parse_args()


df_ob = pd.read_csv(f'data/{flags.datafile}.csv').replace([np.inf, -np.inf], np.nan).dropna()
if flags.dataset is not None:
  assert flags.dataset in df_ob['hp.dataset'].unique()
  df_ob = df_ob[df_ob['hp.dataset']==flags.dataset]
if flags.datasize is not None:
  assert flags.datasize in df_ob['hp.train_dataset_size'].unique()
  df_ob = df_ob[df_ob['hp.train_dataset_size']==flags.datasize]


BLACKLISTED_MEASURES = set([x for x in df_ob.columns if x.startswith('complexity')])
measure = f"complexity.{flags.selected_single_measure}"
if f"{measure}_fft" in BLACKLISTED_MEASURES:
  measure = f"{measure}_fft"
BLACKLISTED_MEASURES.remove(measure)


# Utility functions
def get_complexity_measures(data):
    return [c for c in data.columns if c.startswith("complexity.") and c not in BLACKLISTED_MEASURES]


# Scale complexity measure so that mean is 0 and standard deviation is 1
for c in get_complexity_measures(df_ob):
  df_ob[c] = StandardScaler().fit_transform(df_ob[c].values.reshape(-1, 1)).reshape(-1,)

num_measures = len(get_complexity_measures(df_ob))

n_runs = 1
runs_ood = -1*np.ones((n_runs, num_measures))
runs_val = -1*np.ones((n_runs, num_measures))
runs_train = -1*np.ones((n_runs, num_measures))
runs_weights = -1*np.ones((n_runs, num_measures))
runs_biases = -1*np.ones((n_runs, num_measures))
runs_zeros = 0*np.ones((n_runs, num_measures))


def mean_mse(logits, y):
    logits = logits.squeeze(-1)
    return torch.nn.functional.mse_loss(logits, y)

def penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_()
    loss = mean_mse(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

hp_dict = {"lr": 0, "width": 1, "depth": 2, "dataset": 3, "train_size": 4}

r = 0

def load_data(df, _filter=None, _comp=None, env_split=None):
  df = df[df[_comp].notna()]

  gen_gap = (df['gen.train_acc'].to_numpy() - df['gen.val_acc'].to_numpy())

  lr = df['hp.lr'].to_numpy()
  width = df['hp.model_width'].to_numpy()
  depth = df['hp.model_depth'].to_numpy()
  dataset = df['hp.dataset'].to_numpy()
  train_size = df['train_dataset_size'].to_numpy()

  if _comp != None:
    comp = df[_comp].to_numpy()
    X = np.transpose(np.stack([comp]), (1,0))
    env_specs = np.transpose(np.stack([lr, width, depth, dataset, train_size]), (1,0))
  Y = gen_gap

  d = {}
  envs = {}
  envs_oos = {}
  for idx, i in enumerate(zip(env_specs, X, Y)):
    if flags.exp_type == "v1":
      k = ','.join([f'{x[0]}:{x[1]}' for x in zip(hp_dict.keys(),i[0])])
    elif flags.exp_type == 'v2':
      k = ','.join([f'{x[0]}:{x[1]}' for x in zip(hp_dict.keys(),i[0]) if x[0] != env_split])
    elif flags.exp_type == 'v3':
      k = f'{env_split}:{i[0][hp_dict[env_split]]}'
    else:
      raise RuntimeError
    vx = torch.from_numpy(i[1]).float()
    vy = torch.tensor([i[2]]).float()

    if k not in d:
      d[k] = [[vx], [vy]]
    else:
      d[k][0].append(vx)
      d[k][1].append(vy)

  for _d in d.items():
    k = _d[0]
    vx = torch.stack(_d[1][0])
    vy = torch.stack(_d[1][1]).squeeze(1)

    envs[k] = {"X":vx, "Y":vy}
    envs_oos[k] = {"X":None, "Y":None}

  return envs, envs_oos, {}

cms = get_complexity_measures(df_ob)

if flags.env_split == "all" and flags.exp_type in {'v2', 'v3'}:
  data_obs = [[load_data(df_ob, _comp=cm, env_split=e_s) for e_s in hp_dict.keys()] for cm in cms]
  data_obs = [tuple(dict(ChainMap(*[i[ii][iii] for ii in range(len(hp_dict))])) for iii in range(len(i[0]))) for i in data_obs]
else:
  data_obs = [load_data(df_ob, _comp=cm, env_split=flags.env_split) for cm in cms]

def combine_env(envs):
  X = torch.cat([e['X'] for e in envs.values()])
  Y = torch.cat([e['Y'] for e in envs.values()])
  return X, Y

def estimator(envs, model, seed=99999, model_type="rf", _idx=None, name=None):
    """
    Fits an estimator of generalization quantity given 
    complexity measure

    """

    #wandb.watch(model, log='all')
    wandb.init(project='rgm_single', reinit=True, tags=[flags.wandb_tag])
    wandb.config.update(flags)
    wandb.config.actual_measure = name

    if flags.optim == 'adam':
      optimizer = optim.Adam(model.parameters(), lr=flags.lr)
    elif flags.optim == 'sgdm':
      optimizer = optim.SGD(model.parameters(), lr=flags.lr, momentum=0.9)
    else:
      optimizer = optim.SGD(model.parameters(), lr=flags.lr)

    risks = None
    for step in range(flags.steps):
      for env in envs.values():
        logits = model(env['X'])
        env['mse'] = mean_mse(logits, env['Y'])
        env['irm_penalty'] = penalty(logits, env['Y'])

      risks = torch.stack([e['mse'] for e in envs.values()])

      risk_weightings = (~torch.lt(risks, risks.max())).float().detach()
      robustness_penalty = (risks * risk_weightings).mean()

      train_mse = risks.mean()
      rex_penalty = risks.var()
      irmv1_penalty = torch.stack([e['irm_penalty'] for e in envs.values()]).mean()

      weight_norm = torch.tensor(0.)
      for w in model.parameters():
        weight_norm += w.norm().pow(2)

      # minmax
      loss = robustness_penalty

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if model.w.data.numpy().item() < 0.0:
        if flags.nonnegative_weights_only:
          model.w.data = torch.Tensor([[0.0]])
        runs_zeros[r][_idx] = np.ones((1,))

      wandb.log({'loss': loss.cpu().item(),
                  'train_mse': train_mse.cpu().item(),
                  'irmv1_penalty': irmv1_penalty.cpu().item(),
                  'rex_penalty': rex_penalty.cpu().item(),
                  'robustness_penalty': robustness_penalty.cpu().item(),
                  'risk_argmax': risks.argmax().cpu().item(),
                  'risk_max': risks.max().cpu().item(),
                  'risk_min': risks.min().cpu().item(),
                  'risk_range': (risks.max() - risks.min()).cpu().item(),
                  'weight': model.w.squeeze().cpu().item(),
                  'weight_grad': model.w.grad.squeeze().cpu().item(),
                  'bias': model.b.squeeze().cpu().item() if flags.bias else 0.0,
                  'bias_grad': model.b.grad.squeeze().cpu().item() if flags.bias else 0.0,
                })
    wandb.join()
    np.save(f'temp/single_network/risks/{flags.selected_single_measure}__{flags.env_split}__{flags.exp_type}__{flags.only_bias__ignore_input}.npy', risks.detach().numpy())
    return model

data_train = [combine_env(data_obs[idx][0]) for idx, _ in enumerate(cms)]

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.w = torch.nn.Parameter(torch.Tensor([[flags.weight_init]]))

    if flags.bias:
      self.b = torch.nn.Parameter(torch.zeros(1,))

  def forward(self, input):
    #out = input @ torch.exp(self.w)
    w = self.w
    if flags.only_bias__ignore_input:
      if not flags.bias:
        raise RuntimeError
      w = w * 0.0

    out = input @ w
    if flags.bias:
      out = out + self.b
    return out

gen_gap_stds = [data_train[idx][1].data.numpy().std().item() for idx, _ in enumerate(cms)]
complexity_measure_stds = [data_train[idx][0].data.numpy().std().item() for idx, _ in enumerate(cms)]

start_time = time.time()
baseline_models = [estimator(data_obs[idx][0], MLP(), seed=r*9+idx, _idx=idx, name=name) for idx, name in enumerate(cms)]

baseline_ood_mses = [torch.Tensor([-1.0]).reshape(()) for idx, _ in enumerate(cms)]
baseline_val_mses = [torch.Tensor([-1.0]).reshape(()) for idx, _ in enumerate(cms)]

baseline_train_mses = [torch.mean((data_train[idx][1] - baseline_models[0](data_train[idx][0]))**2) for idx, _ in enumerate(cms)]

for idx, _ in enumerate(cms):
  #runs_weights[r][idx] = baseline_models[idx].w.exp().squeeze().data.numpy()
  runs_weights[r][idx] = baseline_models[idx].w.squeeze().data.numpy()
  if flags.bias:
    runs_biases[r][idx] = baseline_models[idx].b.squeeze().data.numpy()

  runs_ood[r][idx] = baseline_ood_mses[idx]
  runs_val[r][idx] = baseline_val_mses[idx]
  runs_train[r][idx] = baseline_train_mses[idx]
