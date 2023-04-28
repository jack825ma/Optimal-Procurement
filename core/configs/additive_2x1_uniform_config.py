import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Plot
__C.plot.bool = False

# Auction params
__C.num_agents = 2
__C.num_items = 1

# RegretFormer params
__C.net.n_attention_layers = 4
__C.net.n_attention_heads = 6
__C.net.hid_equiv = 128
__C.net.hid_att = 32
__C.net.hid = 128

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_01"
__C.min = 0
__C.max = 1
__C.train.print_iter = 10

__C.train.w_rgt_init_val = 1
__C.train.rgt_target_start = 0.001
__C.train.rgt_target_end = 0.001
__C.train.rgt_lr = 0.5
