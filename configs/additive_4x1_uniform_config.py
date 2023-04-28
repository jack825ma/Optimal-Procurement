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
__C.num_agents = 4
__C.num_items = 1

# RegretFormer params
__C.net.n_attention_layers = 6
__C.net.n_attention_heads = 6
__C.net.hid_equiv = 64
__C.net.hid_att = 64
__C.net.hid = 64

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_01"
__C.min = 0
__C.max = 1
__C.train.print_iter = 10
