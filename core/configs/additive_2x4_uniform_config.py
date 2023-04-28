import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Auction params
__C.num_agents = 2
__C.num_items = 4

# EquivariantNet
__C.net.n_exch_layers = 3

# RegretFormer
__C.net.hid_att = 32
__C.net.hid = 64
__C.net.n_attention_layers = 1
__C.net.n_attention_heads = 1
__C.net.activation_att = 'tanh'
__C.net.pos_enc = False
__C.net.pos_enc_part = 1
__C.net.pos_enc_item = 1

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_01"
__C.min = 0
__C.max = 1

