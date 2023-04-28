import torch
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utility.efficient_allocation import delete_agent, get_v, get_v_sum_but_i, oracle

class Trainer(object):
    def __init__(self, bs, n, m):
        self.bs = bs
        self.n = n
        self.m = m
        self.num_mis = 1
        self.device = 'cuda:0'
        self.mode = 'train'
        self.gd_lr = 0.1
        self.gd_lr_step = 1
        self.gd_iter = 500
        self.init_componenents()

    def init_componenents(self):
        
        self.create_constants()

        self.create_params_to_train()


        self.create_optimizers()

        self.create_masks()

    def create_constants(self):
        self.x_shape = dict()
        self.x_shape["train"] = [self.bs, self.n, self.m]
        self.x_shape["val"] = [self.bs, self.n, self.m]

        self.adv_shape = dict()
        self.adv_shape["train"] = [
            self.n,
            self.num_mis,
            self.bs,
            self.n,
            self.m,
        ]
        self.adv_shape["val"] = [
            self.n,
            self.num_mis,
            self.bs,
            self.n,
            self.m,
        ]

        self.adv_var_shape = dict()
        self.adv_var_shape["train"] = [
            self.num_mis,
            self.bs,
            self.n,
            self.m,
        ]
        self.adv_var_shape["val"] = [
            self.num_mis,
            self.bs,
            self.n,
            self.m,
        ]

        self.u_shape = dict()
        self.u_shape["train"] = [
            self.n,
            self.num_mis,
            self.bs,
            self.n,
        ]
        self.u_shape["val"] = [
            self.n,
            self.num_mis,
            self.bs,
            self.n,
        ]

    def create_params_to_train(self, train=True, val=True):
        # Trainable variable for find best misreport using gradient by inputs
        self.adv_var = dict()
        if train:
            self.adv_var["train"] = torch.zeros(
                self.adv_var_shape["train"], requires_grad=True, device=self.device
            ).float()
        if val:
            self.adv_var["val"] = torch.zeros(self.adv_var_shape["val"], requires_grad=True, device=self.device).float()

    def create_optimizers(self, train=True, val=True):
        # Optimizer for best misreport find
        self.opt2 = dict()
        if train:
            self.opt2["train"] = optim.Adam([self.adv_var["train"]], self.gd_lr)
        if val:
            self.opt2["val"] = optim.Adam([self.adv_var["val"]], self.gd_lr)

        self.sc_opt2 = dict()
        if train:
            self.sc_opt2["train"] = optim.lr_scheduler.StepLR(self.opt2["train"], 1, self.gd_lr_step)
        if val:
            self.sc_opt2["val"] = optim.lr_scheduler.StepLR(self.opt2["val"], 1, self.gd_lr_step)

    def create_masks(self, train=True, val=True):
        self.adv_mask = dict()
        if train:
            self.adv_mask["train"] = np.zeros(self.adv_shape["train"])
            self.adv_mask["train"][np.arange(self.n), :, :, np.arange(self.n), :] = 1.0
            self.adv_mask["train"] = tensor(self.adv_mask["train"]).float()
            
        if val:
            self.adv_mask["val"] = np.zeros(self.adv_shape["val"])
            self.adv_mask["val"][np.arange(self.n), :, :, np.arange(self.n), :] = 1.0
            self.adv_mask["val"] = tensor(self.adv_mask["val"]).float()

        self.u_mask = dict()
        if train:
            self.u_mask["train"] = np.zeros(self.u_shape["train"])
            self.u_mask["train"][np.arange(self.n), :, :, np.arange(self.n)] = 1.0
            self.u_mask["train"] = tensor(self.u_mask["train"]).float()

        if val:
            self.u_mask["val"] = np.zeros(self.u_shape["val"])
            self.u_mask["val"][np.arange(self.n), :, :, np.arange(self.n)] = 1.0
            self.u_mask["val"] = tensor(self.u_mask["val"]).float()

    def mis_step(self, x):
        """
        Find best misreport step using gradient by inputs, trainable inputs: self.adv_var variable
        """
        mode = self.mode

        self.opt2[mode].zero_grad()

        # Get misreports
        x_mis, misreports = self.get_misreports_grad(x)
        #misreports = torch.full(misreports.size(), 1000000).float().to(self.device)

        # Run net for misreports
        a_mis, p_mis = self.net(misreports)
        # Calculate utility
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Calculate loss value
        #print("----")
        #print(utility_mis.size())
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()
        
        #print(self.u_shape[mode])
        #print(self.u_mask[mode].size())
        #print(u_mis)
        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()

    def compute_utility(self, x, alloc, pay):
        """Given seller costs (x), payment (pay) and allocation (alloc), computes utility
        Input params:
            x: [num_batches, num_agents, num_items]
            a: [num_batches, num_agents, num_items]
            p: [num_batches, num_agents]
        Output params:
            utility: [num_batches, num_agents]
        """
        return pay - (alloc * x).sum(dim=-1)

    def compute_regret(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        return self.compute_regret_grad(x, a_true, p_true, x_mis, a_mis, p_mis)

    def compute_regret_grad(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        mode = self.mode

        utility = self.compute_utility(x, a_true, p_true)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        utility_true = utility.repeat(self.n * self.num_mis, 1)
        excess_from_utility = F.relu(
            (utility_mis - utility_true).view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)
        )

        rgt = excess_from_utility.max(3)[0].max(1)[0].mean(dim=1)
        return rgt

    def get_misreports(self, x):
        return self.get_misreports_grad(x)

    def get_misreports_grad(self, x):
        mode = self.mode
        adv_mask = self.adv_mask[mode].to(self.device)

        adv = self.adv_var[mode].unsqueeze(0).repeat(self.n, 1, 1, 1, 1)
        x_mis = x.repeat(self.n * self.num_mis, 1, 1)
        x_r = x_mis.view(self.adv_shape[mode])
        y = x_r * (1 - adv_mask) + adv * adv_mask
        misreports = y.view([-1, self.n, self.m])

        return x_mis, misreports

    def train_regret(self):

        # Get new batch. X - true valuation, ADV - start point for misreport candidates
        # perm - ADV positions in full train dataset
        X, ADV = torch.rand(self.bs, self.n, self.m), torch.rand(self.bs, self.n, self.m)

        x = X.float().to(self.device)

        # Write start adv value for find best misreport variable
        self.adv_var["train"].data = ADV.float().to(self.device)

        self.misreport_cycle(x)

    def misreport_cycle(self, x):
        mode = self.mode

        # Find best misreport cycle
        for i in range(self.gd_iter):
            # Make a gradient step, update self.adv_var variable
            self.mis_step(x)

        for param_group in self.opt2.param_groups:
            param_group["lr"] = self.gd_lr

        self.opt2.state = defaultdict(dict)  # reset momentum


    def net(self, batch):
        """
        :param batch: tensor with bids (supplier costs) in auctions shaped as (batch_size, n_agents, num_items)
        item-wise vickery auction
        """
        print(batch.size())
        alloc = torch.zeros(batch.size())
        pay = torch.zeros([batch.size(dim=0), batch.size(dim=1)])
        for i in range(self.bs):
            auction = batch[i,:,:]
            for item in range(self.m):
                item_bids = auction[:,item]
                win = torch.argmin(item_bids)
                alloc[i,win,item]=1
                pay[i,win] += float(torch.sort(item_bids).values[1])
        return alloc.to(self.device), pay.to(self.device)

n_auctions, n_agents, n_items = 10000, 2, 1
trainer = Trainer(n_auctions, n_agents, n_items)
trainer.train_regret()
