import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from core.plot import plot


class Trainer(object):
    def __init__(self, configuration, net, clip_op_lambda, device):
        self.net = net
        self.config = configuration
        self.device = device
        self.mode = "train"
        self.clip_op_lambda = clip_op_lambda
        self.var = 0
        self.writer = SummaryWriter(self.config.save_data)

        self.init_componenents()

    def init_componenents(self):
        self.create_constants()

        

        self.create_params_to_train()

        self.create_optimizers()

        self.create_masks()

        self.save_config()

    def create_constants(self):
        self.x_shape = dict()
        self.x_shape["train"] = [self.config.train.batch_size, self.config.num_agents, self.config.num_items]
        self.x_shape["val"] = [self.config.val.batch_size, self.config.num_agents, self.config.num_items]

        self.adv_shape = dict()
        self.adv_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.adv_var_shape = dict()
        self.adv_var_shape["train"] = [
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_var_shape["val"] = [
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.u_shape = dict()
        self.u_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
        ]
        self.u_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
        ]

        self.w_rgt = self.config.train.w_rgt_init_val
        self.rgt_target = self.config.train.rgt_target_start
        self.rgt_target_mult = (self.config.train.rgt_target_end / self.config.train.rgt_target_start) ** (
            1.5 / self.config.train.max_iter
        )

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
        self.opt1 = optim.Adam(self.net.parameters(), self.config.train.learning_rate)

        # Optimizer for best misreport find
        self.opt2 = dict()
        if train:
            self.opt2["train"] = optim.Adam([self.adv_var["train"]], self.config.train.gd_lr)
        if val:
            self.opt2["val"] = optim.Adam([self.adv_var["val"]], self.config.val.gd_lr)

        self.sc_opt2 = dict()
        if train:
            self.sc_opt2["train"] = optim.lr_scheduler.StepLR(self.opt2["train"], 1, self.config.train.gd_lr_step)
        if val:
            self.sc_opt2["val"] = optim.lr_scheduler.StepLR(self.opt2["val"], 1, self.config.val.gd_lr_step)

    def create_masks(self, train=True, val=True):
        self.adv_mask = dict()
        if train:
            self.adv_mask["train"] = np.zeros(self.adv_shape["train"])
            self.adv_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["train"] = tensor(self.adv_mask["train"]).float()
            
        if val:
            self.adv_mask["val"] = np.zeros(self.adv_shape["val"])
            self.adv_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["val"] = tensor(self.adv_mask["val"]).float()

        self.u_mask = dict()
        if train:
            self.u_mask["train"] = np.zeros(self.u_shape["train"])
            self.u_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["train"] = tensor(self.u_mask["train"]).float()

        if val:
            self.u_mask["val"] = np.zeros(self.u_shape["val"])
            self.u_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["val"] = tensor(self.u_mask["val"]).float()

    def save_config(self):
        print(self.writer.log_dir)
        print(type(self.config))
        with open(self.writer.log_dir + "/config.json", "w") as f:
            json.dump(self.config, f)

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
        #print(utility_mis)
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()
        #print(u_mis)
        #print(self.u_shape[mode])
        #print(self.u_mask[mode].size())
        #print(u_mis)

        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()

    def print_mechanism(self, pay, alloc, x, n):
        """print for first n data instances
        Input params:
            pay: [num_batches, num_agents]
            alloc: [num_batches, num_agents, num_items]
            x: [num_batches, num_agents, num_items]
            v: [num_batches, num_items]
            n: num instances
        """
        for i in range(n):
            print("------------------")
            print("SELLER COSTS:")
            print(x[i])
            print("ALLOCATION:")
            print(alloc[i])
            print("PAYMENT:")
            print(pay[i])


    def compute_metrics(self, x):
        """
        Validation metrics
        """
        x_mis, misreports = self.get_misreports_grad(x)
        alloc_true, pay_true = self.net(x)
        a_mis, p_mis = self.net(misreports)
        #print('----')
        #print(x_mis)
        #print(misreports)
        #print(a_mis)
        #print(p_mis)
        rgt = self.compute_regret_grad(x, alloc_true, pay_true, x_mis, a_mis, p_mis)

        revenue = self.compute_rev(pay_true)

        return revenue, rgt.mean()

    def compute_rev(self, pay):
        """Given payment (pay), computes revenue
        Input params:
            pay: [num_batches, num_agents]
        Output params:
            revenue: scalar
        """
        return pay.sum(dim=-1).mean()

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
        #print(x)
        utility = self.compute_utility(x, a_true, p_true)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
        #print(utility)
        #print(utility_mis)
        utility_true = utility.repeat(self.config.num_agents * self.config[mode].num_misreports, 1)
        excess_from_utility = F.relu(
            (utility_mis - utility_true).view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)
        )
        #print(excess_from_utility)


        rgt = excess_from_utility.max(3)[0].max(1)[0].mean(dim=1)
        return rgt

    def get_misreports(self, x):
        return self.get_misreports_grad(x)

    def get_misreports_grad(self, x):
        mode = self.mode
        adv_mask = self.adv_mask[mode].to(self.device)

        adv = self.adv_var[mode].unsqueeze(0).repeat(self.config.num_agents, 1, 1, 1, 1)
        x_mis = x.repeat(self.config.num_agents * self.config[mode].num_misreports, 1, 1)
        x_r = x_mis.view(self.adv_shape[mode])
        y = x_r * (1 - adv_mask) + adv * adv_mask
        misreports = y.view([-1, self.config.num_agents, self.config.num_items])
        #misreports[0,0,0]=0.5
        return x_mis, misreports

    def train(self, generator):
        """
        Main function, full train process
        """
        # Make a generators for train and validation
        self.train_gen, self.val_gen = generator

        load_iter = self.config.val.load_iter
        # Load save model
        if load_iter > 0:
            model_path = self.writer.log_dir + "/model_{}".format(load_iter)
            state_dict = torch.load(model_path)
            self.net.load_state_dict(state_dict)
        
        self.eval(load_iter)

    def eval(self, iteration):
        print("Validation on {} iteration".format(iteration))
        self.mode = "val"
        self.net.eval()

        self.eval_grad(iteration)

        if self.config.plot.bool:
            self.plot()

    def eval_grad(self, iteration):
        val_revenue = 0
        val_regret = 0

        for _ in range(self.config.val.num_batches):
            X, ADV, _ = next(self.val_gen.gen_func)

            self.adv_var["val"].data = tensor(ADV).float().to(self.device)

            x = torch.from_numpy(X).float().to(self.device)
            #print(x)
            self.misreport_cycle(x)

            val_revenue_batch, val_regret_batch = self.compute_metrics(x)
            val_revenue += val_revenue_batch
            val_regret += val_regret_batch
            #print("------")
            #print(val_regret_batch)

        val_revenue /= float(self.config.val.num_batches)
        val_regret /= float(self.config.val.num_batches)

        print("Val revenue: {},   regret_grad: {}".format(round(float(val_revenue), 5), round(float(val_regret), 10)))
        self.writer.add_scalar("Validation/revenue", val_revenue, iteration / 1000)
        self.writer.add_scalar("Validation/regret_grad", val_regret, iteration / 1000)

    def plot(self):
        x = np.linspace(self.config.min, self.config.max, self.config.plot.n_points)
        y = x
        if self.config.setting == 'additive_1x2_uniform_416_47':
            y = np.linspace(self.config.min, 7, self.config.plot.n_points)
        x = np.stack([v.flatten() for v in np.meshgrid(x, y)], axis=-1)
        x = np.expand_dims(x, 1)
        x = torch.FloatTensor(x)

        allocation = self.net(x.to(self.device))[0].cpu()
        allocation = (
            allocation.detach()
            .numpy()[:, 0, :]
            .reshape(self.config.plot.n_points, self.config.plot.n_points, self.config.num_items)
        )

        plot(allocation, self.config.save_data, self.config.setting)

    def misreport_cycle(self, x):
        mode = self.mode

        # Find best misreport cycle
        for i in range(self.config[mode].gd_iter):
            # Make a gradient step, update self.adv_var variable
            self.var=i
            self.mis_step(x)

            # Clipping new values of self.adv_var with respect for valuations distribution
            self.clip_op_lambda(self.adv_var[mode])

        for param_group in self.opt2[mode].param_groups:
            param_group["lr"] = self.config[mode].gd_lr

        self.opt2[mode].state = defaultdict(dict)  # reset momentum

    def save(self, iteration):
        torch.save(self.net.state_dict(), self.writer.log_dir + "/model_{}".format(iteration))

