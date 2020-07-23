import os
import sys
import copy
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import timm
import matplotlib.pyplot as plt

from utils.drivers import test, get_dataloader
from pruner.fp_mbnetv3 import FilterPrunerMBNetV3
from pruner.fp_resnet import FilterPrunerResNet

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels import GridInterpolationKernel, AdditiveStructureKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils import standardize

from model.resnet_cifar10 import ResNet20, ResNet32, ResNet44, ResNet56

from math import cos, pi
from torch.utils.tensorboard import SummaryWriter

import PIL

writer = None

def masked_forward(self, input, output):
    return (output.permute(0,2,3,1) * self.mask).permute(0,3,1,2)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1).mean()
        return loss

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = F.softmax(target,dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob).mean()
        return cross_entropy_loss

def set_lr(optim, lr):
    for params_group in optim.param_groups:
        params_group['lr'] = lr

def calculate_lr(initlr, cur_step, total_steps, warmup_steps):
    if cur_step < warmup_steps:
        curr_lr = initlr * (cur_step / warmup_steps)
    else:
        if args.scheduler == 'cosine_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1 + cos(pi * T / (N-1))) / 2
        elif args.scheduler == 'linear_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1-(float(T)/N))
    return curr_lr

class Hyperparams(object):
    def __init__(self, network):
        self.num_levels = 3
        self.cur_level = 1
        self.last_level = 0
        if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']:
            self.dim = [1, 6, int(args.network[6:])/2+2]

    def get_dim(self):
        return int(self.dim[self.cur_level-1])
    
    def random_sample(self):
        return np.random.rand(self.dim[self.cur_level-1]) * (args.upper_channel-args.lower_channel) + args.lower_channel
    
    def increase_level(self):
        if self.cur_level < self.num_levels:
            self.last_level = self.cur_level
            self.cur_level += 1
            return True
        return False
    
    def get_layer_budget_from_parameterization(self, parameterization, mask_pruner, soft=False):
        if not soft:
            parameterization = torch.tensor(parameterization)
            layers = len(mask_pruner.filter_ranks)

        layer_budget = torch.zeros(layers).cuda()
        if self.cur_level == 1:
            for k in range(layers):
                layer_budget[k] = torch.clamp(parameterization[0]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))

        elif self.cur_level == 2:
            if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']:
                depth = int(args.network[6:])
                stage = (depth - 2) // 3
                splits = [1]
                splits.extend([s*stage+1 for s in range(1,4)])
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                splits = np.array(splits)+1
                splits[0] = 0
                last_left = 0
                last_filter = 0
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s+3]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        layer_budget[k] = torch.clamp(last_left+parameterization[s+3]*(mask_pruner.filter_ranks[k].size(0)-last_filter), 1, mask_pruner.filter_ranks[k].size(0))
                        if k == (splits[s+1]-2):
                            last_left = layer_budget[k]
                            last_filter = mask_pruner.filter_ranks[k].size(0)
            else:
                lower = 0
                for p, upper in enumerate(mask_pruner.stages):
                    for k in range(lower, upper+1):
                        layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                    lower = upper+1

        else:
            if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']:
                depth = int(args.network[6:])
                for k in range(1, depth-2, 2):
                    layer_budget[k] = torch.clamp(parameterization[(k-1)//2]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))

                stage = (depth - 2) // 3
                splits = [1]
                splits.extend([s*stage+1 for s in range(1,4)])
                splits = np.array(splits)+1
                splits[0] = 0
                last_left = 0
                last_filter = 0
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s-3]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        layer_budget[k] = torch.clamp(last_left+parameterization[s+3]*(mask_pruner.filter_ranks[k].size(0)-last_filter), 1, mask_pruner.filter_ranks[k].size(0))
                        if k == (splits[s+1]-2):
                            last_left = layer_budget[k]
                            last_filter = mask_pruner.filter_ranks[k].size(0)

            else:
                p = 0
                for l in range(len(mask_pruner.filter_ranks)):
                    k = l
                    while k in mask_pruner.chains and layer_budget[k] == 0:
                        layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        k = mask_pruner.chains[k]
                    if layer_budget[k] == 0:
                        layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        p += 1

        if not soft:
            layer_budget = layer_budget.detach().cpu().numpy()
            for k in range(len(layer_budget)):
                layer_budget[k] = int(layer_budget[k])

        return layer_budget


class RandAcquisition(AcquisitionFunction):
    def setup(self, obj1, obj2, multiplier=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.rand = torch.rand(1) if multiplier is None else multiplier

    def forward(self, X):
        linear_weighted_sum = (1-self.rand) * (self.obj1(X)-args.baseline) + self.rand * (self.obj2(X)-args.baseline)
        return -1*(torch.max((1-self.rand) * (self.obj1(X)-args.baseline), self.rand * (self.obj2(X)-args.baseline)) + (1e-6 * linear_weighted_sum))


def is_pareto_efficient(costs, return_mask = True, epsilon=0):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index]-epsilon, axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

class PareCO:
    def __init__(self, dataset, datapath, model, mask_model, pruner, rank_type='l2_weight', batch_size=32, safeguard=0, global_random_rank=False, lub='', device='cuda', resource='filter'):
        self.device = device
        self.sample_for_ranking = 1 if rank_type in ['l1_weight', 'l2_weight', 'l0_weight', 'l2_bn', 'l1_bn', 'l2_bn_param'] else 5000
        self.safeguard = safeguard
        self.lub = lub
        self.img_size = 32 if 'CIFAR' in args.dataset else 224
        self.batch_size = batch_size
        self.rank_type = rank_type
    
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, eval(args.interpolation), True, args.slim_dataaug)

        if 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10
        elif 'ImageFolder' in dataset:
            num_classes = 1000
        self.num_classes = num_classes
        self.mask_model = mask_model
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

        # we design the code to decouple obtaining pruning masks and pruning
        # however, this feature is not used currently
        self.mask_pruner = eval(pruner)(self.mask_model, rank_type, num_classes, safeguard, random=global_random_rank, device=device, resource=resource) 
        self.pruner = eval(pruner)(self.model, 'l2_weight', num_classes, safeguard, random=global_random_rank, device=device, resource=resource) 

        self.model.train()
        self.mask_model.train()

        self.sampling_weights = np.ones(50)

    def sample_arch(self, START_BO, g, steps, hyperparams, og_flops, full_val_loss, target_flops=0):
        if args.slim:
            if target_flops == 0:
                parameterization = hyperparams.random_sample()
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            else:
                parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        else:
            # random sample to warmup history for MOBO
            if g < START_BO:
                if target_flops == 0:
                    f = np.random.rand(1) * (args.upper_channel-args.lower_channel) + args.lower_channel
                else:
                    f = args.lower_channel
                parameterization = np.ones(hyperparams.get_dim()) * f
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            # put the largest model into the history
            elif g == START_BO:
                if target_flops == 0:
                    parameterization = np.ones(hyperparams.get_dim())
                else:
                    f = args.lower_channel
                    parameterization = np.ones(hyperparams.get_dim()) * f
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            # MOBO
            else:
                # this is the scalarization (lambda_{FLOPs})
                rand = torch.rand(1).cuda()

                # standardize data for building Gaussian Processes
                train_X = torch.FloatTensor(self.X).cuda()
                train_Y_loss = torch.FloatTensor(np.array(self.Y)[:, 0].reshape(-1, 1)).cuda()
                train_Y_loss = standardize(train_Y_loss)

                train_Y_cost = torch.FloatTensor(np.array(self.Y)[:, 1].reshape(-1, 1)).cuda()
                train_Y_cost = standardize(train_Y_cost)

                new_train_X = train_X
                # GP for the cross entropy loss
                gp_loss = SingleTaskGP(new_train_X, train_Y_loss)
                mll = ExactMarginalLogLikelihood(gp_loss.likelihood, gp_loss)
                mll = mll.to('cuda')
                fit_gpytorch_model(mll)


                # GP for FLOPs
                # we use add-gp since FLOPs has addive structure (not exactly though)
                # the parameters for ScaleKernel and MaternKernel simply follow the default
                covar_module = AdditiveStructureKernel(
                    ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            lengthscale_prior=GammaPrior(3.0, 6.0),
                            num_dims=1
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    ),
                    num_dims=train_X.shape[1]
                )
                gp_cost = SingleTaskGP(new_train_X, train_Y_cost, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(gp_cost.likelihood, gp_cost)
                mll = mll.to('cuda')
                fit_gpytorch_model(mll)

                # Build acquisition functions
                UCB_loss = UpperConfidenceBound(gp_loss, beta=0.1).cuda()
                UCB_cost = UpperConfidenceBound(gp_cost, beta=0.1).cuda()

                # Combine them via augmented Tchebyshev scalarization
                self.mobo_obj = RandAcquisition(UCB_loss).cuda()
                self.mobo_obj.setup(UCB_loss, UCB_cost, rand)

                # Bounds for the optimization variable (alpha)
                lower = torch.ones(new_train_X.shape[1])*args.lower_channel
                upper = torch.ones(new_train_X.shape[1])*args.upper_channel
                self.mobo_bounds = torch.stack([lower, upper]).cuda()

                # Pareto-aware sampling
                if args.pas:
                    # Generate approximate Pareto front first
                    costs = []
                    for i in range(len(self.population_data)):
                        costs.append([self.population_data[i]['loss'], self.population_data[i]['ratio']])
                    costs = np.array(costs)
                    efficient_mask = is_pareto_efficient(costs)
                    costs = costs[efficient_mask]
                    loss = costs[:, 0]
                    flops = costs[:, 1]
                    sorted_idx = np.argsort(flops)
                    loss = loss[sorted_idx]
                    flops = flops[sorted_idx]
                    if flops[0] > args.lower_flops:
                        flops = np.concatenate([[args.lower_flops], flops.reshape(-1)])
                        loss = np.concatenate([[8], loss.reshape(-1)])
                    else:
                        flops = flops.reshape(-1)
                        loss = loss.reshape(-1)

                    if flops[-1] < args.upper_flops and (loss[-1] > full_val_loss):
                        flops = np.concatenate([flops.reshape(-1), [args.upper_flops]])
                        loss = np.concatenate([loss.reshape(-1), [full_val_loss]])
                    else:
                        flops = flops.reshape(-1)
                        loss = loss.reshape(-1)

                    # Equation (4) in paper
                    areas = (flops[1:]-flops[:-1])*(loss[:-1]-loss[1:])

                    # Quantize into 50 bins to sample from multinomial
                    self.sampling_weights = np.zeros(50)
                    k = 0
                    while k < len(flops) and flops[k] < args.lower_flops:
                        k+=1
                    for i in range(50):
                        lower = i/50.
                        upper = (i+1)/50.
                        if upper < args.lower_flops or lower > args.upper_flops or lower < args.lower_flops:
                            continue
                        cnt = 1
                        while ((k+1) < len(flops)) and upper > flops[k+1]:
                            self.sampling_weights[i] += areas[k]
                            cnt += 1
                            k += 1
                        if k < len(areas):
                            self.sampling_weights[i] += areas[k]
                        self.sampling_weights[i] /= cnt
                    if np.sum(self.sampling_weights) == 0:
                        self.sampling_weights = np.ones(50)
                        
                    if target_flops == 0:
                        val = np.arange(0.01, 1, 0.02)
                        chosen_target_flops = np.random.choice(val, p=(self.sampling_weights/np.sum(self.sampling_weights)))
                    else:
                        chosen_target_flops = target_flops
                    
                    # Binary search is here
                    lower_bnd, upper_bnd = 0, 1
                    lmda = 0.5
                    for i in range(10):
                        self.mobo_obj.rand = lmda

                        parameterization, acq_value = optimize_acqf(
                            self.mobo_obj, bounds=self.mobo_bounds, q=1, num_restarts=5, raw_samples=1000,
                        )

                        parameterization = parameterization[0].cpu().numpy()
                        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
                        sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
                        ratio = sim_flops/og_flops

                        if np.abs(ratio - chosen_target_flops) <= 0.02:
                            break
                        if args.baseline > 0:
                            if ratio < chosen_target_flops:
                                lower_bnd = lmda
                                lmda = (lmda + upper_bnd) / 2
                            elif ratio > chosen_target_flops:
                                upper_bnd = lmda
                                lmda = (lmda + lower_bnd) / 2
                        else:
                            if ratio < chosen_target_flops:
                                upper_bnd = lmda
                                lmda = (lmda + lower_bnd) / 2
                            elif ratio > chosen_target_flops:
                                lower_bnd = lmda
                                lmda = (lmda + upper_bnd) / 2
                    rand[0] = lmda
                    writer.add_scalar('Binary search trials', i, steps)

                else:
                    parameterization, acq_value = optimize_acqf(
                        self.mobo_obj, bounds=self.mobo_bounds, q=1, num_restarts=5, raw_samples=1000,
                    )
                    parameterization = parameterization[0].cpu().numpy()

                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        return layer_budget, parameterization, self.sampling_weights/np.sum(self.sampling_weights)

    def channels_to_mask(self, layer_budget):
        prune_targets = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            if (self.mask_pruner.filter_ranks[k].size(0) - layer_budget[k]) > 0:
                prune_targets.append((k, (int(layer_budget[k]), self.mask_pruner.filter_ranks[k].size(0) - 1)))
        return prune_targets

    def train(self):
        # This is warming up BO
        # The first START_BO points are a single width-multiplier sampled randomly
        # after that, we do one-step MOBO as described in the paper
        START_BO = args.prior_points

        # This is the history $\mathcal{H}$ in the paper
        self.population_data = []

        # setting up dependencies for pruner
        self.mask_pruner.reset() 
        self.mask_pruner.model.eval()
        self.mask_pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

        self.pruner.model = nn.DataParallel(self.pruner.model)

        # ========== Begin Optimizer ============
        # No weight decay for bn and bias
        iters_per_epoch = len(self.train_loader)
        no_wd_params, wd_params = [], []
        for name, param in self.pruner.model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)

        # Linearly scale learning rate based on batch size
        lr = args.baselr * (args.batch_size / 256.)

        if args.warmup > 0:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr/float(iters_per_epoch*args.warmup), momentum=args.mmt, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr, momentum=args.mmt, nesterov=args.nesterov)
        lrinfo = {'initlr': lr, 'warmup_steps': args.warmup*iters_per_epoch,
                'total_steps': args.epochs*iters_per_epoch}

        # ========== End Optimizer ============

        # label smoothing cross entropy
        criterion = CrossEntropyLabelSmooth(self.num_classes, args.label_smoothing).to('cuda')
        # loss for inplace distillation
        kd = CrossEntropyLossSoft().cuda()

        # setting up dependencies for pruner
        self.pruner.reset() 
        self.pruner.model.eval()
        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

        ind_layers = 0
        checked = np.zeros(len(self.mask_pruner.filter_ranks))
        for l in sorted(self.mask_pruner.filter_ranks.keys()):
            if checked[l]:
                continue
            k = l
            while k in self.mask_pruner.chains:
                k = self.mask_pruner.chains[k]
                checked[k] = 1
            ind_layers += 1
                
        # we've defined three level of alpha dimension
        # level 1 is single width-multiplier (slimmable)
        # level 2 is stage-wise dimension
        # level 3 is layer-wise dimension
        # these are useful for understanding the curse of dimensionality for MOBO
        # we implemented it but have not done investigation on this front yet
        hyperparams = Hyperparams(args.network)
        if args.network not in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']:
            hyperparams.dim = [1, len(self.mask_pruner.stages), ind_layers]
        for _ in range(args.param_level-1):
            hyperparams.increase_level()


        # calculate the FLOPs for the full model
        parameterization = np.ones(hyperparams.get_dim())
        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        og_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)


        # calculate the FLOPs for the smallest model
        if args.lower_channel != 0:
            parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
            args.lower_flops = (float(sim_flops) / og_flops)
            print('Lower flops based on lower channel: {}'.format(args.lower_flops))

        # X is array of the alpha in the paper
        self.X = None
        # Y is list of tuple that contains cross entropy loss and FLOPs
        self.Y = []

        self.pruner.model.train()

        # We use this to track the width-multipliers for visualization
        og_filters = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            og_filters.append(self.mask_pruner.filter_ranks[k].size(0))

        # g is the number of architectures visited so far, i.e. |\mathcal{H}}| in the paper
        g = 0
        start_epoch = 0
        maxloss = 0
        minloss = 0
        ratio_visited = []

        # widths in Algorithm 1 of the paper
        archs = []

        # load model to resume training if possible
        if os.path.exists(os.path.join('./ckpt', '{}.pt'.format(args.name))):
            ckpt = torch.load(os.path.join('./ckpt', '{}.pt'.format(args.name)))
            self.X = ckpt['X']
            self.Y = ckpt['Y']
            self.population_data = ckpt['population_data']
            self.pruner.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optim_state_dict'])
            start_epoch = ckpt['epoch']+1
            if len(self.population_data) > 1:
                g = len(self.X)
                archs = [data['filters'] for data in self.population_data[-args.num_sampled_arch:]]
            if 'ratio_visited' in ckpt:
                ratio_visited = ckpt['ratio_visited']
            print('Loading checkpoint from epoch {}'.format(start_epoch-1))


        full_val_loss = 0

        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            for i, (batch, label) in enumerate(self.train_loader):
                self.pruner.model.train()
                cur_step = iters_per_epoch*epoch+i
                lr = calculate_lr(lrinfo['initlr'], cur_step, lrinfo['total_steps'], lrinfo['warmup_steps'])
                set_lr(optimizer, lr)
                batch, label = batch.to(device), label.to(device)

                if not args.normal_training:
                    if not args.slim:

                        # tau is n in paper
                        # every tau step, we gonna sample a new set of widths based on MOBO-RS
                        # but before that, we need to make sure self.Y is up-to-date
                        if cur_step % args.tau == 0:
                            # Calibration historical data
                            if len(self.Y) > 1:
                                diff = 0
                                for j in range(len(self.Y)):
                                    with torch.no_grad():
                                        self.remove_mask()
                                        self.mask(self.population_data[j]['filters'])
                                        output = self.pruner.model(batch)
                                        loss = criterion(output, label).item()

                                        if self.Y[j][1] == 1:
                                            full_val_loss = loss

                                        diff += np.abs(loss - self.Y[j][0])
                                        self.Y[j][0] = loss
                                        self.population_data[j]['loss'] = loss

                    # every n, sample width either from MOBO-RS or randomly (Slim)
                    if cur_step % args.tau == 0:
                        archs = []
                        ratios = []
                        sampled_sim_flops = []
                        parameterizations = []
                        # Sample width
                        for _ in range(args.num_sampled_arch):
                            # sample_arch implements pareto-aware MOBO-RS and random sampling
                            layer_budget, parameterization, weights = self.sample_arch(START_BO, g, cur_step, hyperparams, og_flops, full_val_loss)
                            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
                            sampled_sim_flops.append(sim_flops)
                            ratio = sim_flops/og_flops
                            ratios.append(ratio)
                            ratio_visited.append(ratio)

                            parameterizations.append(parameterization)
                            g += 1

                            # from width-multipliers to channel masks
                            # we follow slimmable to take the first few channels
                            prune_targets = self.channels_to_mask(layer_budget)

                            # for visualization (this is for debugging)
                            if not args.slim:
                                for tgt in args.track_flops:
                                    if np.abs(ratio-tgt) <= 0.02:
                                        filters = np.array(og_filters)
                                        for k, (start, end) in prune_targets:
                                            filters[k] -= (end-start+1)

                                        ax = plt.figure(figsize=(6,4), dpi=300)
                                        plt.bar(range(len(filters)), np.array(og_filters), color='grey')
                                        plt.bar(range(len(filters)), filters)
                                        plt.xlabel('Layer index')
                                        plt.ylabel('Filter counts')
                                        plt.title('FLOPs: {:.2f}'.format(ratio))
                                        writer.add_figure('Arch/FLOPs: {}'.format(tgt), ax, g)
                                        break
                            
                            archs.append(prune_targets)

                        if not args.slim:
                            # Visualize approximate Pareto curve during training (this is for debugging)
                            if (g//args.num_sampled_arch+1) % (10//args.num_sampled_arch) == 0:
                                costs = []
                                for j in range(len(self.population_data)):
                                    costs.append([self.population_data[j]['loss'], self.population_data[j]['ratio']])
                                costs = np.array(costs)
                                efficient_mask = is_pareto_efficient(costs)
                                costs = costs[efficient_mask]
                                loss = costs[:, 0]
                                flops = costs[:, 1]
                                sorted_idx = np.argsort(flops)
                                loss = loss[sorted_idx]
                                flops = flops[sorted_idx]
                                if flops[0] > args.lower_flops:
                                    flops = np.concatenate([[args.lower_flops], flops.reshape(-1)])
                                    loss = np.concatenate([[8], loss.reshape(-1)])
                                else:
                                    flops = flops.reshape(-1)
                                    loss = loss.reshape(-1)

                                if flops[-1] < 1 and loss[-1] > full_val_loss:
                                    flops = np.concatenate([flops.reshape(-1), [1]])
                                    loss = np.concatenate([loss.reshape(-1), [full_val_loss]])
                                else:
                                    flops = flops.reshape(-1)
                                    loss = loss.reshape(-1)

                                ax = plt.figure(figsize=(6,4), dpi=300)
                                plt.plot(flops, loss, '--.', drawstyle='steps-post')
                                plt.plot(np.arange(0.01, 1, 0.02), weights)
                                plt.xlabel('FLOPs ratio')
                                plt.ylabel('Training loss')

                                plt.title('Archs: {}'.format(g+1))
                                writer.add_figure('Trade-off curve', ax, g+1)

                            # Expand history ($\mathcal{H}$) for MOBO-RS and getting the approximate Pareto frontier
                            if self.X is None:
                                self.X = np.array(parameterizations)
                            else:
                                self.X = np.concatenate([self.X, parameterizations], axis=0)
                            for ratio, sim_flops, prune_targets in zip(ratios, sampled_sim_flops, archs):
                                self.Y.append([0, ratio])
                                self.population_data.append({'loss': 0, 'flops': sim_flops, 'ratio': ratio, 'filters': prune_targets})


                        # Add the smallest model to widths for the sandwich rule
                        parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
                        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
                        sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
                        ratio = sim_flops/og_flops
                        prune_targets = self.channels_to_mask(layer_budget)
                        archs.append(prune_targets)

                # Inplace distillation
                self.pruner.model.zero_grad()
                self.remove_mask()
                t_output = self.pruner.model(batch)
                loss = criterion(t_output, label)
                loss.backward()
                maxloss = loss.item()
                for prune_targets in archs:
                    self.remove_mask()
                    self.mask(prune_targets)
                    output = self.pruner.model(batch)
                    loss = kd(output, t_output.detach())
                    loss.backward()
                    minloss = loss.item()

                if cur_step % args.print_freq == 0:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    writer.add_scalar('Loss for largest model', maxloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Loss for smallest model', minloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Learning rate', lr, epoch*len(self.train_loader)+i)

                optimizer.step()
                sys.stdout.flush()

            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt')
            torch.save({'model_state_dict': self.pruner.model.state_dict(), 'optim_state_dict': optimizer.state_dict(),
                        'epoch': epoch, 'population_data': self.population_data, 'X': self.X, 'Y': self.Y, 'ratio_visited': ratio_visited}, os.path.join('./ckpt', '{}.pt'.format(args.name)))
            if len(ratio_visited) > 0:
                writer.add_histogram('FLOPs visited', np.array(ratio_visited), epoch+1)
            print('Epoch {} | Time: {:.2f}s'.format(epoch, time.time()-start_time))

            if args.normal_training:
                test_top1, test_top5 = test(self.pruner.model, self.test_loader, device='cuda')
                writer.add_scalar('Test acc/Top-1', test_top1, epoch+1)
                writer.add_scalar('Test acc/Top-5', test_top1, epoch+1)

    def mask(self, prune_targets):
        for layer_index, filter_index in prune_targets:
            self.pruner.activation_to_conv[layer_index].mask[filter_index[0]:filter_index[1]+1].zero_()

    def remove_mask(self):
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            self.pruner.activation_to_conv[k].mask.zero_()
            self.pruner.activation_to_conv[k].mask += 1

def get_args():
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument("--name", type=str, default='test', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--network", type=str, default='ResNet20', help='The network to use')
    parser.add_argument("--reinit", action='store_true', default=False, help='Not using pre-trained models, has to be specified for re-training timm models')
    parser.add_argument("--resource_type", type=str, default='flops', help='FLOPs')
    parser.add_argument("--pruner", type=str, default='FilterPrunerResNet', help='Different network require differnt pruner implementation (FilterPrunerResNet | FilterPrunerMBNetV3)')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--print_freq", type=int, default=500, help='Logging frequency in iterations')

    # Training
    parser.add_argument("--epochs", type=int, default=120, help='Number of training epochs')
    parser.add_argument("--warmup", type=int, default=5, help='Number of warmup epochs')
    parser.add_argument("--baselr", type=float, default=0.05, help='Learning rate')
    parser.add_argument("--scheduler", type=str, default='cosine_decay', help='Support: cosine_decay | linear_decay')
    parser.add_argument("--mmt", type=float, default=0.9, help='Momentum')
    parser.add_argument("--tau", type=int, default=625, help='n in the paper. Sample width every tau iterations')
    parser.add_argument("--wd", type=float, default=4e-5, help='The weight decay used')
    parser.add_argument("--label_smoothing", type=float, default=1e-1, help='Label smoothing')
    parser.add_argument("--batch_size", type=int, default=1024, help='Batch size for training')
    parser.add_argument("--logging", action='store_true', default=False, help='Log the output')
    parser.add_argument("--normal_training", action='store_true', default=False, help='For independent trained model')
    parser.add_argument("--nesterov", action='store_true', default=False, help='Nesterov or not')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')

    # Channel
    parser.add_argument("--param_level", type=int, default=1, help='Dimension of alpha (1: network-wise, 2: stage-wise, 3: layer-wise)')
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound for alpha')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound for alpha')
    parser.add_argument("--lower_flops", type=float, default=0.1, help='lower bound for FLOPs (not used)')
    parser.add_argument("--upper_flops", type=float, default=1, help='upper bound for FLOPs')
    parser.add_argument("--slim", action='store_true', default=False, help='Use slimmable training')
    parser.add_argument("--num_sampled_arch", type=int, default=1, help='Number of arch sampled in between largest and smallest (M in paper)')
    parser.add_argument('--track_flops', nargs='+', default=[0.35, 0.5, 0.75], help='For visualization only')

    # GP-related hyper-param (PareCO)
    parser.add_argument("--prior_points", type=int, default=10, help='Used for warming up the histroy for MOBO')
    parser.add_argument("--baseline", type=int, default=-3, help='Use for scalarization')
    parser.add_argument("--pas", action='store_true', default=False, help='Pareto-aware scalarization')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('./runs/{}'.format(args.name))
    if args.logging:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        sys.stdout = open('./log/{}.log'.format(args.name), 'a')
    print(args)

    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageFolder' in args.dataset:
        num_classes = 1000

    device = 'cuda'

    # we design the code to decouple obtaining pruning masks and pruning
    # however, this feature is not used currently
    #
    # mask_model is for obtaining pruning masks
    # model is the real model
    #
    # this design can definitely be simplied, we do so just for the sake of convenience
    if args.network in 'mobilenetv3':
        mask_model = timm.create_model('mobilenetv3_large_100', pretrained=not args.reinit)
    elif args.network in ['mobilenetv2_035', 'mobilenetv2_050', 'mobilenetv2_075', 'mobilenetv2_100']:
        mask_model = timm.create_model(args.network, pretrained=not args.reinit)

    else:
        mask_model = eval(args.network)(num_classes=num_classes)
    mask_model = mask_model.to(device)

    if args.network == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=not args.reinit)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(masked_forward)
                m.mask = nn.Parameter(torch.ones(m.weight.size(0)), requires_grad=False)
    elif args.network in ['mobilenetv2_035', 'mobilenetv2_050', 'mobilenetv2_075', 'mobilenetv2_100']:
        model = timm.create_model(args.network, pretrained=not args.reinit)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(masked_forward)
                m.mask = nn.Parameter(torch.ones(m.weight.size(0)), requires_grad=False)
    else:
        model = eval(args.network)(num_classes=num_classes)
    model = model.to(device)

    pareco = PareCO(args.dataset, args.datapath, model, mask_model, args.pruner, batch_size=args.batch_size, device=device, resource=args.resource_type)

    pareco.train()
