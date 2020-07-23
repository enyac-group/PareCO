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
plt.switch_backend('agg')

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
        self.mask_pruner = eval(pruner)(self.mask_model, rank_type, num_classes, safeguard, random=global_random_rank, device=device, resource=resource) 
        self.pruner = eval(pruner)(self.model, 'l2_weight', num_classes, safeguard, random=global_random_rank, device=device, resource=resource) 

        self.model.train()
        self.mask_model.train()

        self.sampling_weights = np.ones(50)

    def sample_arch(self, START_BO, g, hyperparams, og_flops, empty_val_loss, full_val_loss, target_flops=0):
        # Warming up the history with a single width-multiplier
        if g < START_BO:
            if target_flops == 0:
                f = np.random.rand(1) * (args.upper_channel-args.lower_channel) + args.lower_channel
            else:
                f = args.lower_channel
            parameterization = np.ones(hyperparams.get_dim()) * f
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        # Put largest model into the history
        elif g == START_BO:
            if target_flops == 0:
                parameterization = np.ones(hyperparams.get_dim())
            else:
                f = args.lower_channel
                parameterization = np.ones(hyperparams.get_dim()) * f
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        # MOBO-RS
        else:
            rand = torch.rand(1).cuda()

            train_X = torch.FloatTensor(self.X).cuda()
            train_Y_loss = torch.FloatTensor(np.array(self.Y)[:, 0].reshape(-1, 1)).cuda()
            train_Y_loss = standardize(train_Y_loss)

            train_Y_cost = torch.FloatTensor(np.array(self.Y)[:, 1].reshape(-1, 1)).cuda()
            train_Y_cost = standardize(train_Y_cost)

            new_train_X = train_X
            gp_loss = SingleTaskGP(new_train_X, train_Y_loss)
            mll = ExactMarginalLogLikelihood(gp_loss.likelihood, gp_loss)
            mll = mll.to('cuda')
            fit_gpytorch_model(mll)


            # Use add-gp for cost
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

            UCB_loss = UpperConfidenceBound(gp_loss).cuda()
            UCB_cost = UpperConfidenceBound(gp_cost).cuda()
            self.mobo_obj = RandAcquisition(UCB_loss).cuda()
            self.mobo_obj.setup(UCB_loss, UCB_cost, rand)

            lower = torch.ones(new_train_X.shape[1])*args.lower_channel
            upper = torch.ones(new_train_X.shape[1])*args.upper_channel
            self.mobo_bounds = torch.stack([lower, upper]).cuda()

            if args.pas:
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
                    loss = np.concatenate([[empty_val_loss], loss.reshape(-1)])
                else:
                    flops = flops.reshape(-1)
                    loss = loss.reshape(-1)

                if flops[-1] < args.upper_flops and (loss[-1] > full_val_loss):
                    flops = np.concatenate([flops.reshape(-1), [args.upper_flops]])
                    loss = np.concatenate([loss.reshape(-1), [full_val_loss]])
                else:
                    flops = flops.reshape(-1)
                    loss = loss.reshape(-1)

                areas = (flops[1:]-flops[:-1])*(loss[:-1]-loss[1:])

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
                writer.add_scalar('Binary search trials', i, g)

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

    def search(self):
        START_BO = args.prior_points
        self.population_data = []

        self.mask_pruner.reset() 
        self.mask_pruner.model.eval()
        self.mask_pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

        self.pruner.model = nn.DataParallel(self.pruner.model)
        self.pruner.model.load_state_dict(torch.load(args.model)['model_state_dict'])

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
                
        hyperparams = Hyperparams(args.network)
        if args.network not in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56']:
            hyperparams.dim = [1, len(self.mask_pruner.stages), ind_layers]
        for _ in range(args.param_level-1):
            hyperparams.increase_level()
        
        parameterization = np.ones(hyperparams.get_dim())
        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        og_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)

        if args.lower_channel != 0:
            parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
            args.lower_flops = (float(sim_flops) / og_flops)
            print('Lower flops based on lower channel: {}'.format(args.lower_flops))


        self.X = None
        self.Y = []

        self.pruner.model.train()

        og_filters = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            og_filters.append(self.mask_pruner.filter_ranks[k].size(0))

        g = 0
        start_epoch = 0
        maxloss = 0
        minloss = 0
        ratio_visited = []
        smallest_archs = []

        parameterization = np.zeros(hyperparams.get_dim())
        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        prune_targets = self.channels_to_mask(layer_budget)
        empty_val_loss = 0
        self.remove_mask()
        self.mask(prune_targets)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to('cuda')
                y = y.to('cuda')
                output = self.pruner.model(x)
                empty_val_loss += self.criterion(output, y).item()
                if i == 1:
                    break

        full_val_loss = 0
        for g in range(args.history_length):
            start_time = time.time()
            layer_budget, parameterization, weights = self.sample_arch(START_BO, g, hyperparams, og_flops, empty_val_loss, full_val_loss)
            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
            ratio = sim_flops/og_flops
            ratio_visited.append(ratio)

            # Eval on validation set
            parameterization = parameterization.reshape(-1)
            prune_targets = self.channels_to_mask(layer_budget)

            cur_loss = 0
            self.remove_mask()
            self.mask(prune_targets)
            with torch.no_grad():
                for i, (x, y) in enumerate(self.train_loader):
                    x = x.to('cuda')
                    y = y.to('cuda')
                    output = self.pruner.model(x)
                    cur_loss += self.criterion(output, y).item()
                    if i == 1:
                        break

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
                            

            if (g+1) % 10 == 0:
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
                    loss = np.concatenate([[empty_val_loss], loss.reshape(-1)])
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

            if self.X is None:
                self.X = np.array([parameterization])
            else:
                self.X = np.concatenate([self.X, [parameterization]], axis=0)
            self.Y.append([cur_loss, ratio])
            self.population_data.append({'loss': cur_loss, 'flops': sim_flops, 'ratio': ratio, 'filters': prune_targets})
            sys.stdout.flush()

            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt')
            torch.save({'model_state_dict': self.pruner.model.state_dict(),
                        'population_data': self.population_data, 'X': self.X, 'Y': self.Y},
                        os.path.join('./ckpt', '{}.pt'.format(args.name)))
            writer.add_histogram('FLOPs visited', np.array(ratio_visited), g)
            print('Arch {} | Time: {:.2f}s'.format(g, time.time()-start_time))
        

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
    parser.add_argument("--model", type=str, default='', help='The pre-trained model used for the search')
    parser.add_argument("--network", type=str, default='ResNet20', help='The network to use')
    parser.add_argument("--reinit", action='store_true', default=False, help='Not using pre-trained models, has to be specified for re-training timm models')
    parser.add_argument("--resource_type", type=str, default='flops', help='FLOPs')
    parser.add_argument("--pruner", type=str, default='FilterPrunerMBNetV3', help='Different network require differnt pruner implementation')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--print_freq", type=int, default=500, help='Logging frequency in iterations')

    # Training
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training')
    parser.add_argument("--logging", action='store_true', default=False, help='Log the output')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')

    # Channel
    parser.add_argument("--param_level", type=int, default=1, help='Dimension of alpha (1: network-wise, 2: stage-wise, 3: layer-wise)')
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound for alpha')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound for alpha')
    parser.add_argument("--lower_flops", type=float, default=0.1, help='lower bound for FLOPs (not used)')
    parser.add_argument("--upper_flops", type=float, default=1, help='upper bound for FLOPs')
    parser.add_argument('--track_flops', nargs='+', default=[0.35, 0.5, 0.75], help='For visualization only')

    # GP-related hyper-param (PareCO)
    parser.add_argument("--history_length", type=int, default=1000, help='Total number of width to visit')
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

    pareco.search()
