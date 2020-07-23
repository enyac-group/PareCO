import os
import sys
import copy
import time
import math
import torch
import queue
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import timm

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
from botorch.acquisition import UpperConfidenceBound, qMaxValueEntropy
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

from botorch.gen import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions

from model.resnet_cifar10 import ResNet20, ResNet32, ResNet44, ResNet56

import PIL

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

def masked_forward(self, input, output):
    return (output.permute(0,2,3,1) * self.mask).permute(0,3,1,2)


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


    def channels_to_mask(self, layer_budget):
        prune_targets = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            if (self.mask_pruner.filter_ranks[k].size(0) - layer_budget[k]) > 0:
                prune_targets.append((k, (int(layer_budget[k]), self.mask_pruner.filter_ranks[k].size(0) - 1)))
        return prune_targets

    def eval(self):
        # Load checkpoint using name!!
        data_dict = torch.load(os.path.join('./ckpt', '{}.pt'.format(args.name)))

        population_data = data_dict['population_data']

        self.pruner.model = nn.DataParallel(self.pruner.model)
        if 'state_dict' in data_dict:
            self.pruner.model.load_state_dict(data_dict['state_dict'])
        elif 'model_state_dict' in data_dict:
            self.pruner.model.load_state_dict(data_dict['model_state_dict'])
        print('Load from epoch: {}'.format(data_dict['epoch']))

        self.mask_pruner.reset() 
        self.mask_pruner.model.eval()
        self.mask_pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

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

        print('Alpha dim: {}'.format(hyperparams.dim[-1]))

        parameterization = np.ones(hyperparams.get_dim())
        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        og_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)

        if args.lower_channel != 0:
            parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
            args.lower_flops = (float(sim_flops) / og_flops)
            print('Lower flops based on lower channel: {}'.format(args.lower_flops))


        og_filters = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            og_filters.append(self.mask_pruner.filter_ranks[k].size(0))


        # Reset BN stats, check full model acc
        for m in self.pruner.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        with torch.no_grad():
            self.remove_mask()
            self.pruner.model.train()
            for i, (batch, label) in enumerate(self.train_loader):
                batch = batch.to('cuda')
                out = model(batch)
                if 'CIFAR' not in args.dataset and i == 2:
                    break
            full_test_top1, full_test_top5  = test(self.pruner.model, self.test_loader, device='cuda')
        print('Full: {:.2f}, {:.2f} MFLOPS: {:.3f}'.format(full_test_top1, full_test_top5, og_flops*1e-6))


        # Reset BN stats, check smallest model acc
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            f = args.lower_channel
            parameterization = np.ones(hyperparams.get_dim()) * f

            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            smallest_flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
            prune_targets = self.channels_to_mask(layer_budget)

            self.pruner.model.train()
            for m in self.pruner.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = None
            self.remove_mask()
            self.mask(prune_targets)
            loss = 0
            for i, (batch, label) in enumerate(self.train_loader):
                batch = batch.to('cuda')
                label = label.to('cuda')
                out = model(batch)
                loss += criterion(out, label).item()
                if 'CIFAR' not in args.dataset and i == 2:
                    break

            smallest_test_top1, smallest_test_top5  = test(self.pruner.model, self.test_loader, device='cuda')
        print('Smallest: {:.2f}, {:.2f} MFLOPS: {:.3f}'.format(smallest_test_top1, smallest_test_top5, smallest_flops*1e-6))

        cnt_filters = np.array(og_filters)
        for k, (start, end) in prune_targets:
            cnt_filters[k] -= (end-start+1)

        widths = [cnt_filters/np.array(og_filters)]
        train_loss = [loss]
        test_top1s = [smallest_test_top1]
        test_top5s = [smallest_test_top5]
        new_flops = [float(smallest_flops)/og_flops]

        if not args.uniform:
            # Get approximate Pareto frontier using history data
            costs = []
            filters = []
            for i in range(len(population_data)-2):
                if 'acc' in population_data[i]:
                    costs.append([1-population_data[i]['acc']/100., population_data[i]['ratio']])
                elif 'loss' in population_data[i]:
                    costs.append([population_data[i]['loss'], population_data[i]['ratio']])
                filters.append(population_data[i]['filters'])

            max_flops = 0
            costs = np.array(costs)
            tmp_costs = np.array(costs)
            global_efficient_mask = np.zeros(len(costs))
            while max_flops < 0.9:
                efficient_mask = is_pareto_efficient(tmp_costs)
                efficient_mask[costs[:, 1] < max_flops] = False
                if np.sum(efficient_mask) == 0:
                    break
                max_flops = np.max(costs[efficient_mask][:, 1])
                global_efficient_mask = np.logical_or(global_efficient_mask, efficient_mask)
                tmp_costs[efficient_mask] = np.ones_like(tmp_costs[efficient_mask])
            efficient_mask = global_efficient_mask 
            filters = np.array(filters)[efficient_mask]
            ratios = costs[efficient_mask][:, 1]

            for li, layer_budget in enumerate(filters):
                lb = []
                for k in sorted(self.mask_pruner.filter_ranks.keys()):
                    lb.append(self.mask_pruner.filter_ranks[k].size(0))

                for l, (s, e) in layer_budget:
                    lb[l] -= (e-s)+1
                lb = np.array(lb)
                sim_flops = self.mask_pruner.simulate_and_count_flops(lb)
                ratio = float(sim_flops)/og_flops
                new_flops.append(ratio)

                start = time.time()
                pruning_t = time.time() - start
                start = time.time()

                self.pruner.model.train()
                self.remove_mask()
                self.mask(layer_budget)

                finetuning_t = time.time() - start
                start = time.time()
                loss = 0
                with torch.no_grad():
                    self.pruner.model.train()
                    for m in self.pruner.model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.reset_running_stats()
                            m.momentum = None
                    for i, (batch, label) in enumerate(self.train_loader):
                        batch = batch.to('cuda')
                        label = label.to('cuda')
                        out = model(batch)
                        loss += criterion(out, label).item()
                        if 'CIFAR' not in args.dataset and i == 2:
                            break
                    test_top1, test_top5 = test(self.pruner.model, self.test_loader, device='cuda')
                    train_loss.append(loss)

                testing_t = time.time() - start

                test_top1s.append(test_top1)
                test_top5s.append(test_top5)
                cnt_filters = np.array(og_filters)
                for k, (start, end) in layer_budget:
                    cnt_filters[k] -= (end-start+1)
                widths.append(cnt_filters/np.array(og_filters))


                print('({}/{}) Loss: {:2f} Acc: {:.2f} {:.2f}, MFLOPs: {:.3f} ({:.2f} %) | Pruning: {:.2f}, Tuning: {:.2f}, Testing: {:.2f}'.format(li, len(filters), loss, test_top1, test_top5, og_flops*ratio*1e-6, ratio*100., pruning_t, finetuning_t, testing_t))

        else:
            num = 40
            for g in range(num):
                f = np.sqrt((float(g) / num)*(args.upper_flops-args.lower_flops) + args.lower_flops)

                parameterization = np.ones(hyperparams.get_dim()) * f
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
                flops = self.mask_pruner.simulate_and_count_flops(layer_budget)
                ratio = float(flops) / og_flops
                prune_targets = self.channels_to_mask(layer_budget)
                
                self.pruner.model.train()
                for m in self.pruner.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = None
                self.remove_mask()
                self.mask(prune_targets)
                loss = 0
                with torch.no_grad():
                    self.pruner.model.train()
                    for i, (batch, label) in enumerate(self.train_loader):
                        batch = batch.to('cuda')
                        label = label.to('cuda')
                        out = self.pruner.model(batch)
                        loss += criterion(out, label).item()
                        if 'CIFAR' not in args.dataset and i == 2:
                            break
                    test_top1, test_top5 = test(self.pruner.model, self.test_loader, device='cuda')
                train_loss.append(loss)
                test_top1s.append(test_top1)
                test_top5s.append(test_top5)
                new_flops.append(ratio)

                cnt_filters = np.array(og_filters)
                for k, (start, end) in prune_targets:
                    cnt_filters[k] -= (end-start+1)
                widths.append(cnt_filters/np.array(og_filters))


                print('WM: {:.2f} Acc: {:.2f} {:.2f}, MFLOPs: {:.3f} ({:.2f} %)'.format(f, test_top1, test_top5, og_flops*ratio*1e-6, ratio*100.))


        # Get approximate Pareto frontier using the training loss and FLOPs
        efficient_mask = is_pareto_efficient(np.stack([np.array(train_loss), np.array(new_flops)], axis=1))
        new_flops = np.array(new_flops)[efficient_mask]
        test_top1s = np.array(test_top1s)[efficient_mask]
        test_top5s = np.array(test_top5s)[efficient_mask]
        widths = np.array(widths)[efficient_mask]

        new_flops = np.concatenate([new_flops.reshape(-1), [1]])
        test_top1s = np.concatenate([test_top1s.reshape(-1), [full_test_top1]])
        test_top5s = np.concatenate([test_top5s.reshape(-1), [full_test_top5]])
        widths = np.concatenate([widths, np.ones((1, len(cnt_filters)))])

        sorted_idx = np.argsort(new_flops)
        new_flops = new_flops[sorted_idx]
        test_top1s = test_top1s[sorted_idx]
        test_top5s = test_top5s[sorted_idx]
        widths = widths[sorted_idx]

        if not args.uniform:
            np.savetxt('{}_eval_pareto.txt'.format(args.name), np.stack([new_flops, test_top1s, test_top5s]))
            np.savetxt('{}_eval_pareto_widths.txt'.format(args.name), widths)
        else:
            np.savetxt('{}_eval_uniform_pareto.txt'.format(args.name), np.stack([new_flops, test_top1s, test_top5s]))
            np.savetxt('{}_eval_uniform_pareto_widths.txt'.format(args.name), widths)

    def mask(self, prune_targets):
        for layer_index, filter_index in prune_targets:
            self.pruner.activation_to_conv[layer_index].mask[filter_index[0]:filter_index[1]+1].zero_()

    def remove_mask(self):
        for m in self.pruner.model.modules():
            if hasattr(m, 'mask'):
                m.mask.zero_()
                m.mask += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--network", type=str, default='Conv6', help='The model used to derive mask')
    parser.add_argument("--resource_type", type=str, default='filter', help='determining the threshold')
    parser.add_argument("--pruner", type=str, default='FilterPrunerResNet', help='Different network require differnt pruner implementation')
    parser.add_argument("--logging", action='store_true', default=False, help='Log the output')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training.')
    parser.add_argument("--reinit", type=str, default='none', help='Re-init the pruned network')
    parser.add_argument("--uniform", action='store_true', default=False, help='Evaluate Slim via a single width-multiplier')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound for alpha')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound for alpha')
    parser.add_argument("--lower_flops", type=float, default=0.1, help='lower bound for FLOPs')
    parser.add_argument("--upper_flops", type=float, default=1, help='upper bound for FLOPs')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if args.logging:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        sys.stdout = open('./log/{}.log'.format(args.name), 'a')
    print(args)

    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageNet' in args.dataset:
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

    pareco.eval()