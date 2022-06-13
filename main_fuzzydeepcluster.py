# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from distutils.command.sdist import sdist
import math
import umap
import os
import shutil
import time
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
from logging import getLogger
#from comet_ml import Experiment, OfflineExperiment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
from scipy.sparse import csr_matrix

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    pairwise_distances
)
from src.multicropdataset import MultiCropDataset, MultiCropDatasetcifar10, MultiCropDatasetcifar100, MultiCropDatasetImageNet
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--dataset", type=str, default="imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## dcv2 specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=[3000, 3000, 3000], type=int, nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--percent_worst", default=0.1, type=int, nargs="+",
                    help="Percentage of worst not classes for negative sampling to take into considereation")
parser.add_argument("--nmb_cmeans_iters", default=2, type=int,
                    help="Numbers of etaration of cmeans")

parser.add_argument("--fuzzy_param", default=2.0, type=float,
                    help="fuzzy parameter")
parser.add_argument("--freeze", default=True, type=bool,
                    help="Freeze network after triple learning")
parser.add_argument("--triplet", default=0, type=int,
                    help="Triplet method to use")
#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs_con", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--epochs", default=40, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--base_lr_contr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--wd_contr", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    if args.dataset == "imagenet":    
        train_dataset = MultiCropDatasetImageNet(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            return_index=True,
        )
    elif args.dataset == "cifar10":
        train_dataset = MultiCropDatasetcifar10(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            return_index=True,
        )
    elif args.dataset == "cifar100":
        train_dataset = MultiCropDatasetcifar100(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            return_index=True,
        )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=0,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer_contr = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr_contr,
        momentum=0.9,
        weight_decay=args.wd_contr,
    )
    optimizer_contr = LARC(optimizer=optimizer_contr, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr_contr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr_contr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs_contr - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building contrastive optimizer done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint_backbone.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer_contr,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
    if os.path.isfile(mb_path):
        mb_ckp = torch.load(mb_path)
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
        local_memory_membership = mb_ckp["local_memory_membership"]
    else:
        local_memory_embeddings, local_memory_membership = init_memory(train_loader, model)

    # Create an experiment with your api key
#    if args.rank == 0:
#        experiment = OfflineExperiment(
#            api_key="ceYg4Xql1HqeiIsYQJtBJuECo",
#            project_name="fuzzydeepcluster",
#            workspace="wolodja",
#            offline_directory="/cgtvx/FuzzyDeepCluster/experiments",
#            auto_output_logging="simple"
#        )
#    else:
#        experiment = OfflineExperiment(
#            api_key="ceYg4Xql1HqeiIsYQJtBJuECo",
#            project_name="fuzzydeepcluster",
#            workspace="wolodja",
#            offline_directory="/cgtvx/FuzzyDeepCluster/experiments",
#            disabled=True
#        )
#
#    experiment.add_tag("Backbone training")
#    
#    experiment.log_parameters({
#        "nmb_crops": args.nmb_crops,
#        "size_crops": args.size_crops,
#        "min_scale_crops": args.min_scale_crops,
#        "max_scale_crops": args.max_scale_crops,
#        "crops_for_assign": args.crops_for_assign,
#        "feat_dim": args.feat_dim,
#        "percent_worst": args.percent_worst,
#        "epochs": args.epochs_con,
#        "batch_size": args.batch_size,
#        "base_lr": args.base_lr_contr,
#        "weight_decay": args.wd_contr,
#        "final_lr": args.final_lr,
#        "warmup_epochs": args.warmup_epochs,
#        "start_warmup": args.start_warmup,
#        "arch": args.arch,
#        "hidden_mlp": args.hidden_mlp,
#        "workers": args.workers,
#        "sync_bn": args.sync_bn,
#        "nmb_cmeans_iters": args.nmb_cmeans_iters
#    })
    cudnn.benchmark = True
#    with experiment.train():
    for epoch in range(start_epoch, args.epochs_con):

        # train the network for one epoch
        logger.info("============ Starting backbone train epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_embeddings, local_memory_membership = train_backbone(
                train_loader,
                model,
                optimizer_contr,
                epoch,
                lr_schedule,
                local_memory_embeddings,
                local_memory_membership,
                args.nmb_cmeans_iters,
                args.percent_worst
        )
        training_stats.update(scores)
        logger.info(f"Loss avg {scores[1]} for epoch {scores[0]}")
#       experiment.log_metric("loss", scores[1], step=scores[0])
            
        validate_contrastive(local_memory_embeddings, None, epoch, args)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer_contr.state_dict(),
            }
            torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint_backbone.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs_con - 1:
                shutil.copyfile(
                        os.path.join(args.dump_path, "checkpoint_backbone.pth.tar"),
                        os.path.join(args.dump_checkpoints, "ckp-backbone-" + str(epoch) + ".pth"),
                )
        torch.save({"local_memory_embeddings": local_memory_embeddings,
                        "local_memory_membership": local_memory_membership}, mb_path)
    
    #TODO add visualization of embedings with classes
    # add head
    model.module.add_prototypes(args.nmb_prototypes)
    if args.freeze:
        for name, p in model.named_parameters():
            if "prototypes" not in name:
                p.requires_grad = False

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    logger.info("Building optimizer done.")
    
    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model.module.cuda(),
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]
    
#    if args.rank == 0:
#        experiment = OfflineExperiment(
#            api_key="ceYg4Xql1HqeiIsYQJtBJuECo",
#            project_name="fuzzydeepcluster",
#            workspace="wolodja",
#            offline_directory="/cgtvx/FuzzyDeepCluster/experiments",
#            auto_output_logging="simple"
#        )
#    else:
#        experiment = OfflineExperiment(
#            api_key="ceYg4Xql1HqeiIsYQJtBJuECo",
#            project_name="fuzzydeepcluster",
#            workspace="wolodja",
#            offline_directory="/cgtvx/FuzzyDeepCluster/experiments",
#            disabled=True
#        )
#    
#    experiment.add_tag("Head training")
#    
#    experiment.log_parameters({
#        "nmb_crops": args.nmb_crops,
#        "size_crops": args.size_crops,
#        "min_scale_crops": args.min_scale_crops,
#        "max_scale_crops": args.max_scale_crops,
#        "crops_for_assign": args.crops_for_assign,
#        "feat_dim": args.feat_dim,
#        "epochs": args.epochs,
#        "batch_size": args.batch_size,
#        "base_lr": args.base_lr,
#        "weight_decay": args.wd,
#        "arch": args.arch,
#        "hidden_mlp": args.hidden_mlp,
#        "workers": args.workers,
#        "sync_bn": args.sync_bn,
#        "temperature": args.temperature,
#        "nmb_prototypes": args.nmb_prototypes
#    })
#    with experiment.train():
    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info("============ Starting head train epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)
            
        # train the network
        scores = train_head(
                train_loader,
                optimizer,
                model,
                epoch,
                local_memory_membership
        )
        training_stats.update(scores)
        #experiment.log_metric("loss", scores[1], step=scores[0])
        logger.info(f"Loss {scores[1]} for epoch {scores[0]}")

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict()
            }
            torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint_final.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                        os.path.join(args.dump_path, "checkpoint_final.pth.tar"),
                        os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )

def train_backbone(loader, model, optimizer, epoch, schedule, local_memory_embeddings, local_memory_membership, nmb_cmeans_iters=30, percent_worst=0.2):
    model.train()
    losses = AverageMeter()

    local_memory_membership = cluster_memory(local_memory_membership, local_memory_embeddings, nmb_cmeans_iters)
    logger.info('Clustering for epoch {} done.'.format(epoch))

    start_idx = 0
    for it, (idx, inputs) in enumerate(loader):
        # update learning rate
        iteration = epoch * len(loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]
            
        # ============ multi-res forward passes ... ============
        emb = model(inputs)
        bs = int(emb.shape[0] / len(args.crops_for_assign))

        # ============ Triplet loss ... ============
        # order embd based on fuzzy clustering with centroids
        if args.triplet == 0:
            loss = triplet_each(local_memory_membership, local_memory_embeddings, emb, start_idx, bs, percent_worst)
        else:
            loss = triplet_all(local_memory_membership, local_memory_embeddings, emb, start_idx, bs, percent_worst)
            
        if loss != 0 and not torch.isnan(loss).any():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        if args.rank ==0 and it % 50 == 0:
            logger.info(f'Train: Epoch [{epoch}], Step [{it}/{len(loader)}], loss: {loss.item():.3f}, lr {optimizer.optim.param_groups[0]["lr"]:.4f}')
    return (epoch, losses.avg), local_memory_embeddings, local_memory_membership

def train_head(loader, optimizer, model, epoch, local_memory_membership):
    model.train()
    losses = AverageMeter()
    
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
    
    # ============ get clusters ... ============
    assignments = -100 * torch.ones(len(args.nmb_prototypes), local_memory_membership.shape[1]).long()
    assignments_to_assign = torch.argmax(local_memory_membership, dim=2).detach().cpu()
    if assignments.shape[0] > assignments_to_assign.shape[0]:
        assignments[:assignments_to_assign.shape[0]] = assignments_to_assign
    elif assignments.shape[0] < assignments_to_assign.shape[0]:
        assignments = assignments_to_assign[:assignments.shape[0]]
    else:
        assignments = assignments_to_assign
    
    start_idx = 0
    for it, (idx, inputs) in enumerate(loader):            
         # ============ multi-res forward passes ... ============
        optimizer.zero_grad()
        emb, output = model(inputs)
        bs = int(emb.shape[0] / len(args.crops_for_assign))

        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / args.temperature
            targets = assignments[h,idx].repeat(sum(args.nmb_crops)).cuda()#(non_blocking=True)
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        start_idx += bs
        
        losses.update(loss.item(), inputs[0].size(0))
        if args.rank ==0 and it % 50 == 0:
            logger.info(f'Train: Epoch [{epoch}], Step [{it}/{len(loader)}], loss: {loss.item():.3f}')
            
    return (epoch, losses.avg)

def validate_contrastive(embedings, experiment, step, args):
    reducer = umap.UMAP(n_components=2)
    for i in range(len(args.crops_for_assign)):
        uembedings = reducer.fit_transform(embedings[i].detach().cpu().numpy())
        plt.scatter(uembedings[:,0], uembedings[:,1], cmap="Spectral")
        #experiment.log_figure(figure=plt, figure_name=f"UMAP contrastvie for crop={i}", step=step)
        plt.clf()

def init_memory(dataloader, model):
    size_memory_per_process = len(dataloader.dataset)
    local_memory_embeddings = torch.zeros(len(args.crops_for_assign), size_memory_per_process, args.feat_dim).cuda()
    local_memory_membership = torch.zeros(len(args.crops_for_assign), size_memory_per_process, args.nmb_prototypes[0]).float().cuda()
    start_idx = 0
    with torch.no_grad():
        logger.info('Start initializing the memory banks')
        for index, inputs in dataloader:
            nmb_unique_idx = inputs[0].size(0)
            index = index.cuda(non_blocking=True)
            
            # get embeddings
            outputs = []
            for crop_idx in args.crops_for_assign:
                inp = inputs[crop_idx].cuda(non_blocking=True)
                outputs.append(model(inp))
            
             # fill the memory bank
            for mb_idx, embeddings in enumerate(outputs):
                local_memory_embeddings[mb_idx][
                    start_idx : start_idx + nmb_unique_idx
                ] = embeddings

                random_num_list = torch.rand((nmb_unique_idx, args.nmb_prototypes[0])).float()
                summation = random_num_list.sum(dim=1)
                temp_list = torch.zeros_like(random_num_list)
                for a in range(random_num_list.shape[0]):
                    temp_list[a] = torch.div(random_num_list[a], summation[a])
                local_memory_membership[mb_idx, start_idx : start_idx + nmb_unique_idx] = temp_list
            start_idx += nmb_unique_idx
    logger.info('Initializion of the memory banks done.')
    return local_memory_embeddings, local_memory_membership


def cluster_memory(local_memory_membership, local_memory_embeddings, nmb_cmeans_iters=100):
    j = 0
    #print(local_memory_embeddings)
    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            for z in range(nmb_cmeans_iters): #TODO bigger values than kmeans
                #calculating the cluster center, is done in every iteration
                centroid_mem_val = local_memory_membership[j,:]
                cluster_centers = torch.zeros(K, args.feat_dim).float().cuda()
                for k in range(K):
                    x = centroid_mem_val[:,k]
                    xraised = torch.pow(x, args.fuzzy_param)
                    denominator = torch.sum(xraised).cuda()
                    temp_num = torch.zeros(x.shape[0], args.feat_dim).float()
                    for i in range(x.shape[0]):
                        data_point = local_memory_embeddings[j,i]
                        temp_num[i] = torch.mul(data_point, xraised[i])
                    
                    numerator = temp_num.sum(dim=0).cuda()
                    cluster_centers[k] = torch.div(numerator, denominator)
                #print(f"Clusters {i_K} {z} {cluster_centers}")
                
                # updating the membership values using the cluster centers
                p = torch.tensor(float(2/(args.fuzzy_param-1))).cuda()
                for i in range(local_memory_embeddings.shape[1]):
                    x = local_memory_embeddings[j,i]
                    distances = x.sub(cluster_centers)
                    distances = torch.linalg.norm(distances, dim=1)
                    for g in range(K):
                        den = torch.pow(torch.div(distances[g],distances), p)
                        den_sum = torch.sum(den)
                        local_memory_membership[j,i,g] = torch.div(torch.tensor(1.), den_sum)
                #print(f"Memberhsip {i_K} {z} {local_memory_membership[j,i]}")
            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)
        
        return local_memory_membership

def triplet_each(local_memory_membership, local_memory_embeddings, emb, start_idx, bs, percent_worst):
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
    loss = 0
    for i, crop_idx in enumerate(args.crops_for_assign):
        emb_crop = emb[crop_idx * bs : (crop_idx + 1) * bs]
        distances = pairwise_distances(emb_crop)
        indices_tuple = torch.zeros((3, emb_crop.shape[0], emb_crop.shape[1]))
        lmemory_membership = local_memory_membership[i, start_idx : start_idx + emb_crop.shape[0]]
        classes_ind = torch.argmax(lmemory_membership, dim=1)
        keep = []
        for em_idx in range(len(emb_crop)):
            em_class = classes_ind[em_idx]
            same_classes = (classes_ind == em_class).nonzero().reshape(-1)
            postive_idx = torch.argmax(distances[em_idx, same_classes])
            not_same_classes = (classes_ind != em_class).nonzero().reshape(-1)
            numb = torch.mul(not_same_classes.shape[0], percent_worst)
            numb = numb.long()
            if same_classes.shape[0] == 0 or not_same_classes.shape[0] == 0:
                logger.info(f"For {crop_idx * bs + em_idx} there are {same_classes.shape[0]} same classes and {not_same_classes.shape[0]} not same classes")
                continue
            elif not_same_classes.shape[0] < 2:
                lowest_classes_idx = not_same_classes
            elif numb < 2:
                lowest_classes_idx = torch.topk(lmemory_membership[not_same_classes,em_class], 2)[1]
            else:
                lowest_classes_idx = torch.topk(lmemory_membership[not_same_classes,em_class], numb)[1]
            negative_idx = torch.argmin(distances[em_idx, lowest_classes_idx])
                
            keep.append(em_idx)
            indices_tuple[0, em_idx] = emb_crop[em_idx]
            indices_tuple[1, em_idx] = emb_crop[postive_idx]
            indices_tuple[2, em_idx] = emb_crop[negative_idx]
        loss += triplet_loss(indices_tuple[0,keep], indices_tuple[1,keep], indices_tuple[2,keep])
        
        # ============ update memory banks ... ============
        embe = emb.detach()
        local_memory_embeddings[i, start_idx : start_idx + bs] = embe[crop_idx * bs : (crop_idx + 1) * bs]
        
    loss /= len(args.crops_for_assign)
    return loss

def triplet_all(local_memory_membership, local_memory_embeddings, emb, start_idx, bs, percent_worst):
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
    distances = pairwise_distances(emb)
    indices_tuple = torch.zeros((3, emb.shape[0], emb.shape[1]))
    lmemory_membership = local_memory_membership[:, start_idx : start_idx + int(len(emb)/len(args.crops_for_assign))]
    classes_ind = torch.argmax(lmemory_membership, dim=2)
    keep = []
    for em_idx in range(len(emb)):
        if em_idx >= len(emb)/len(args.crops_for_assign):
            crope = 1
            mem_idx = em_idx - int(len(emb)/len(args.crops_for_assign))
        else:
            crope = 0
            mem_idx = em_idx
            
        em_class = classes_ind[crope, mem_idx]
        same_classes = (classes_ind[crope] == em_class).nonzero().reshape(-1)
        same_classes_crop = same_classes + int(len(emb)/len(args.crops_for_assign))
        same_classes_whole = torch.cat([same_classes, same_classes_crop], dim=0)
        postive_idx = torch.argmax(distances[em_idx, same_classes_whole])
        
        not_same_classes = (classes_ind[crope] != em_class).nonzero().reshape(-1)
        not_same_classes_crop = not_same_classes + int(len(emb)/len(args.crops_for_assign))
        not_same_classes_whole = torch.cat([not_same_classes, not_same_classes_crop], dim=0)
        
        numb = torch.mul(not_same_classes_whole.shape[0], percent_worst)
        numb = numb.long()
        if same_classes_whole.shape[0] == 0 or not_same_classes_whole.shape[0] == 0:
            logger.info(f"There are {same_classes_whole.shape[0]} same classes and {not_same_classes_whole.shape[0]} not same classes")
            continue
        elif not_same_classes_whole.shape[0] < 2:
            lowest_classes_idx = not_same_classes
        elif numb < 2:
            lowest_classes_idx = torch.topk(lmemory_membership.reshape(-1, lmemory_membership.shape[-1])[not_same_classes,em_class], 2)[1]
        else:
            lowest_classes_idx = torch.topk(lmemory_membership.reshape(-1, lmemory_membership.shape[-1])[not_same_classes,em_class], numb)[1]
        negative_idx = torch.argmin(distances[em_idx, lowest_classes_idx])
                
        keep.append(em_idx)
        indices_tuple[0, em_idx] = emb[em_idx]
        indices_tuple[1, em_idx] = emb[postive_idx]
        indices_tuple[2, em_idx] = emb[negative_idx]
    
    loss = triplet_loss(indices_tuple[0,keep], indices_tuple[1,keep], indices_tuple[2,keep])
    # ============ update memory banks ... ============
    embe = emb.detach()
    for i, crop_idx in enumerate(args.crops_for_assign):
        local_memory_embeddings[i, start_idx : start_idx + bs] = embe[crop_idx * bs : (crop_idx + 1) * bs]

    return loss

if __name__ == "__main__":
    main()

