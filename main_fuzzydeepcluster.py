# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import math
import os
import shutil
import time
from logging import getLogger

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
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
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

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=1e10, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
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
    train_dataset = MultiCropDataset(
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
        nmb_prototypes=None,
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
    logger.info("Building contrastive optimizer done.")
    
    torch.nn.TripletMarginLoss(margin=1.0, p=2.0,
    
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

    cudnn.benchmark = True
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_embeddings, local_memory_membership = train_backbone(
            train_loader,
            model,
            optimizer_contr,
            epoch,
            local_memory_embeddings,
            local_memory_membership
        )
        training_stats.update(scores)

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
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint_backbone.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-backbone-" + str(epoch) + ".pth"),
                )
        torch.save({"local_memory_embeddings": local_memory_embeddings,
                    "local_memory_membership": local_memory_membership}, mb_path)
    
    # add head
    model.add_prototypes(args.nmb_prototypes)
    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False
        
        model.prototypes.weight.requires_grad = True
        model.prototypes.bias.requires_grad = True
    
    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")
    
    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
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
    
    for epoch in range(40):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)
        
         # train the network
        scores = train_head(
            optimizer,
            model,
            epoch,
            lr_schedule,
            local_memory_embeddings,
            local_memory_membership
        )
        training_stats.update(scores)
        
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

def train_backbone(loader, model, optimizer, epoch, local_memory_embeddings, local_memory_membership):
    model.train()
    losses = AverageMeter()

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
    cluster_memory(model, local_memory_embeddings, len(loader.dataset))
    logger.info('Clustering for epoch {} done.'.format(epoch))

    start_idx = 0
    for it, (idx, inputs) in enumerate(loader):
        # ============ multi-res forward passes ... ============
        emb = model(inputs)
        emb = emb.detach()
        bs = inputs[0].size(0)
        
        # ============ Triplet loss ... ============
        # order embd based on fuzzy clustering with centroids
        indices_tuple = [[], [], []]
        indexes_positive_used = {x: [] for x in range(local_memory_membership.shape[-1])}
        indexes_negative_used = {x: [] for x in range(local_memory_membership.shape[-1])}
        for em_idx in range(len(emb)):
           em_class = torch.argmax(local_memory_membership[:, em_idx, :].mean(dim=1).mean(dim=0))
           local_memory_membership_without = torch.cat([local_memory_membership[:, 0:em_idx], local_memory_membership[:,em_idx+1:]])
           classes_value = local_memory_membership_without.mean(dim=3).mean(dim=0).max(dim=1)
           classes_ind = torch.argmax(local_memory_membership_without.mean(dim=3).mean(dim=0), dim=1)
           same_classes = (classes_ind == em_class).nonzero()
           not_used_same_classes = same_classes[~indexes_positive_used[em_class]]
           postive_idx = torch.argmin(classes_value[not_used_same_classes])
           not_same_classes = (classes_ind != em_class).nonzero()
           not_used_not_same_classes = not_same_classes[~indexes_negative_used[em_class]]
           classes_value_specific = local_memory_membership_without[:,:,:,em_class].mean(dim=2).mean(dim=0)
           negative_idx = torch.argmax(classes_value_specific[not_used_not_same_classes])
           
           indices_tuple[0].append(emb[em_idx])
           indices_tuple[0].append(emb[postive_idx])
           indices_tuple[0].append(emb[negative_idx])
           indexes_positive_used[em_class].append(postive_idx)
           indexes_negative_used[em_class].append(negative_idx)
           
        loss = triplet_loss(indices_tuple[0], indices_tuple[1], indices_tuple[2])
        loss.backward()
        optimizer.step()

        # ============ update memory banks ... ============
        for i, crop_idx in enumerate(args.crops_for_assign):
            local_memory_embeddings[i][start_idx : start_idx + bs] = \
                emb[crop_idx * bs : (crop_idx + 1) * bs]
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        if args.rank ==0 and it % 50 == 0:
            logger.info(f'Train: Epoch [{epoch}], Step [{idx}/{len(loader)}] loss: {loss.item():.3f}')
    return (epoch, losses.avg), local_memory_embeddings, local_memory_membership

def train_head(loader, optimizer, model, epoch, schedule, local_memory_embeddings, local_memory_membership):
    model.train()
    losses = AverageMeter()
    
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
    logger.info('Clustering for epoch {} done.'.format(epoch))

    assignments = get_clusters(local_memory_membership, len(loader.dataset))
    for idx in enumerate(local_memory_embeddings.shape[1]): #TODO jakis batch
        # update learning rate
        iteration = epoch * len(loader) + idx
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]
            
         # ============ multi-res forward passes ... ============
        inputs = local_memory_embeddings[:,idx,].cuda()
        _, output = model(inputs)
        
        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / args.temperature
            targets = assignments[h][idx].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inputs[0].size(0))
        if args.rank ==0:
            logger.info(f'Train: Epoch [{epoch}], Step [{idx}/{len(loader)}] loss: {loss.item():.3f}')
            
    return (epoch, losses.avg)
    
def init_memory(dataloader, model):
    size_memory_per_process = len(dataloader) * args.batch_size
    local_memory_index = torch.zeros(size_memory_per_process).long().cuda()
    local_memory_embeddings = torch.zeros(len(args.crops_for_assign), size_memory_per_process, args.feat_dim).cuda()
    local_memory_membership = torch.zeros(len(args.crops_for_assign), size_memory_per_process, len(args.nmb_prototypes), args.nmb_prototypes[0]).long().cuda()
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
            local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
            for mb_idx, embeddings in enumerate(outputs):
                local_memory_embeddings[mb_idx][
                    start_idx : start_idx + nmb_unique_idx
                ] = embeddings
            
            # fill the membership matrix
            for i_K, K in enumerate(args.nmb_prototypes):
                random_num_list = torch.rand(args.crops_for_assign, nmb_unique_idx, K) #TODO check
                summation = random_num_list.sum(dim=2)
                temp_list =  torch.div(random_num_list, summation, dim=2)
                local_memory_membership[:, start_idx : start_idx + nmb_unique_idx, i_K] = temp_list
            
            start_idx += nmb_unique_idx
    logger.info('Initializion of the memory banks done.')
    return local_memory_index, local_memory_embeddings, local_memory_membership


def cluster_memory(model, local_memory_index, local_memory_membership, local_memory_embeddings, size_dataset, nmb_cmeans_iters=10):
    cluster_centers = torch.zeros(len(args.nmb_prototypes), args.nmb_prototypes, args.feat_dim).long()
    with torch.no_grad():
        for n_iter in range(nmb_cmeans_iters): #TODO bigger values than kmeans
            # run distributed c-means
            j = 0
            for i_K, K in enumerate(args.nmb_prototypes):
                # init centroids with elements from memory bank of rank 0
                centroids = torch.empty(K, args.feat_dim).cuda(non_blocking=True)
                if args.rank == 0:
                    random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                    assert len(random_idx) >= K, "please reduce the number of centroids"
                    centroids = local_memory_embeddings[j][random_idx]
                
                centroid_mem_val = local_memory_membership[j, :, i_K]
                xraised = torch.pow(centroid_mem_val, args.fuzzy_param)
                denominator = torch.sum(xraised, dim=1)
                temp_num = torch.zeros(centroid_mem_val.shape[0], args.feat_dim)
                for index in range(centroid_mem_val.shape[0]):
                    data_point = local_memory_embeddings[j, index]
                    prod = torch.mul(data_point, xraised[index])
                    temp_num[index] = prod
                
                numerator = map(sum, zip(*temp_num))
                center = torch.div(numerator, denominator)
                cluster_centers[i_K] = nn.functional.normalize(center, p=2)
                
                # updating the membership values using the cluster centers
                p = torch.tensor(float(2/(args.fuzzy_param-1)))
                for index in range(centroid_mem_val.shape[0]):
                    data_point = local_memory_embeddings[j, index]
                    distances = torch.tensor([np.linalg.norm(map(operator.sub, data_point, cluster_centers[h])) for h in range(len(args.nmb_prototypes))])
                    for cluster_idx in range(len(args.nmb_prototypes)):
                        den = torch.sum(torch.pow(torch.div(distances[cluster_idx], distances), p))
                        local_memory_membership[j, index, cluster_idx] = torch.div(torch.tensor(1), den).float()
                    
                # next memory bank to use
                j = (j + 1) % len(args.crops_for_assign)

def get_clusters(local_memory_membership, size_dataset):
    assignments = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    j = 0
    for i_K, K in enumerate(args.nmb_prototypes):
        for index in range(size_dataset):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(local_memory_membership[j, index]))
            assignments[i_K, index] = torch.tensor(idx)
        j = (j + 1) % len(args.crops_for_assign)
    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":
    main()

