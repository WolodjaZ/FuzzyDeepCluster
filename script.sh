python3 -m torch.distributed.launch --nproc_per_node=1 main_fuzzydeepcluster.py --data_path /cgtvx/imagenet --nmb_crops 2 --size_crops 64 --min_scale_crops 0.08 --max_scale_crops 1. --crops_for_assign 0 1 --temperature 0.1 --feat_dim 128 --nmb_prototypes 4 4 --epochs 2 --epochs_con 2 --batch_size 4 --base_lr 4.8 --final_lr 0.0048 --wd 0.000001 --warmup_epochs 10 --start_warmup 0.3 --arch resnet50 --dump_path ./experiments
