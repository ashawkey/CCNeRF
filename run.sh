# training on single objects
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train.py --config configs/hotdog_hybrid.txt
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train.py --config configs/chair_hybrid.txt
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ficus_hybrid.txt

# testing composition (modify the composition settings in compose.py)
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python compose.py --config configs/hotdog_hybrid.txt --ckpt none
