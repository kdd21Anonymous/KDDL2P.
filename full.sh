GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset wisconsin
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset wisconsin
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset wisconsin

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset film
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset film
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset film

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset cornell
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset cornell
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset cornell

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset texas
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset texas
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset texas

