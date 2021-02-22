GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset Cora
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset Cora
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset Cora

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset CiteSeer
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset CiteSeer
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset CiteSeer

CUDA_VISIBLE_DEVICES=$GPU python -u L2D.py --dataset PubMed
CUDA_VISIBLE_DEVICES=$GPU python -u L2Q.py --dataset PubMed
CUDA_VISIBLE_DEVICES=$GPU python -u L2S.py --dataset PubMed