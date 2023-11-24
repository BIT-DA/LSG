
CUDA_VISIBLE_DEVICES=7 python src/main_lsg.py  --root ./Aircraft --batch_size 24 --logdir lsg/ --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained --gcn_path gcn/aircraft/gcn-aircraft.pkl --seed 2023 --cg 1.0 --cc 8.0 --lr 1e-3 --edge_ratio 0.003
