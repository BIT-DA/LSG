# stage one: train a gcn on semantic graph

python src/main_gcn.py --root ./Aircraft --batch_size 24 --logdir gcn --seed 2023 --max_iter 5005 --lr 0.001 --edge_ratio 0.003
python src/main_gcn.py --root ./StanfordCars --batch_size 24 --logdir gcn --seed 2023 --max_iter 5005 --lr 0.001 --edge_ratio 0.003
python src/main_gcn.py --root ./CUB200 --batch_size 24 --logdir gcn --seed 2023 --max_iter 5005 --lr 0.001 --edge_ratio 0.003

