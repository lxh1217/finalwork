@echo off
REM DiffKG 训练命令（Windows）

cd src
python main.py --model_name DiffKG --dataset Grocery_and_Gourmet_Food --emb_size 64 --gnn_layer 2 --layer_num_kg 1 --noise_scale 0.1 --steps 5 --lr 1e-3 --l2 1e-7 --batch_size 512 --diffusion_batch_size 512 --epoch 5 --test_epoch 1 --topk 10,20


