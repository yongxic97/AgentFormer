#!/bin/bash

# sed -i 's/user_give_z_at_test          : true/user_give_z_at_test          : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
# sed -i 's/print_csv                    : true/print_csv                    : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

## First run three more 50 epochs for op.weight = 16

# 1. 0523_0101_take2
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0101_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0101_take2

# 2. 0523_0101_take3
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0101_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0101_take3

# 3. 0523_0101_take4
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0101_take4
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0101_take4


## Then run three more 50 epochs for op.weight = 8
sed -i 's/    weight: 16.0/    weight: 8.0/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

# 1. 0523_0103_take1
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0103_take1
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0103_take1

# 2. 0523_0103_take2
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0103_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0103_take2

# 3. 0523_0103_take3
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0103_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0103_take3