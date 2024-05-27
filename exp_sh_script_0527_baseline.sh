#!/bin/bash

# 1. 0523_0000_take1
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0000_take1
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0000_take1


# 2. 0523_0000_take2
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0000_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0000_take2

# 3. 0523_0000_take3
# train
python train.py --cfg twop_eth_agentformer_pre
# move all the trained checkpoints to that directory
mkdir ./results/twop_eth_agentformer_pre/models_0523_0000_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0523_0000_take3
