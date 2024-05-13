#!/bin/bash

## 0507_0204_take2, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0507_0204_take2/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0507_0204_take2


## 0507_0204_take3, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0507_0204_take3/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0507_0204_take3


## 0507_0204_take4, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0507_0204_take4/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0507_0204_take4
