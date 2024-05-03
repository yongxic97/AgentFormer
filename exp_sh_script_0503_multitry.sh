#!/bin/bash


# 0503_0102_take1
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0503_0102_take1
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0102_take1

# 0503_0102_take2
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0503_0102_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0102_take2

# 0503_0102_take3
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0503_0102_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0102_take3


sed -i 's/    weight: 1.0/    weight: 2.0/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

# 0503_0202_take1
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0503_0202_take1
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0202_take1



# # 0503_0102_take1
# python train.py --cfg twop_eth_agentformer_pre
# mkdir ./results/twop_eth_agentformer_pre/models_0503_0102_take1
# mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0102_take1

# ## if time is enough
# sed -i 's/z_type                       : 'beta'/z_type                       : 'gaussian'/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

# # 0503_0101_take2
# python train.py --cfg twop_eth_agentformer_pre
# mkdir ./results/twop_eth_agentformer_pre/models_0503_0101_take2
# mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0101_take2


# sed -i 's/z_type                       : 'gaussian'/z_type                       : 'beta'/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

# # 0503_0102_take2
# python train.py --cfg twop_eth_agentformer_pre
# mkdir ./results/twop_eth_agentformer_pre/models_0503_0102_take2
# mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0503_0102_take2
