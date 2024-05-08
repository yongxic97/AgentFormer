#!/bin/bash

# 0506_0102_take2, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre --start_epoch 8
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take2

# 0506_0102_take3, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take3

# 0506_0102_take4, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take4
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take4

# 0506_0102_take5, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take5
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take5

# 0506_0102_take6, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take6
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take6

# 0506_0102_take7, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take7
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take7

# 0506_0102_take8, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take8
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take8

# 0506_0102_take9, tentatively train for 10 epochs
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take9
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take9

# 0506_0102_take10, train for 100 epochs
sed -i 's/num_epochs                   : 30/num_epochs                   : 100/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0506_0102_take10
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0506_0102_take10
