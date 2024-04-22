#!/bin/bash

python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0419_0103_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0103_take2


python train.py --cfg twop_eth_agentformer_pre
mkdir ./results/twop_eth_agentformer_pre/models_0419_0103_take3
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0103_take3
