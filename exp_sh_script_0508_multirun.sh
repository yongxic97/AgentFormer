#!/bin/bash

# sed -i 's/user_give_z_at_test          : true/user_give_z_at_test          : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
# sed -i 's/print_csv                    : true/print_csv                    : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml


## 0508_0101_take2, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0101_take2/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0101_take2


## 0508_0101_take3, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0101_take3/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0101_take3


## 0508_0101_take4, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0101_take4/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0101_take4


## 0508_0101_take5, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0101_take5/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0101_take5

###############################################################################################################

# ignore masked samples when averaging over samples
sed -i 's/    dominant_only             : false/    dominant_only             : true/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml


## 0508_0102_take2, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0102_take2/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0102_take2


## 0508_0102_take3, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0102_take3/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0102_take3


## 0508_0102_take4, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0102_take4/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0102_take4


## 0508_0102_take5, 30-120 eps

# move the pretrained checkpoint 28 to the temp dir where this training is stored.
mv ./results/twop_eth_agentformer_pre/models_0508_0102_take5/* ./results/twop_eth_agentformer_pre/models
# train
python train.py --cfg twop_eth_agentformer_pre --start_epoch 28
# move all the trained checkpoints to that directory
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0508_0102_take5

