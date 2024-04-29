python train.py --cfg twop_eth_agentformer_pre --start_epoch 14

mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0418_0101

sed -i 's/twop                         : true/twop                         : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

python train.py --cfg twop_eth_agentformer_pre