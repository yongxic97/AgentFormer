python train.py --cfg twop_eth_agentformer_pre --start_epoch 2

mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0103

python train.py --cfg twop_eth_agentformer_pre

mkdir ./results/twop_eth_agentformer_pre/models_0419_0103_take2

mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0103_take2



sed -i 's/tf_dropout: 0.15/tf_dropout: 0.1/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

python train.py --cfg twop_eth_agentformer_pre

mkdir ./results/twop_eth_agentformer_pre/models_0419_0105
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0105

mkdir ./results/twop_eth_agentformer_pre/models_0419_0105_take2
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0419_0105_take2
