python train.py --cfg twop_eth_agentformer_pre
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0426_0102_take1

python train.py --cfg twop_eth_agentformer_pre
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0426_0102_take2

sed -i 's/pretrain                     : false/pretrain                     : true/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
python train.py --cfg twop_eth_agentformer_pre
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0426_0103_take1

python train.py --cfg twop_eth_agentformer_pre
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0426_0103_take2
