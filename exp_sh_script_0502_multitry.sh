python train.py --cfg twop_eth_agentformer_pre --start_epoch 38

# 0502_0101
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0502_0101_take1

sed -i 's/user_give_z_at_test          : true/user_give_z_at_test          : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
sed -i 's/print_csv                    : true/print_csv                    : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
sed -i 's/learn_prior                  : true/learn_prior                  : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml
sed -i 's/pretrain                     : true/pretrain                     : false/' ./cfg/eth_ucy/eth/twop_eth_agentformer_pre.yml

python train.py --cfg twop_eth_agentformer_pre

mkdir ./results/twop_eth_agentformer_pre/models_0502_0102_take1
# 0502_0102
mv ./results/twop_eth_agentformer_pre/models/* ./results/twop_eth_agentformer_pre/models_0502_0102_take1
