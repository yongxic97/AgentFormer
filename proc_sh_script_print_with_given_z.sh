#!/bin/bash
z_begin=1
z_step=4
z_end=9
len=10
eps_start=6
eps_end=6

for ((i=eps_start;i<=eps_end;i+=2)); do
    for ((z=z_begin;z<=z_end;z+=z_step)); do
        result=$(echo "scale=2; $z / $len" | bc)
        echo "Getting results for z=${result}, at # epoch ${i}"
        python test.py --cfg twop_eth_agentformer_pre --user_z ${result} --epochs ${i}
    done
done

# for ((i=eps_start;i<=eps_end;i+=2)); do
#     python test.py --cfg twop_eth_agentformer_pre --epochs ${i}
# done