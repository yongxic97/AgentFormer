#!/bin/bash
z_begin=1
z_step=1
z_end=9
len=10
eps_start=30
eps_end=92
step_len=8

for ((i=eps_end;i>=eps_start;i-=step_len)); do
    for ((z=z_begin;z<=z_end;z+=z_step)); do
        result=$(echo "scale=2; $z / $len" | bc)
        echo "Getting results for z=${result}, at # epoch ${i}"
        python test.py --cfg twop_eth_agentformer_pre --user_z ${result} --epochs ${i}
    done
done

# for ((i=eps_start;i<=eps_end;i+=2)); do
#     python test.py --cfg twop_eth_agentformer_pre --epochs ${i}
# done