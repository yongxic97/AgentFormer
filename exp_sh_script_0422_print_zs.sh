#!/bin/bash
program='test.py'

for ((i=0;i<=100;i+=2)); do
    echo "Getting results for # epoch ${i}"
    python ${program} --cfg twop_eth_agentformer_pre --epochs ${i}
done