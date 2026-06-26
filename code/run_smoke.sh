#!/bin/bash
# FDRE-HRDC Smoke实验: 3 iterations x 300k steps, Eureka环境, DeepSeek API

export DEEPSEEK_API_KEY="your-deepseek-api-key-here"

cd /c/Users/Administrator/Desktop/FDRE_HRDC_最终交付_20260614/code

echo "============================================"
echo "FDRE-HRDC Smoke 实验启动"
echo "配置: 3轮迭代 x 300k步, 1 seed, DeepSeek API"
echo "启动时间: $(date)"
echo "============================================"

python scripts/run_experiment.py \
    --config configs/lunarlander_smoke_3x300k.json \
    --llm-provider deepseek \
    --llm-model deepseek-chat

echo ""
echo "============================================"
echo "实验结束: $(date)"
echo "输出目录: outputs/lunarlander_smoke_3x300k"
echo "============================================"
