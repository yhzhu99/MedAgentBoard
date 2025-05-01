#!/bin/bash

# 设置并发数
MAX_CONCURRENT=4
CURRENT_JOBS=0

# 创建一个临时文件来跟踪正在运行的进程
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

# 运行命令并管理并发
run_command() {
    local cmd="$1"

    # 检查当前运行的进程数
    CURRENT_JOBS=$(wc -l < $TEMP_FILE)

    # 如果当前运行的进程数达到最大值，等待一个进程完成
    while [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; do
        for pid in $(cat $TEMP_FILE); do
            if ! kill -0 $pid 2>/dev/null; then
                # 进程已结束，从文件中删除
                grep -v "^$pid$" $TEMP_FILE > ${TEMP_FILE}.new
                mv ${TEMP_FILE}.new $TEMP_FILE
            fi
        done
        CURRENT_JOBS=$(wc -l < $TEMP_FILE)
        if [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; then
            sleep 1
        fi
    done

    # 运行命令
    echo "Running: $cmd"
    eval "$cmd" &

    # 记录进程ID
    echo $! >> $TEMP_FILE
}

# 定义所有数据集
DATASETS=("PLABA" "cochrane" "elife" "med_easi" "plos_genetics")

# 定义single_llm的prompt类型
PROMPT_TYPES=("basic" "optimized" "few_shot")

# 定义AgentSimp的参数组合
COMMUNICATION_TYPES=("pipeline" "synchronous")
CONSTRUCTION_TYPES=("direct" "iterative")

# 定义模型
MODEL="deepseek-v3-official"

echo "Starting experiments..."

# 1. Single LLM with different prompting strategies
for dataset in "${DATASETS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        cmd="python -m medagentboard.laysummary.single_llm --dataset $dataset --prompt_type $prompt_type --model_key $MODEL"
        run_command "$cmd"
    done
done

# 2. AgentSimp with default configuration
for dataset in "${DATASETS[@]}"; do
    cmd="python -m medagentboard.laysummary.multi_agent_agentsimp --dataset $dataset --model $MODEL"
    run_command "$cmd"
done

# 等待所有任务完成
wait

echo "All experiments completed!"