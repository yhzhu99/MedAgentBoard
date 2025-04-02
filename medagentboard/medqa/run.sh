#!/bin/bash

# 设置并发数
MAX_CONCURRENT=3
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

# 定义数据集和任务类型
QA_DATASETS=("MedQA" "PubMedQA")
VQA_DATASETS=("PathVQA" "VQA-RAD")

# 为每个数据集定义可用的qa_type
declare -A DATASET_QA_TYPES
DATASET_QA_TYPES[MedQA]="mc"
DATASET_QA_TYPES[PubMedQA]="mc ff"
DATASET_QA_TYPES[PathVQA]="mc"
DATASET_QA_TYPES[VQA-RAD]="mc ff"

echo "Starting experiments..."

# 1. ColaCare
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_colacare --dataset $dataset --qa_type $qa_type --meta_model deepseek-v3-ark --doctor_models deepseek-v3-ark deepseek-v3-ark deepseek-v3-ark"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_colacare --dataset $dataset --qa_type $qa_type --meta_model deepseek-v3-ark --doctor_models qwen-vl-max qwen-vl-max qwen-vl-max"
        run_command "$cmd"
    done
done

# 2. MDAgents
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_mdagents --dataset $dataset --qa_type $qa_type --model deepseek-v3-ark --meta_model deepseek-v3-ark --decision_model deepseek-v3-ark"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_mdagents --dataset $dataset --qa_type $qa_type --model qwen-vl-max --meta_model deepseek-v3-ark --decision_model qwen-vl-max"
        run_command "$cmd"
    done
done

# 3. MedAgent
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_medagent --dataset $dataset --qa_type $qa_type --moderator_model deepseek-v3-ark --recruiter_model deepseek-v3-ark --agent_model deepseek-v3-ark"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_medagent --dataset $dataset --qa_type $qa_type --moderator_model deepseek-v3-ark --recruiter_model deepseek-v3-ark --agent_model qwen-vl-max"
        run_command "$cmd"
    done
done

# 4. ReConcile
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_reconcile --dataset $dataset --qa_type $qa_type --agents deepseek-v3-ark qwen-max-latest qwen-vl-max --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentboard.medqa.multi_agent_reconcile --dataset $dataset --qa_type $qa_type --agents qwen-vl-max qwen2.5-vl-32b-instruct qwen2.5-vl-72b-instruct --max_rounds 3"
        run_command "$cmd"
    done
done

# 5. Single LLM with different prompting strategies
PROMPT_TYPES=("zero_shot" "few_shot" "cot" "self_consistency" "cot_sc")

for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            cmd="python -m medagentboard.medqa.single_llm --dataset $dataset --qa_type $qa_type --prompt_type $prompt_type --model_key deepseek-v3-ark"
            run_command "$cmd"
        done
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            cmd="python -m medagentboard.medqa.single_llm --dataset $dataset --qa_type $qa_type --prompt_type $prompt_type --model_key qwen-vl-max"
            run_command "$cmd"
        done
    done
done

# 等待所有任务完成
wait

echo "All experiments completed!"