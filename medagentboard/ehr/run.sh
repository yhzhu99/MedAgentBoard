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

# 定义数据集和任务类型
EHR_DATASETS=("mimic-iv" "tjh")

# 为每个数据集定义可用的task类型
declare -A DATASET_TASKS
DATASET_TASKS[mimic-iv]="mortality readmission"
DATASET_TASKS[tjh]="mortality"  # tjh只有mortality任务

echo "Starting EHR experiments..."

# 1. ColaCare
for dataset in "${EHR_DATASETS[@]}"; do
    for task in ${DATASET_TASKS[$dataset]}; do
        cmd="python -m medagentboard.ehr.multi_agent_colacare --dataset $dataset --task $task"
        run_command "$cmd"
    done
done

# 2. MedAgent
for dataset in "${EHR_DATASETS[@]}"; do
    for task in ${DATASET_TASKS[$dataset]}; do
        cmd="python -m medagentboard.ehr.multi_agent_medagent --dataset $dataset --task $task"
        run_command "$cmd"
    done
done

# 3. ReConcile
for dataset in "${EHR_DATASETS[@]}"; do
    for task in ${DATASET_TASKS[$dataset]}; do
        cmd="python -m medagentboard.ehr.multi_agent_reconcile --dataset $dataset --task $task"
        run_command "$cmd"
    done
done





# 等待所有任务完成
wait

echo "All EHR experiments completed!"