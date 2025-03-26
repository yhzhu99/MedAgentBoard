llm_model_name="deepseek-v3-official"
vllm_model_name="qwen-vl-max"
meta_model_name="deepseek-v3-official"
decision_model_name="deepseek-v3-official"
doctor1_model_name_llm="deepseek-v3-official"
doctor2_model_name_llm="deepseek-v3-official"
doctor3_model_name_llm="deepseek-v3-official"
doctor1_model_name_vllm="qwen-vl-max"
doctor2_model_name_vllm="qwen-vl-max"
doctor3_model_name_vllm="qwen-vl-max"

# # === 1. MedQA ===
# # === Single LLM with 0-shot for QA ===
# # === MedQA ===
# python -m medqa.single_llm \
#     --dataset "MedQA" \
#     --prompt_type "zero_shot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === PubMedQA ===
# python -m medqa.single_llm \
#     --dataset "PubMedQA" \
#     --prompt_type "zero_shot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === Single LLM with few-shot for QA ===
# # === MedQA ===
# python -m medqa.single_llm \
#     --dataset "MedQA" \
#     --prompt_type "few_shot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === PubMedQA ===
# python -m medqa.single_llm \
#     --dataset "PubMedQA" \
#     --prompt_type "few_shot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === Single LLM with cot for QA ===
# # === MedQA ===
# python -m medqa.single_llm \
#     --dataset "MedQA" \
#     --prompt_type "cot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === PubMedQA ===
# python -m medqa.single_llm \
#     --dataset "PubMedQA" \
#     --prompt_type "cot" \
#     --model_name "${llm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === MedAgent for QA ===
# # === MedQA ===
# python -m medqa.multi_agent_medagent \
#     --dataset "MedQA" \
#     --meta_model ${meta_model_name} \
#     --decision_model ${decision_model_name} \
#     --doctor_models ${doctor1_model_name_llm} ${doctor2_model_name_llm} ${doctor3_model_name_llm} \
#     --start_pos 0 \
#     --end_pos 2

# # === PubMedQA ===
# python -m medqa.multi_agent_medagent \
#     --dataset "PubMedQA" \
#     --meta_model ${meta_model_name} \
#     --decision_model ${decision_model_name} \
#     --doctor_models ${doctor1_model_name_llm} ${doctor2_model_name_llm} ${doctor3_model_name_llm} \
#     --start_pos 0 \
#     --end_pos 2


# # === ColaCare for QA ===
# # === MedQA ===
# python -m medqa.multi_agent_colacare \
#     --dataset "MedQA" \
#     --meta_model ${meta_model_name} \
#     --decision_model ${decision_model_name} \
#     --doctor_models ${doctor1_model_name_llm} ${doctor2_model_name_llm} ${doctor3_model_name_llm} \
#     --start_pos 0 \
#     --end_pos 2

# # === PubMedQA ===
# python -m medqa.multi_agent_colacare \
#     --dataset "PubMedQA" \
#     --meta_model ${meta_model_name} \
#     --decision_model ${decision_model_name} \
#     --doctor_models ${doctor1_model_name_llm} ${doctor2_model_name_llm} ${doctor3_model_name_llm} \
#     --start_pos 0 \
#     --end_pos 2

# # === 2. MedVQA ===
# # === Single LLM with 0-shot for VQA ===
# # === Path-VQA ===
# python -m medqa.single_vllm \
#     --dataset "Path_VQA" \
#     --prompt_type "zero_shot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === VQA-Rad ===
# python -m medqa.single_vllm \
#     --dataset "VQA_Rad" \
#     --prompt_type "zero_shot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === Single LLM with few-shot for VQA ===
# # === Path-VQA ===
# python -m medqa.single_vllm \
#     --dataset "Path_VQA" \
#     --prompt_type "few_shot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === VQA-Rad ===
# python -m medqa.single_vllm \
#     --dataset "VQA_Rad" \
#     --prompt_type "few_shot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === Single LLM with cot for VQA ===
# # === Path-VQA ===
# python -m medqa.single_vllm \
#     --dataset "Path_VQA" \
#     --prompt_type "cot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# # === VQA-Rad ===
# python -m medqa.single_vllm \
#     --dataset "VQA_Rad" \
#     --prompt_type "cot" \
#     --model_name "${vllm_model_name}" \
#     --start_pos 0 \
#     --end_pos 2

# === MedAgent for VQA ===
# === Path-VQA ===
python -m medqa.multi_agent_medagent \
    --dataset "Path_VQA" \
    --meta_model ${meta_model_name} \
    --decision_model ${decision_model_name} \
    --doctor_models ${doctor1_model_name_vllm} ${doctor2_model_name_vllm} ${doctor3_model_name_vllm} \
    --start_pos 2 \
    --end_pos 3

# === VQA-Rad ===
python -m medqa.multi_agent_medagent \
    --dataset "VQA_Rad" \
    --meta_model ${meta_model_name} \
    --decision_model ${decision_model_name} \
    --doctor_models ${doctor1_model_name_vllm} ${doctor2_model_name_vllm} ${doctor3_model_name_vllm} \
    --start_pos 0 \
    --end_pos 1

# === ColaCare for VQA ===
# === Path-VQA ===
python -m medqa.multi_agent_colacare \
    --dataset "Path_VQA" \
    --meta_model ${meta_model_name} \
    --decision_model ${decision_model_name} \
    --doctor_models ${doctor1_model_name_vllm} ${doctor2_model_name_vllm} ${doctor3_model_name_vllm} \
    --start_pos 2 \
    --end_pos 3

# === VQA-Rad ===
python -m medqa.multi_agent_colacare \
    --dataset "VQA_Rad" \
    --meta_model ${meta_model_name} \
    --decision_model ${decision_model_name} \
    --doctor_models ${doctor1_model_name_vllm} ${doctor2_model_name_vllm} ${doctor3_model_name_vllm} \
    --start_pos 0 \
    --end_pos 1