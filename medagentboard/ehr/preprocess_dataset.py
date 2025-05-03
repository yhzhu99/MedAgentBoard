import pandas as pd
from medagentboard.utils.json_utils import save_json, load_json, load_jsonl

def curate_instructions_section(text):
    start_marker = "Instructions & Output Format:"
    end_marker = "floating-point number"

    start_index = text.find(start_marker)
    if start_index == -1:
        return text  # Start marker not found

    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return text  # End marker not found

    # Keep everything before the start marker and after the end marker
    text =  text[:start_index] + "Please output a " + text[end_index:]

    start_marker = "Example Format:"
    end_marker = "Now, please analyze"

    start_index = text.find(start_marker)
    if start_index == -1:
        return text  # Start marker not found

    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return text  # End marker not found

    # Keep everything before the start marker and after the end marker
    text =  text[:start_index] + text[end_index:]

    updated_instruction = text.replace("System Prompt: ", "").replace("User Prompt: ", "").replace("years Your Task: ", "years. ")
    return updated_instruction

# structured EHR
datasets = ["tjh", "mimic-iv"]
tasks = {
    "tjh": ["mortality"],
    "mimic-iv": ["mortality", "readmission"],
}

for dataset in datasets:
    for task in tasks[dataset]:
        print(f"Processing {dataset} {task}")
        processed_data = []

        test_data = pd.read_pickle(f"my_datasets/raw/structured_ehr/{dataset}/{task}/test_data.pkl")
        for item in test_data:
            qid = item['id']
            question = item['x_ehr_prompt']
            question = curate_instructions_section(question)
            answer = item[f'y_{task}'][0]
            processed_data.append(
                {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                }
            )
        save_json(processed_data, f"my_datasets/processed/ehr/{dataset}/ehr_timeseries_{task}_test.json")