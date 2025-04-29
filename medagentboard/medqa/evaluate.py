import os
import json
import numpy as np
import argparse
from medagentboard.utils.llm_scoring import llm_score
from medagentboard.utils.json_utils import preprocess_response_string

def bootstrap(data_path):
    """
    Bootstrap the sample qids with replacement, the size of the sample is the same as the original data.
    """
    data_qids = []
    with open(data_path, "r") as f:
        data = json.load(f)
        for datum in data:
            data_qids.append(datum["qid"])
            
    return [data_qids[i] for i in np.random.randint(0, len(data_qids), len(data_qids))]

def extract_digits(string):
    """
    Extract digits from a string.
    """
    return "".join(filter(str.isdigit, string))

def add_llm_score_to_json(json_file_path: str, llm_score: str):
    """
    Reads a JSON file, adds an 'llm_score' key at the top level, 
    and saves the modified data back to the original file path.

    Args:
        json_file_path: The path to the JSON file to modify.
        llm_score: The score string to add under the 'llm_score' key.

    Raises:
        FileNotFoundError: If the specified json_file_path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        Exception: For other potential I/O or unexpected errors.
    """
    try:
        # 1. Read the existing JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Add the llm_score key-value pair
        # Ensure it's treated as a string if that's the requirement,
        # otherwise, you might want to convert llm_score to int/float earlier.
        data['llm_score'] = llm_score 

        # 3. Write the modified data back to the original file
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # Use indent for readability; ensure_ascii=False for wider char support
            json.dump(data, f, indent=2, ensure_ascii=False) 
            
        # print(f"Successfully added 'llm_score' to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{json_file_path}'")
        raise # Re-raise the exception if you want the caller to handle it
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{json_file_path}'. Is it a valid JSON file?")
        raise # Re-raise
    except Exception as e:
        print(f"An unexpected error occurred while processing '{json_file_path}': {e}")
        raise # Re-raise

def extract_score_from_llm_output(output_string: str) -> int | None:
    """
    Extracts the first integer value associated with the key "score" 
    from a potentially malformed JSON-like string output by an LLM.

    It specifically looks for '"score":' followed by optional whitespace 
    and then an integer. It does not rely on full JSON parsing.

    Args:
        output_string: The string output from the LLM, expected to contain 
                       a '"score": <number>' pattern.

    Returns:
        The extracted integer score if found, otherwise None.
    """
    if not output_string:
        return None

    # Option 1: Using string manipulation (more step-by-step)
    try:
        # Find the position of '"score":'
        score_key = '"score":'
        key_index = output_string.find(score_key)
        
        if key_index == -1:
            # Try with single quotes as a fallback, as LLMs might hallucinate them
            score_key = "'score':" 
            key_index = output_string.find(score_key)
            if key_index == -1:
                return None # Key not found

        # Start searching for the number right after '"score":'
        start_search_index = key_index + len(score_key)
        
        # Skip any whitespace characters immediately after the colon
        num_start_index = start_search_index
        while num_start_index < len(output_string) and output_string[num_start_index].isspace():
            num_start_index += 1
            
        if num_start_index == len(output_string):
            return None # Reached end of string without finding a number

        # Extract consecutive digits
        num_end_index = num_start_index
        while num_end_index < len(output_string) and output_string[num_end_index].isdigit():
            num_end_index += 1
            
        # If no digits were found right after skipping whitespace
        if num_end_index == num_start_index:
            return None 
            
        # Extract the number string and convert to int
        number_str = output_string[num_start_index:num_end_index]
        return int(number_str)
        
    except Exception:
        # Catch any unexpected errors during string processing
        return None

if __name__ == "__main__":
    np.random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="logs/medqa")
    parser.add_argument("--bootstrap", type=bool, default=True, help="Whether to bootstrap the data")
    parser.add_argument("--n_bootstrap", type=int, default=10, help="Number of bootstrap samples")
    parser.add_argument("--judge_model", type=str, default="deepseek-v3-ark", help="LLM used for free-form evaluation")
    args = parser.parse_args()
    
    # Loop through the datasets
    for dataset_dir in os.listdir(args.logs_dir):
        if not os.path.isdir(os.path.join(args.logs_dir, dataset_dir)):
            continue
        
        dataset = dataset_dir
        print(f"Dataset: {dataset}")
        
        # Loop through the question types
        for qtype_dir in os.listdir(os.path.join(args.logs_dir, dataset)):
            if not os.path.isdir(os.path.join(args.logs_dir, dataset, qtype_dir)):
                continue
            
            qtype = qtype_dir
            print(f"Question Type: {qtype}")
            data_path = f"my_datasets/processed/{dataset}/medqa_{"mc" if qtype == "multiple_choice" else "ff"}.json"
            qids = []
            MODEL_WIDTH = 27
            METRIC_WIDTH = 10
            VALUE_WIDTH = 8 
            TOTAL_WIDTH = 8
            
            print(f"{'Model':<{MODEL_WIDTH}} | {'Metric':<{METRIC_WIDTH}} | {'Mean':>{VALUE_WIDTH}} | {'Std':>{VALUE_WIDTH}} | {'Total':>{TOTAL_WIDTH}}")
            print(f"{'-' * MODEL_WIDTH}-+-{'-' * METRIC_WIDTH}-+-{'-' * VALUE_WIDTH}-+-{'-' * VALUE_WIDTH}-+-{'-' * TOTAL_WIDTH}")
            
            model_order = ["ColaCare", "MDAgents", "MedAgent", "ReConcile", "SingleLLM_zero_shot", "SingleLLM_few_shot", "SingleLLM_self_consistency", "SingleLLM_cot", "SingleLLM_cot_sc", "linkbert", "gatortron", "m3ae", "biomedgpt", "mumc"]
            
            if bootstrap:
                for i in range(args.n_bootstrap):
                    qids.append(bootstrap(data_path)) # qid shape: (n_bootstrap, n)
                
                # Loop through the model results
                for model_dir in model_order:
                    if model_dir in os.listdir(os.path.join(args.logs_dir, dataset, qtype)):
                        if not os.path.isdir(os.path.join(args.logs_dir, dataset, qtype, model_dir)):
                            continue
                        
                        model = model_dir
                        result = {"model": model, "acc": [], "score": [], "total": 0}
                        
                        # Loop through each bootstrap sample
                        for i in range(len(qids)):
                            if qtype == "multiple_choice":
                                correct = 0
                                
                            elif qtype == "free-form":
                                score = 0
                                
                            total = len(qids[0])
                            for qid in qids[i]:
                                for ans_file in os.listdir(os.path.join(args.logs_dir, dataset, qtype, model)):
                                    if extract_digits(qid) == extract_digits(ans_file):
                                        try:
                                            ans_data = json.load(open(os.path.join(args.logs_dir, dataset, qtype, model, ans_file), "r"))
                                        except Exception as e:
                                            print(f"Error loading {os.path.join(args.logs_dir, dataset, qtype, model, ans_file)}: {e}")
                                            continue

                                        if qtype == "multiple_choice" and ans_data["ground_truth"] == ans_data["predicted_answer"]:
                                            correct += 1
                                            
                                        # Use LLM-as-a-judge for free-form questions
                                        elif qtype == "free-form":
                                            # Check if the llm_score is already computed
                                            if "llm_score" in ans_data:
                                                score += int(ans_data["llm_score"])
                                            # If not, compute it and save it
                                            else:
                                                try:
                                                    ans_score = llm_score(ans_data["question"], ans_data["ground_truth"], ans_data["predicted_answer"], dataset, args.judge_model).strip()
                                                    if len(ans_score) > 10:
                                                        ans_score = extract_score_from_llm_output(ans_score)
                                                    add_llm_score_to_json(os.path.join(args.logs_dir, dataset, qtype, model, ans_file), ans_score) # Save the score to the JSON file
                                                    score += int(ans_score)
                                                    
                                                except Exception as e:
                                                    print(f"Error adding llm score to {os.path.join(args.logs_dir, dataset, qtype, model, ans_file)}: {e}")
                                                    continue
                                        
                            if qtype == "multiple_choice":
                                result["acc"].append(correct / total)
                                result["total"] += total
                                
                            elif qtype == "free-form":
                                result["score"].append(score / total)
                                result["total"] += total
                        
                        if qtype == "multiple_choice":
                            metric_name = "Accuracy"
                            mean_value = round(np.mean(result["acc"]), 4)
                            std_dev = round(np.std(result["acc"]), 4)
                            total_str = str(result['total'])
                            
                        elif qtype == "free-form":
                            metric_name = "LLM Score"
                            mean_value = round(np.mean(result["score"]), 4)
                            std_dev = round(np.std(result["score"]), 4)
                            total_str = str(result['total'])
                        
                        print(f"{model:<{MODEL_WIDTH}} | {metric_name:<{METRIC_WIDTH}} | {mean_value:>{VALUE_WIDTH}.4f} | {std_dev:>{VALUE_WIDTH}.4f} | {total_str:>{TOTAL_WIDTH}}")
            # else:
            #     with open(data_path, "r") as f:
            #         data = json.load(f)
            #         for datum in data:
            #             qids.append(datum["qid"])        