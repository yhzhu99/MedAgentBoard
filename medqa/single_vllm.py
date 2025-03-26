from openai import OpenAI
import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from utils import LLM_MODELS_SETTINGS, encode_image
# nohup python -m medqa.qa_model --dataset MedQA --prompt_type few_shot > MedQA_few_shot.log 2>&1 &

# Closed question prompt
def zero_shot_prompt(question: str):
    prompt = f"Question: {question} \n" \
        f"Please simply respond with 'yes' or 'no', nothing else. \n"
    return prompt

def few_shot_prompt(dataset: str, question: str):
    if dataset == "Path_VQA":
        examples = "Example 1: Question: Are bile duct cells and canals of Hering stained here with an immunohistochemical stain for cytokeratin 7?" \
                "Answer: yes \n" \
                "Example 2: Question: Does preserved show dissolution of the tissue?" \
                "Answer: no \n" \
            
    elif dataset == "VQA_Rad":
        examples = "Example 1: Question: Are regions of the brain infarcted?" \
                "Answer: yes \n" \
                "Example 2: Question: Are the lungs normal appearing?" \
                "Answer: no \n" \
        
    prompt = f"Question: {question} \n" \
        f"Please simply respond with 'yes' or 'no', nothing else. \n" \
        f"Here are some examples for you reference: '''{examples}''' "
        
    return prompt
    
def cot_prompt(question: str):
    response_field = f"'Thought': [the step-by-step reasoning behind your answer] \n" \
        f"'Answer': [simply 'yes' or 'no', nothing else]" \

    prompt = f"Question: {question} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. " \
        f"Please give your answer in JSON format with the following two fields: '''{response_field}''' "
    return prompt

def inference(datatset, question, image_path, model_key, prompt_type):
    
    response_format = None
    
    if prompt_type == "zero_shot":
        prompt = zero_shot_prompt(question)
    elif prompt_type == "few_shot":
        prompt = few_shot_prompt(dataset, question)
    elif prompt_type == "cot":
        prompt = cot_prompt(question)
        response_format = {"type": "json_object"}
    
    model_settings = LLM_MODELS_SETTINGS[model_key]
    client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
    
    user_message = [
            {"type": "text", "text": prompt}, 
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
    ]
    
    response = client.chat.completions.create(
                    model=model_settings["model_name"],
                    messages=[
                        {"role": "system", "content": "You will be answering a question regarding the medical image."},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=response_format,
                    stream=False
                )
        
    return response.choices[0].message.content
    
if __name__ == "__main__":
    # get the dataset and prompt type from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use for testing")
    parser.add_argument("--prompt_type", type=str, help="Prompt type to use for testing")
    parser.add_argument("--model_name", type=str, help="Model name to use for inference")
    parser.add_argument("--start_pos", type=int, help="Starting position of the file")
    parser.add_argument("--end_pos", type=int, help="Ending position of the file")
    args = parser.parse_args()
    
    dataset = args.dataset # ["Path-VQA", "VQA-Rad"]]
    prompt_type = args.prompt_type # ["zero_shot", "few_shot", "cot"]
    model_name = args.model_name
    start_pos = args.start_pos
    end_pos = args.end_pos
    
    data_path = f"./cleaned_datasets/{dataset.lower()}.json"
    output_path = f"./output/{dataset}/{prompt_type}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output file created at: {output_path}")
    
    # load the data
    with open(data_path, "r") as file:
        data = [json.loads(line) for line in file]
    
    # get the data based on the start and end position
    data = data[start_pos:end_pos] if end_pos != -1 else data[start_pos:]
    total_lines = len(data)
        
    with open(output_path, "a") as output_file:
        
        for datum in tqdm(data, total=total_lines):
            # Closed question
            if datum["answer"].lower() in ["yes", "no"]:
                model_answer = inference(dataset, datum["question"], datum["image_path"], model_name, prompt_type)
                
                output = {
                    "qid": datum["qid"],
                    "question": datum["question"],
                    "ground_truth": datum["answer"],
                    "model_answer": model_answer
                }
                
                json.dump(output, output_file)
                output_file.write("\n")
            else:
                continue