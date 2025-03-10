import logging
import os
import json
from tqdm import tqdm
from medqa.graph import MedQAGraph

# Set up logging to track the flow of the multi-agent collaboration
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Run the inference: python -m medqa.inference
def inference(question: str, max_rounds: int = 5):
    log = {}
    qa_graph = MedQAGraph(question, max_rounds=max_rounds)
    final_answer, log = qa_graph.run()
    
    return final_answer, log

def format_question(question: str, options: list):
    return f"{question}\n\n + {options}"

if __name__ == "__main__":
    
    medqa_path = "/Users/maxhe/Downloads/Research/dataset/MedQA/medqa_test"
    
    for file_path in os.listdir(medqa_path):
        file_path = os.path.join(medqa_path, file_path)
        log_file = os.path.basename(file_path).replace('.jsonl', '_log.jsonl')
        log_path = os.path.join(medqa_path, log_file)
        print(f"Log file saved at: {log_path}")
        correct = 0
        count = 0
        
        with open(file_path, "r") as file:
            total_lines = sum(1 for _ in file)
        
        with open(file_path, "r") as question_file, open(log_path, "a") as log_file:
            for line in tqdm(question_file, total=total_lines):
                datum = json.loads(line)
                question = datum["question"]
                options = datum["options"]
                ground_truth = datum["answer_idx"]
                model_answer, log = inference(format_question(question, options))
                if model_answer.lower() == ground_truth.lower():
                    correct += 1
                
                json.dump(log, log_file, indent=4)
                log_file.write("\n")
                
                count += 1
                if count == 40:
                    break
        
        print(f"Accuracy: {correct}/{count}")
        # print(f"Accuracy for {file_path}: {correct/len(file)}")
                    