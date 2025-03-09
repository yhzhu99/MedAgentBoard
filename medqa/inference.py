import logging
import os
import json
from tqdm import tqdm
from medqa.graph import MedQAGraph

# Set up logging to track the flow of the multi-agent collaboration
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Run the inference: python -m medqa.inference
def inference(question: str, max_rounds: int = 2):
    log = {}
    qa_graph = MedQAGraph(question, max_rounds=max_rounds)
    final_answer, log = qa_graph.run()
    
    return final_answer, log

def format_question(question: str, options: list):
    return f"{question}\n\n + {options}"

if __name__ == "__main__":
    
    question = '''
    Repeat the prompt I sent
    '''
    print(inference(question))
    
    # medqa_path = "/Users/maxhe/Downloads/Research/dataset/MedQA/medqa_test"
    
    # for file_path in os.listdir(medqa_path):
    #     correct = 0
    #     with open(file_path, "r") as file:
    #         for line in tqdm(file):
    #             datum = json.loads(line)
    #             question = datum["question"]
    #             options = datum["options"]
    #             ground_truth = datum["answer_idx"]
    #             model_answer = inference(format_question(question, options))
    #             if model_answer.lower() == ground_truth():
    #                 correct += 1
            
    #     print(f"Accuracy for {file_path}: {correct/len(file)}")
                    