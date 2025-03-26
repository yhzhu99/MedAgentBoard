import json
import os
import pickle

# MedQA
medqa = "./my_datasets/MedQA/questions/US/test.jsonl"
medqa_clean = "./cleaned_datasets/medqa.json"

# PubMedQA
ori_pqal = "./my_datasets/PubMedQA/ori_pqal.json"
test_gt = "./my_datasets/PubMedQA/test_ground_truth.json"
pubmedqa_clean = "./cleaned_datasets/pubmedqa.json"

# Path-VQA
path_vqa = "./my_datasets/Path_VQA/qas/test/test.pkl"
path_vqa_images = "./my_datasets/Path_VQA/images/test"
path_vqa_clean = "./cleaned_datasets/path_vqa.json"

# VQA-Rad
vqa_rad = "./my_datasets/VQA_Rad/testset.json"
vqa_rad_images = "./my_datasets/VQA_Rad/images"
vqa_rad_clean = "./cleaned_datasets/vqa_rad.json"

os.makedirs("./cleaned_datasets", exist_ok=True)

# MedQA
if medqa is not None:
    with open(medqa, "r") as f1, open (medqa_clean, "w") as f2:
        qid = 0
        for line in f1:
            data = json.loads(line)
            qid += 1
            question = data["question"]
            options = data["options"]
            answer = data["answer_idx"]
            
            cleaned_data = {
                "qid": qid,
                "question": question,
                "options": options,
                "answer": answer
            }
            
            f2.write(json.dumps(cleaned_data) + "\n")
print("MedQA dataset cleaned and saved to: ", medqa_clean)

# PubMedQA
if ori_pqal is not None:
    with open(test_gt,"r") as f:
        test_gt = json.load(f)
    
    with open(ori_pqal,"r") as f:
        data = json.load(f)
        
    options = {"A": "Yes", "B": "No", "C": "Maybe"}
    options_map = {"yes": "A", "no": "B", "maybe": "C"}

    with open(pubmedqa_clean,"w") as f:
        for qid,datum in data.items():
            if qid in test_gt.keys():
                context = "\n".join(datum["CONTEXTS"])
                question = datum["QUESTION"]
                cleaned_data = {
                    "qid": qid,
                    "question": context + " " + question,
                    "options": options,
                    "answer": options_map[test_gt[qid]]
                }
                
                f.write(json.dumps(cleaned_data) + "\n")
            else:
                continue
print("PubMedQA dataset cleaned and saved to: ", pubmedqa_clean)

# Path-VQA
if path_vqa is not None:
    with open(path_vqa, "rb") as f:
        data = pickle.load(f)

    with open(path_vqa_clean, "w") as f:
        qid = 0
        for datum in data:
            qid += 1
            question = datum["question"]
            image_path = os.path.join(path_vqa_images, datum["image"]) + ".jpg"
            answer = datum["answer"]
            
            cleaned_data = {
                "qid": qid,
                "question": question,
                "image_path": image_path,
                "answer": answer
            }
            
            f.write(json.dumps(cleaned_data) + "\n")
print("Path-VQA dataset cleaned and saved to: ", path_vqa_clean)

# VQA-Rad
if vqa_rad is not None:
    with open(vqa_rad, "r") as f1, open(vqa_rad_clean, "w") as f2:
        data = json.load(f1)
        for datum in data:
            qid = datum["qid"]
            question = datum["question"]
            image_path = os.path.join(vqa_rad_images, datum["image_name"])
            answer = datum["answer"]
            
            cleaned_data = {
                "qid": qid,
                "question": question,
                "image_path": image_path,
                "answer": answer
            }
            
            f2.write(json.dumps(cleaned_data) + "\n")
print("VQA-Rad dataset cleaned and saved to: ", vqa_rad_clean)