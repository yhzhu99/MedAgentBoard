import os
import random
import argparse
from typing import List, Dict, Any

from medagentboard.utils.json_utils import save_json, load_json, load_jsonl

# Define paths
RAW_DATA_DIR = "./my_datasets/raw"
PROCESSED_DATA_DIR = "./my_datasets/processed"

def random_select_samples(data: List[Dict[str, Any]], sample_size: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly select a subset of samples from the dataset.

    Args:
        data: The complete dataset to sample from
        sample_size: Number of samples to select (default: 200)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        A randomly selected subset of the input data
    """
    if sample_size >= len(data):
        return data

    random.seed(seed)
    return random.sample(data, sample_size)


def process_medqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, sample_size: int = None):
    """
    Process the MedQA dataset from raw format to standardized format.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        sample_size: Number of samples to select (None for all samples)
    """
    # Define paths
    medqa_path = os.path.join(raw_dir, "MedQA", "questions", "US", "test.jsonl")
    output_path = os.path.join(output_dir, "MedQA", "medqa.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load JSONL data
    medqa_data = load_jsonl(medqa_path)

    processed_data = []

    for i, item in enumerate(medqa_data):
        # Convert options to the standardized format
        options_dict = {}
        for j, option in enumerate(item["options"]):
            options_dict[chr(65 + j)] = option  # A, B, C, D, etc.

        answer = item["answer_idx"]

        curated_data = {
            "qid": f"medqa_mc_{str(i + 1).zfill(3)}",
            "question": item["question"],
            "options": options_dict,
            "answer": answer
        }

        processed_data.append(curated_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data = random_select_samples(processed_data, sample_size)

    # Save processed data
    save_json(processed_data, output_path)
    print(f"MedQA dataset processed and saved to: {output_path}")

def process_pubmedqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, sample_size: int = None):
    """
    Process the PubMedQA dataset from raw format to standardized format.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        sample_size: Number of samples to select (None for all samples)
    """
    # Define paths
    ori_pqal_path = os.path.join(raw_dir, "PubMedQA", "ori_pqal.json")
    test_gt_path = os.path.join(raw_dir, "PubMedQA", "test_ground_truth.json")
    output_path = os.path.join(output_dir, "PubMedQA", "medqa.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load datasets
    test_gt = load_json(test_gt_path)
    data = load_json(ori_pqal_path)

    # Define standard options
    options = {"A": "Yes", "B": "No", "C": "Maybe"}
    options_map = {"yes": "A", "no": "B", "maybe": "C"}

    processed_data = []

    for qid, item_data in data.items():
        if qid in test_gt.keys():
            context = "\n".join(item_data["CONTEXTS"])
            question = item_data["QUESTION"]

            curated_data = {
                "qid": f"pubmedqa_{qid}",
                "question": context + " " + question,
                "options": options,
                "answer": options_map[test_gt[qid]]
            }

            processed_data.append(curated_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data = random_select_samples(processed_data, sample_size)

    # Save processed data
    save_json(processed_data, output_path)
    print(f"PubMedQA dataset processed and saved to: {output_path}")

def process_pathvqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, sample_size: int = None):
    """
    Process the PathVQA dataset from raw format to standardized format.
    This function expects the dataset to be in JSON format (converted from .pkl),
    to avoid using the pickle module.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        sample_size: Number of samples to select (None for all samples)
    """
    # Define paths - note we're using JSON instead of pkl
    path_vqa_path = os.path.join(raw_dir, "PathVQA", "qas", "test", "test.json")
    path_vqa_images = os.path.join(raw_dir, "PathVQA", "images", "test")
    output_path = os.path.join(output_dir, "PathVQA", "medqa.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(path_vqa_path):
        pkl_path = os.path.join(raw_dir, "PathVQA", "qas", "test", "test.pkl")
        if os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Expected JSON file not found at {path_vqa_path}. "
                f"Please convert the pickle file {pkl_path} to JSON format first."
            )
        else:
            raise FileNotFoundError(f"Neither JSON nor PKL file found for PathVQA dataset.")

    # Load the dataset
    data = load_json(path_vqa_path)

    processed_data = []

    for i, item in enumerate(data):
        question = item["question"]
        image_path = os.path.join(path_vqa_images, item["image"]) + ".jpg"
        answer = item["answer"]

        curated_data = {
            "qid": f"pathvqa_{str(i + 1).zfill(3)}",
            "question": question,
            "image_path": image_path,
            "answer": answer
        }

        processed_data.append(curated_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data = random_select_samples(processed_data, sample_size)

    # Save processed data
    save_json(processed_data, output_path)
    print(f"PathVQA dataset processed and saved to: {output_path}")

def process_vqa_rad(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, sample_size: int = None):
    """
    Process the VQA-RAD dataset from raw format to standardized format.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        sample_size: Number of samples to select (None for all samples)
    """
    # Define paths
    vqa_rad_path = os.path.join(raw_dir, "VQA-RAD", "testset.json")
    vqa_rad_images = os.path.join(raw_dir, "VQA-RAD", "images")
    output_path = os.path.join(output_dir, "VQA-RAD", "medqa.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load dataset
    data = load_json(vqa_rad_path)

    processed_data = []

    for item in data:
        qid = item["qid"]
        question = item["question"]
        image_path = os.path.join(vqa_rad_images, item["image_name"])
        answer = item["answer"]

        curated_data = {
            "qid": f"vqa_rad_{qid}",
            "question": question,
            "image_path": image_path,
            "answer": answer
        }

        processed_data.append(curated_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data = random_select_samples(processed_data, sample_size)

    # Save processed data
    save_json(processed_data, output_path)
    print(f"VQA-RAD dataset processed and saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process medical datasets into a standardized format")
    parser.add_argument("--medqa", action="store_true", help="Process MedQA dataset")
    parser.add_argument("--pubmedqa", action="store_true", help="Process PubMedQA dataset")
    parser.add_argument("--pathvqa", action="store_true", help="Process PathVQA dataset")
    parser.add_argument("--vqa-rad", action="store_true", help="Process VQA-RAD dataset")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR, help="Directory containing raw datasets")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR, help="Directory to save processed datasets")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to randomly select (None for all samples)")

    args = parser.parse_args()

    # If no dataset is specified, show help
    if not (args.medqa or args.pubmedqa or args.pathvqa or args.vqa_rad or args.all):
        parser.print_help()
        return

    # Process requested datasets
    if args.all or args.medqa:
        process_medqa(args.raw_dir, args.output_dir, args.sample_size)

    if args.all or args.pubmedqa:
        process_pubmedqa(args.raw_dir, args.output_dir, args.sample_size)

    if args.all or args.pathvqa:
        process_pathvqa(args.raw_dir, args.output_dir, args.sample_size)

    if args.all or args.vqa_rad:
        process_vqa_rad(args.raw_dir, args.output_dir, args.sample_size)

    print("All requested datasets processed successfully!")

if __name__ == "__main__":
    main()