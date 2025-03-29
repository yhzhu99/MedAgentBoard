import os
import random
import argparse
from typing import List, Dict, Any
import pandas as pd
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
        curated_data = {
            "qid": f"medqa_mc_{str(i + 1).zfill(3)}",
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer_idx"]
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
    output_path_base = os.path.join(output_dir, "PubMedQA")
    output_path_mc = os.path.join(output_path_base, "pubmedqa_mc.json")
    output_path_ff = os.path.join(output_path_base, "pubmedqa_ff.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)

    # Load datasets
    data = load_json(ori_pqal_path)

    # Define standard options
    options = {"A": "Yes", "B": "No", "C": "Maybe"}
    options_map = {"Yes": "A", "No": "B", "Maybe": "C"}

    processed_data_mc = []
    processed_data_ff = []

    for qid, item_data in data.items():
        context = " ".join(item_data["CONTEXTS"])  # Concatenate contexts into a single string
        question = item_data["QUESTION"]
        answer = item_data["final_decision"].capitalize()

        # Free-form version: Concatenate context into the question
        free_form_question = (
            f"{question}\n\n"
            f"Context: {context}"
        )

        # Multiple-choice version: Concatenate context into the question
        mc_question = (
            f"{question}\n\n"
            f"Context: {context}"
        )


        # Add both versions to the processed data
        free_form_data = {
            "qid": f"pubmedqa_ff_{qid}",
            "question": free_form_question,
            "answer": f"{answer}. {item_data["LONG_ANSWER"]}"  # Use the long answer for free-form
        }

        mc_data = {
            "qid": f"pubmedqa_mc_{qid}",
            "question": mc_question,
            "options": options,
            "answer": options_map[answer]  # Map ground truth to option key
        }

        processed_data_ff.append(free_form_data)
        processed_data_mc.append(mc_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data_ff = random_select_samples(processed_data_ff, sample_size)
        processed_data_mc = random_select_samples(processed_data_mc, sample_size)

    # Save processed data
    save_json(processed_data_ff, output_path_ff)
    print(f"PubMedQA dataset (free-form) processed and saved to: {output_path_ff}")
    save_json(processed_data_mc, output_path_mc)
    print(f"PubMedQA dataset (free-form) processed and saved to: {output_path_mc}")


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
    path_vqa_path = os.path.join(raw_dir, "PathVQA", "qas", "test", "test.pkl")
    path_vqa_images = os.path.join(raw_dir, "PathVQA", "images", "test")
    output_path = os.path.join(output_dir, "PathVQA", "pathvqa_mc.json")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the dataset
    data = pd.read_pickle(path_vqa_path)

    processed_data = []

    # Define standard options for yes/no questions
    options = {"A": "Yes", "B": "No"}
    options_map = {"yes": "A", "no": "B"}

    for i, item in enumerate(data):
        answer = str(item["answer"]).lower().strip()

        # Only process questions with yes/no answers
        if answer not in ["yes", "no"]:
            continue

        question = item["question"]
        image_path = os.path.join(path_vqa_images, item["image"]) + ".jpg"

        curated_data = {
            "qid": f"pathvqa_mc_{str(i + 1).zfill(6)}",
            "question": question,
            "image_path": image_path,
            "options": options,
            "answer": options_map[answer]  # Map answer to option key
        }

        processed_data.append(curated_data)

    # Apply sampling if requested
    if sample_size is not None:
        processed_data = random_select_samples(processed_data, sample_size)

    # Save processed data
    save_json(processed_data, output_path)
    print(f"PathVQA dataset (yes/no questions only) processed and saved to: {output_path}")
    print(f"Total yes/no questions found: {len(processed_data)}")


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
    parser.add_argument("--sample-size", type=int, default=200, help="Number of samples to randomly select (None for all samples)")

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