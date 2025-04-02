import json
import os
import re
from typing import Any, List, Dict, Union, Optional, Iterator

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file

    Args:
        data: Data to be saved (must be JSON serializable)
        filepath: Save path including filename
        indent: JSON indentation format, defaults to 2
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(f"Data saved to: {filepath}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def save_jsonl(data_list: List[Any], filepath: str) -> None:
    """
    Save a list of items to a JSONL file (each item on a separate line)

    Args:
        data_list: List of items to be saved (each must be JSON serializable)
        filepath: Save path including filename
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write JSONL file - one JSON object per line
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Data saved to JSONL file: {filepath}")

def load_jsonl(filepath: str) -> List[Any]:
    """
    Load data from a JSONL file (each line is a separate JSON object)

    Args:
        filepath: Path to JSONL file

    Returns:
        List of loaded items

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data_list.append(json.loads(line))

    return data_list

def iter_jsonl(filepath: str) -> Iterator[Any]:
    """
    Iterate through items in a JSONL file without loading everything into memory

    Args:
        filepath: Path to JSONL file

    Yields:
        Each item from the JSONL file

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                yield json.loads(line)

def merge_json_files(filepaths: List[str], output_filepath: str) -> None:
    """
    Merge multiple JSON files (assuming each contains list data)

    Args:
        filepaths: List of JSON file paths to merge
        output_filepath: Output file path for merged data
    """
    merged_data = []

    for filepath in filepaths:
        data = load_json(filepath)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)

    save_json(merged_data, output_filepath)
    print(f"Merged {len(filepaths)} files into: {output_filepath}")

def update_json(filepath: str, new_data: Any) -> None:
    """
    Update an existing JSON file

    Args:
        filepath: Path to JSON file to update
        new_data: New data (if dict, will merge with existing; if list, will append to existing)
    """
    if os.path.exists(filepath):
        existing_data = load_json(filepath)

        if isinstance(existing_data, dict) and isinstance(new_data, dict):
            # If both are dictionaries, merge them
            existing_data.update(new_data)
        elif isinstance(existing_data, list) and isinstance(new_data, list):
            # If both are lists, extend them
            existing_data.extend(new_data)
        else:
            # Other cases, replace completely
            existing_data = new_data
    else:
        existing_data = new_data

    save_json(existing_data, filepath)
    print(f"Updated file: {filepath}")

def preprocess_response_string(response_text: str) -> str:
    if response_text.startswith('```json') and response_text.endswith('```'):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith('```') and response_text.endswith('```'):
        response_text = response_text[3:-3].strip()
    response_text = response_text.replace("```", "").replace("json", "").strip()
    # Remove trailing commas
    response_text = re.sub(r',\s*}', '}', response_text)
    response_text = re.sub(r',\s*]', ']', response_text)
    return response_text
