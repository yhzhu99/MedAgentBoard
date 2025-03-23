from openai import OpenAI
import os
import base64
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Configure LLM model settings
LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 官方站点",
        "reasoning": False,
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
        "comment": "DeepSeek R1 推理模型 官方站点",
        "reasoning": True,
    },
    "deepseek-v3-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-v3",
        "comment": "DeepSeek V3 阿里站点",
        "reasoning": False,
    },
    "deepseek-r1-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-r1",
        "comment": "DeepSeek R1 推理模型 阿里站点",
        "reasoning": True,
    },
    "qwen-max-latest": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max-latest",
        "comment": "通义千问 Max",
        "reasoning": False,
    },
    "qwen-vl-max": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-vl-max",
        "comment": "qwen-vl-max",
        "reasoning": False,
    }
}


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file does not exist
        IOError: If there is an error reading the image file
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    except IOError as e:
        raise IOError(f"Error reading image file: {e}")


def get_structured_vqa_response(
    image_path: str, 
    prompt: str, 
    model_key: str = "qwen-vl-max"
) -> Dict[str, str]:
    """
    Get a structured VQA response with explanation and answer fields.
    
    Args:
        image_path: Path to the image file
        prompt: The text prompt for the VQA task
        model_key: Key of the model in LLM_MODELS_SETTINGS
        
    Returns:
        Dictionary with 'explanation' and 'answer' fields
        
    Raises:
        KeyError: If the specified model key does not exist
        Exception: For any API call errors
    """
    if model_key not in LLM_MODELS_SETTINGS:
        raise KeyError(f"Model '{model_key}' not found in settings.")
    
    model_settings = LLM_MODELS_SETTINGS[model_key]
    
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except (FileNotFoundError, IOError) as e:
        raise e
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=model_settings["api_key"],
        base_url=model_settings["base_url"],
    )
    
    # Prepare system message to enforce structured output
    system_message = {
        "role": "system",
        "content": "You are a helpful visual question answering assistant. Always provide your response in JSON format with 'explanation' and 'answer' fields."
    }
    
    # Prepare user message with image and prompt
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
            {
                "type": "text", 
                "text": f"{prompt} Please output a JSON dictionary with two fields: 'explanation' and 'answer'."
            },
        ],
    }
    
    try:
        # Make API call
        completion = client.chat.completions.create(
            model=model_settings["model_name"],
            messages=[system_message, user_message],
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_content = completion.choices[0].message.content
        parsed_response = parse_structured_output(response_content)
        return parsed_response
        
    except Exception as e:
        raise Exception(f"Error calling LLM API: {e}")


def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse the LLM response to extract structured output.
    
    Args:
        response_text: The text response from the LLM
        
    Returns:
        Dictionary with 'explanation' and 'answer' fields
    """
    try:
        # Try to parse as JSON
        parsed = json.loads(response_text)
        
        # Ensure required fields exist
        if "explanation" not in parsed or "answer" not in parsed:
            # If missing fields, attempt to extract from text
            return {
                "explanation": parsed.get("explanation", "No explanation provided"),
                "answer": parsed.get("answer", "No answer provided")
            }
        return parsed
    except json.JSONDecodeError:
        # If not valid JSON, extract from text
        # This is a fallback in case the model didn't format as JSON
        lines = response_text.strip().split('\n')
        explanation = ""
        answer = ""
        
        for line in lines:
            if line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
        
        return {
            "explanation": explanation or "Could not parse explanation",
            "answer": answer or "Could not parse answer"
        }


def main():
    """Main function to demonstrate VQA with structured output."""
    image_path = "vqa/my_datasets/test_img.jpg"
    prompt = "How is the patient oriented in the image? Please give brief and exact answer."
    model_key = "qwen-vl-max"
    
    try:
        result = get_structured_vqa_response(image_path, prompt, model_key)
        print(f"Answer: {result['answer']}")
        print(f"Explanation: {result['explanation']}")
        print("\nFull response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
