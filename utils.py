from dotenv import load_dotenv
import os
import base64


load_dotenv()

LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 Official",
        "reasoning": False,
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
        "comment": "DeepSeek R1 Reasoning Model Official",
        "reasoning": True,
    },
    "deepseek-v3-huoshan": {
        "api_key": os.getenv("HUOSHAN_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "deepseek-v3-250324",
        "comment": "DeepSeek V3 Huoshan",
        "reasoning": False,
    },
    "deepseek-r1-huoshan": {
        "api_key": os.getenv("HUOSHAN_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "deepseek-r1-250120",
        "comment": "DeepSeek R1 Reasoning Model Huoshan",
        "reasoning": True,
    },
    "deepseek-v3-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-v3",
        "comment": "DeepSeek V3 Ali",
        "reasoning": False,
    },
    "deepseek-r1-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-r1",
        "comment": "DeepSeek R1 Reasoning Model Ali",
        "reasoning": True,
    },
    "qwen-max-latest": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max-latest",
        "comment": "Qwen Max",
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
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except IOError as e:
        raise IOError(f"Error reading image file: {e}")