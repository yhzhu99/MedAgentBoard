"""
medagentboard/medqa/single_llm.py

Unified script for handling both text-only and vision-language model inference
for medical question answering tasks. Supports multiple prompting techniques:
- Zero-shot prompting
- Few-shot prompting with examples
- Chain-of-thought (CoT) prompting
- Self-consistency (majority voting)
- CoT with self-consistency

Works with both multiple-choice and free-form questions, and can process
image-based questions when an image path is provided.
"""

from openai import OpenAI
import os
import json
import argparse
from collections import Counter
import time
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


class SingleModelInference:
    """
    Unified class for running inference with a single LLM or VLLM model
    using various prompting techniques.
    """

    def __init__(self, model_key: str = "qwen-max-latest", sample_size: int = 5):
        """
        Initialize the inference handler.

        Args:
            model_key: Key identifying the model in LLM_MODELS_SETTINGS
            sample_size: Number of samples for self-consistency methods
        """
        self.model_key = model_key
        self.sample_size = sample_size

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up OpenAI client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        print(f"Initialized SingleModelInference with model: {model_key}, sample_size: {sample_size}")

    def _call_llm(self,
                 system_message: str,
                 user_message: Union[str, List],
                 response_format: Optional[Dict] = None,
                 n_samples: int = 1,
                 max_retries: int = 3) -> List[str]:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message (text or multimodal content)
            response_format: Optional format specification for response
            n_samples: Number of samples to generate
            max_retries: Maximum number of retry attempts

        Returns:
            List of LLM response texts
        """
        retries = 0
        all_responses = []

        # For each sample we need
        remaining_samples = n_samples

        while remaining_samples > 0 and retries < max_retries:
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]

                # Some models might not properly support n > 1, so we make multiple calls if needed
                current_n = min(remaining_samples, 1)  # Request just 1 at a time to be safe

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    n=current_n,
                    stream=False
                )

                responses = [choice.message.content for choice in completion.choices]
                all_responses.extend(responses)
                remaining_samples -= len(responses)

                # Reset retry counter on successful API call
                retries = 0

            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    if all_responses:  # If we have some responses, use those rather than failing
                        print(f"Warning: Only obtained {len(all_responses)}/{n_samples} samples after max retries")
                        break
                    else:
                        raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief pause before retrying

        return all_responses

    def _prepare_user_message(self,
                            prompt: str,
                            image_path: Optional[str] = None) -> Union[str, List]:
        """
        Prepare user message with optional image content.

        Args:
            prompt: Text prompt
            image_path: Optional path to image

        Returns:
            User message as string or list for multimodal content
        """
        if image_path:
            try:
                base64_image = encode_image(image_path)
                return [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                # Fall back to text-only if image encoding fails
                return prompt
        else:
            return prompt

    def zero_shot_prompt(self,
                        question: str,
                        options: Optional[Dict[str, str]] = None) -> str:
        """
        Create a zero-shot prompt for either multiple-choice or free-form questions.

        Args:
            question: Question text
            options: Optional multiple choice options

        Returns:
            Formatted prompt string
        """
        if options:
            # Multiple choice
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Please respond with the letter of the correct option (A, B, C, etc.) only."
            )
        else:
            # Free form
            prompt = (
                f"Question: {question}\n\n"
                f"Please provide a concise and accurate answer."
            )

        return prompt

    def few_shot_prompt(self,
                       question: str,
                       options: Optional[Dict[str, str]] = None,
                       dataset: str = "MedQA") -> str:
        """
        Create a few-shot prompt with examples relevant to the dataset.

        Args:
            question: Question text
            options: Optional multiple choice options
            dataset: Dataset name to select appropriate examples

        Returns:
            Formatted prompt string with examples
        """
        # Define example pairs for different datasets and question types
        examples = {
            "MedQA_mc": (
                "Example 1: Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
                "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
                "She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), "
                "blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. "
                "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
                "Which of the following is the best treatment for this patient?\n"
                "Options:\n"
                "A: Ampicillin\n"
                "B: Ceftriaxone\n"
                "C: Ciprofloxacin\n"
                "D: Doxycycline\n"
                "E: Nitrofurantoin\n"
                "Answer: E\n\n"

                "Example 2: Question: A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died "
                "only after she awoke in the morning. No cause of death was determined based on the autopsy. "
                "Which of the following precautions could have prevented the death of the baby?\n"
                "Options:\n"
                "A: Placing the infant in a supine position on a firm mattress while sleeping\n"
                "B: Routine postnatal electrocardiogram (ECG)\n"
                "C: Keeping the infant covered and maintaining a high room temperature\n"
                "D: Application of a device to maintain the sleeping position\n"
                "E: Avoiding pacifier use during sleep\n"
                "Answer: A"
            ),

            "PubMedQA_mc": (
                "Example 1: Question: Are sugars-free medicines more erosive than sugars-containing medicines?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "C: Maybe\n"
                "Answer: B\n\n"

                "Example 2: Question: Can autologous platelet-rich plasma gel enhance healing after surgical extraction of mandibular third molars?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "C: Maybe\n"
                "Answer: A"
            ),

            "PubMedQA_ff": (
                "Example 1: Question: Does melatonin supplementation improve sleep quality in adults with primary insomnia?\n"
                "Answer: Yes, melatonin supplementation has been shown to improve sleep quality parameters in adults with primary insomnia, "
                "including reduced sleep onset latency, increased total sleep time, and improved overall sleep quality without significant adverse effects.\n\n"

                "Example 2: Question: Is chronic stress associated with increased risk of cardiovascular disease?\n"
                "Answer: Yes, chronic stress is associated with increased risk of cardiovascular disease through multiple mechanisms, "
                "including elevated blood pressure, increased inflammation, endothelial dysfunction, and unhealthy behavioral coping mechanisms."
            ),

            "PathVQA_mc": (
                "Example 1: Question: Are bile duct cells stained with this immunohistochemical marker?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "Answer: A\n\n"

                "Example 2: Question: Is this a well-differentiated tumor?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "Answer: B"
            ),

            "VQA-RAD_mc": (
                "Example 1: Question: Is there evidence of pneumonia in this chest X-ray?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "Answer: A\n\n"

                "Example 2: Question: Is the heart size enlarged in this radiograph?\n"
                "Options:\n"
                "A: Yes\n"
                "B: No\n"
                "Answer: B"
            ),

            "VQA-RAD_ff": (
                "Example 1: Question: What abnormality is visible in this CT scan of the abdomen?\n"
                "Answer: The CT scan shows a hypodense mass in the liver, approximately 3cm in diameter, with irregular borders, "
                "suggestive of a hepatocellular carcinoma. There is also mild splenomegaly but no ascites.\n\n"

                "Example 2: Question: What is the primary finding in this chest X-ray?\n"
                "Answer: The chest X-ray demonstrates right upper lobe consolidation with air bronchograms, "
                "consistent with lobar pneumonia. No pleural effusion or pneumothorax is identified."
            )
        }

        # Determine which example set to use
        example_key = f"{dataset}_mc" if options else f"{dataset}_ff"

        # Fallbacks if specific dataset examples are not available
        if example_key not in examples:
            if options:
                example_key = "MedQA_mc"  # Default MC examples
            else:
                example_key = "PubMedQA_ff"  # Default FF examples

        example_text = examples[example_key]

        # Format options if provided
        if options:
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Please respond with the letter of the correct option (A, B, C, etc.) only.\n\n"
                f"Here are some examples for your reference:\n\n{example_text}"
            )
        else:
            prompt = (
                f"Question: {question}\n\n"
                f"Please provide a concise and accurate answer.\n\n"
                f"Here are some examples for your reference:\n\n{example_text}"
            )

        return prompt

    def cot_prompt(self,
                  question: str,
                  options: Optional[Dict[str, str]] = None) -> str:
        """
        Create a chain-of-thought prompt that encourages step-by-step reasoning.

        Args:
            question: Question text
            options: Optional multiple choice options

        Returns:
            Formatted CoT prompt string
        """
        if options:
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            response_format = (
                "{\n"
                "  \"Thought\": \"step-by-step reasoning process\",\n"
                "  \"Answer\": \"selected option letter (A, B, C, etc.)\"\n"
                "}"
            )

            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Let's work through this step-by-step to find the correct answer.\n\n"
                f"Please provide your response in JSON format as follows:\n{response_format}"
            )
        else:
            response_format = (
                "{\n"
                "  \"Thought\": \"step-by-step reasoning process\",\n"
                "  \"Answer\": \"final answer to the question\"\n"
                "}"
            )

            prompt = (
                f"Question: {question}\n\n"
                f"Let's work through this step-by-step to find the correct answer.\n\n"
                f"Please provide your response in JSON format as follows:\n{response_format}"
            )

        return prompt

    def process_item(self,
                    item: Dict[str, Any],
                    prompt_type: str,
                    dataset: str) -> Dict[str, Any]:
        """
        Process a single item using the specified prompting technique.

        Args:
            item: Input data dictionary with question, options, etc.
            prompt_type: Type of prompting to use
            dataset: Dataset name

        Returns:
            Result dictionary with predicted answer and metadata
        """
        start_time = time.time()

        # Extract item fields
        qid = item.get("qid", "unknown")
        question = item.get("question", "")
        options = item.get("options")
        image_path = item.get("image_path")
        ground_truth = item.get("answer", "")

        print(f"Processing {qid} with {prompt_type} prompting")

        # Determine if it's a multiple-choice or free-form question
        is_mc = options is not None

        # Set system message based on task type
        if image_path:
            system_message = "You are a medical vision expert analyzing medical images and answering questions about them."
        else:
            system_message = "You are a medical expert answering medical questions with precise and accurate information."

        # Generate prompt based on technique
        if prompt_type == "zero_shot":
            prompt = self.zero_shot_prompt(question, options)
            response_format = None
            n_samples = 1

        elif prompt_type == "few_shot":
            prompt = self.few_shot_prompt(question, options, dataset)
            response_format = None
            n_samples = 1

        elif prompt_type == "cot":
            prompt = self.cot_prompt(question, options)
            response_format = {"type": "json_object"}
            n_samples = 1

        elif prompt_type == "self_consistency":
            # For self-consistency, use zero-shot but with multiple samples
            prompt = self.zero_shot_prompt(question, options)
            response_format = None
            n_samples = self.sample_size  # Use the configured sample_size

        elif prompt_type == "cot_sc":
            # For CoT with self-consistency, use CoT with multiple samples
            prompt = self.cot_prompt(question, options)
            response_format = {"type": "json_object"}
            n_samples = self.sample_size  # Use the configured sample_size

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Prepare user message (text-only or multimodal)
        user_message = self._prepare_user_message(prompt, image_path)

        # Call LLM to get responses
        responses = self._call_llm(
            system_message=system_message,
            user_message=user_message,
            response_format=response_format,
            n_samples=n_samples
        )

        voting_details = None
        # Process responses based on prompt type
        if prompt_type in ["cot", "cot_sc"]:
            # Extract answers from JSON responses
            parsed_responses = []

            for response in responses:
                try:
                    parsed = json.loads(preprocess_response_string(response))
                    thought = parsed.get("Thought", "") or parsed.get("thought", "")
                    answer = parsed.get("Answer", "") or parsed.get("answer", "")
                    parsed_responses.append({"thought": thought, "answer": answer, "full_response": response})
                except json.JSONDecodeError:
                    # Fallback parsing for malformed JSON
                    lines = response.strip().split('\n')
                    thought = ""
                    answer = ""

                    for line in lines:
                        if "thought" in line.lower() and ":" in line:
                            thought = line.split(":", 1)[1].strip()
                        elif "answer" in line.lower() and ":" in line:
                            answer = line.split(":", 1)[1].strip()

                    parsed_responses.append({"thought": thought, "answer": answer, "full_response": response})

            if prompt_type == "cot":
                # For CoT, use the first response
                predicted_answer = parsed_responses[0]["answer"]
                reasoning = parsed_responses[0]["thought"]
                individual_responses = [parsed_responses[0]]
            else:
                # For CoT-SC, use majority voting on answers
                answers = [r["answer"] for r in parsed_responses]
                answer_counts = Counter(answers)
                predicted_answer = answer_counts.most_common(1)[0][0]

                # Detailed voting breakdown
                voting_details = {
                    "vote_counts": dict(answer_counts),
                    "winning_answer": predicted_answer,
                    "total_votes": sum(answer_counts.values())
                }

                # Collect all reasoning paths
                reasoning = "\n\n".join([f"Path {i+1}: {r['thought']}\nAnswer: {r['answer']}" for i, r in enumerate(parsed_responses)])
                individual_responses = parsed_responses

        elif prompt_type == "self_consistency":
            # For self-consistency, use majority voting
            answer_counts = Counter(responses)
            predicted_answer = answer_counts.most_common(1)[0][0]

            # Detailed voting breakdown
            voting_details = {
                "vote_counts": dict(answer_counts),
                "winning_answer": predicted_answer,
                "total_votes": sum(answer_counts.values())
            }

            reasoning = f"Majority vote from {len(responses)} samples: {dict(answer_counts)}"
            individual_responses = [{"answer": r, "full_response": r} for r in responses]

        else:
            # For zero-shot and few-shot, use the first response
            predicted_answer = responses[0].strip()
            reasoning = "Direct answer, no explicit reasoning"
            individual_responses = [{"answer": predicted_answer, "full_response": responses[0]}]

        # Clean up the predicted answer (extract just the option letter for MC)
        if is_mc and len(predicted_answer) > 1:
            # Look for option letters in the answer
            for option in options.keys():
                if option in predicted_answer or option.lower() in predicted_answer.lower():
                    predicted_answer = option
                    break

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare the result structure with improved details
        result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "case_history": {
                "reasoning": reasoning,
                "prompt_type": prompt_type,
                "model": self.model_key,
                "raw_responses": responses,
                "individual_responses": individual_responses,
                "voting_details": voting_details,
                "processing_time": processing_time
            }
        }

        return result


def main():
    """
    Main function to process medical QA datasets with various prompting techniques.
    """
    parser = argparse.ArgumentParser(description="Run single model inference on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (MedQA, PubMedQA, PathVQA, VQA-RAD)")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True,
                       help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--prompt_type", type=str, required=True,
                       choices=["zero_shot", "few_shot", "cot", "self_consistency", "cot_sc"],
                       help="Prompting technique to use")
    parser.add_argument("--model_key", type=str, default="qwen-max-latest",
                       help="Model key from LLM_MODELS_SETTINGS")
    parser.add_argument("--sample_size", type=int, default=5,
                        help="Number of samples for self-consistency methods")
    args = parser.parse_args()

    # Dataset and QA type
    dataset_name = args.dataset
    qa_type = args.qa_type
    prompt_type = args.prompt_type
    model_key = args.model_key
    sample_size = args.sample_size

    print(f"Dataset: {dataset_name}")
    print(f"QA Type: {qa_type}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Model: {model_key}")
    print(f"Sample Size: {sample_size}")

    # Method name for logging
    method = f"SingleLLM_{prompt_type}"

    # Set up data path
    data_path = f"./my_datasets/processed/{dataset_name}/medqa_{qa_type}.json"

    # Set up logs directory
    qa_format_dir = "multiple_choice" if qa_type == "mc" else "free-form"
    logs_dir = os.path.join("logs", dataset_name, qa_format_dir, method)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Data path: {data_path}")
    print(f"Logs directory: {logs_dir}")

    # Initialize the model
    model = SingleModelInference(model_key=model_key, sample_size=sample_size)

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} items from {data_path}")

    # Track stats
    processed_count = 0
    skipped_count = 0
    error_count = 0
    correct_count = 0

    # Process each item
    for item in tqdm(data, desc=f"Processing {dataset_name} with {prompt_type}"):
        qid = item["qid"]

        # Skip if already processed
        result_path = os.path.join(logs_dir, f"{qid}-result.json")
        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            skipped_count += 1
            continue

        try:
            # Process the item
            result = model.process_item(
                item=item,
                prompt_type=prompt_type,
                dataset=dataset_name
            )

            # Save the result
            save_json(result, result_path)

            # Update stats
            processed_count += 1
            if result["predicted_answer"] == result["ground_truth"]:
                correct_count += 1

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            error_count += 1

    # Print summary
    print("\n" + "="*50)
    print(f"Processing Summary for {dataset_name} ({qa_type}) with {prompt_type}:")
    print(f"Total items: {len(data)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Errors: {error_count}")

    if processed_count > 0:
        accuracy = (correct_count / processed_count) * 100
        print(f"Accuracy of processed items: {accuracy:.2f}%")

    print("="*50)


if __name__ == "__main__":
    main()