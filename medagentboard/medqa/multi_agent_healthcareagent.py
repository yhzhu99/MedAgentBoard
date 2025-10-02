"""
medagentboard/medqa/multi_agent_healthcareagent.py

This file implements the HealthcareAgent framework as a standalone, end-to-end baseline.
It is inspired by the paper "Healthcare agent: eliciting the power of large language models for medical consultation".
The framework processes a single medical query through a multi-step pipeline involving planning,
preliminary analysis, internal safety review ("discuss"), and final response modification.
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Optional, List
from openai import OpenAI
from tqdm import tqdm

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string

# --- Prompts adapted from the "Healthcare agent" paper's logic ---

# Corresponds to the "Planner" module to decide the initial action
PLANNER_PROMPT_TEMPLATE = """
Based on the provided medical query, determine the best initial course of action.
- If the query is ambiguous, lacks critical details for a safe conclusion, or would benefit from further clarification, choose 'INQUIRY'.
- If you have sufficient information to provide a confident and safe diagnosis or answer, choose 'DIAGNOSE'.

Medical Query:
Question: {question}
{options_text}
{image_text}

Respond with a single word: DIAGNOSE or INQUIRY.
"""

# Corresponds to the "Inquiry" submodule to generate clarifying questions
INQUIRY_PROMPT_TEMPLATE = """
You are a medical doctor analyzing a case. To form an accurate and safe conclusion for the query below, you need more information.
Generate a list of the top 3 most critical follow-up questions you would ask to better understand the situation.

Medical Query:
Question: {question}
{options_text}
{image_text}

Return a JSON object with a single key "questions" containing a list of strings.
Example: {{"questions": ["How long have you experienced this symptom?", "Is there any associated pain?"]}}
"""

# Corresponds to the "Medical Diagnosis" submodule for generating a preliminary response
PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE = """
As a medical doctor, provide a preliminary analysis of the following case based on the available information.
{inquiry_context}

Your output MUST be a JSON object with two keys:
1. "explanation": Your detailed reasoning and diagnostic process.
2. "answer": Your conclusion. For multiple-choice questions, this must be ONLY the option letter (e.g., 'A', 'B').

Medical Query:
Question: {question}
{options_text}
{image_text}
"""

# --- Safety Module Prompts (The "Discuss" Phase) ---

SAFETY_ETHICS_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Does it include necessary disclaimers about being an AI and the potential risks of its advice?
Provide concise feedback for improvement if it's lacking. If it's good, state that.

AI Response to be Reviewed:
{preliminary_response}

Your Feedback:
"""

SAFETY_EMERGENCY_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Does the case involve any potentially serious or life-threatening symptoms?
If so, highlight them and suggest adding a clear warning to seek immediate medical attention. If not, state that.

AI Response to be Reviewed:
{preliminary_response}

Your Feedback:
"""

SAFETY_ERROR_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Are there any potential factual errors, misinterpretations of the image/text, or logical contradictions?
Point out any potential errors and suggest corrections. If none are found, state that.

AI Response to be Reviewed:
{preliminary_response}

Your Feedback:
"""

# --- Final Modification Prompt (The "Modify" Phase) ---

FINAL_MODIFICATION_PROMPT_TEMPLATE = """
You are a senior medical supervisor tasked with creating the final, definitive response.
Revise the preliminary analysis below by incorporating the feedback from the internal safety review.
The final output must be a single, polished JSON object with "explanation" and "answer" keys.

1.  **Original Medical Query:**
    Question: {question}
    {options_text}
    {image_text}

2.  **Preliminary Analysis (Draft):**
    {preliminary_response}

3.  **Internal Safety Review Feedback:**
    - Ethics & Disclaimer Feedback: {ethics_feedback}
    - Emergency Situation Feedback: {emergency_feedback}
    - Factual Error Feedback: {error_feedback}

Your task is to integrate the feedback to create a final, safe, and accurate response.
Ensure the explanation is comprehensive and the answer is correct.
For multiple-choice questions, the 'answer' field must contain ONLY the option letter.

**Final Revised JSON Output:**
"""


class HealthcareAgentFramework:
    """
    A standalone framework that implements the HealthcareAgent methodology.
    """

    def __init__(self, model_key: str):
        """
        Initialize the framework.

        Args:
            model_key: The LLM model key from LLM_MODELS_SETTINGS to be used for all internal steps.
        """
        self.model_key = model_key

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        print(f"Initialized HealthcareAgentFramework with model: {self.model_name}")

    def _call_llm(self,
                  prompt: str,
                  image_path: Optional[str] = None,
                  expect_json: bool = True,
                  max_retries: int = 3) -> str:
        """
        A helper function to call the LLM with a given prompt and optional image.
        """
        system_message = {"role": "system", "content": "You are a highly capable and meticulous medical AI assistant."}
        user_content = [{"type": "text", "text": prompt}]

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            base64_image = encode_image(image_path)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        user_message = {"role": "user", "content": user_content}

        messages = [system_message, user_message]
        response_format = {"type": "json_object"} if expect_json else None

        retries = 0
        while retries < max_retries:
            try:
                print(f"Calling LLM (JSON: {expect_json})...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format
                )
                response = completion.choices[0].message.content
                print(f"LLM call successful. Response snippet: {response[:80]}...")
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts.")
                time.sleep(1)
        return "" # Should not be reached

    def run_query(self, data_item: Dict) -> Dict:
        """
        Processes a single medical query through the full HealthcareAgent pipeline.
        """
        qid = data_item["qid"]
        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")
        ground_truth = data_item.get("answer")

        print(f"\n{'='*20} Processing QID: {qid} with HealthcareAgentFramework {'='*20}")
        start_time = time.time()

        case_history = {
            "steps": []
        }

        # --- Prepare context strings used in multiple prompts ---
        options_text = ""
        if options:
            options_text = "Options:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
        image_text = "An image is provided for context." if image_path else ""

        try:
            # === STEP 1: Planner Module ===
            planner_prompt = PLANNER_PROMPT_TEMPLATE.format(
                question=question, options_text=options_text, image_text=image_text
            )
            action = self._call_llm(planner_prompt, image_path, expect_json=False).strip().upper()
            case_history["steps"].append({"step": "1_Planner", "decision": action})

            # === STEP 2: Inquiry Module (Optional) ===
            inquiry_context = ""
            if "INQUIRY" in action:
                inquiry_prompt = INQUIRY_PROMPT_TEMPLATE.format(
                    question=question, options_text=options_text, image_text=image_text
                )
                inquiry_response_str = self._call_llm(inquiry_prompt, image_path, expect_json=True)
                inquiry_result = json.loads(preprocess_response_string(inquiry_response_str))
                questions = inquiry_result.get("questions", [])
                case_history["steps"].append({"step": "2_Inquiry", "generated_questions": questions})
                if questions:
                    inquiry_context = "To provide a robust answer, the following questions should be considered:\n- " + "\n- ".join(questions)
                    inquiry_context += "\n\nGiven this, here is a preliminary analysis based on the limited information:"
            else:
                 case_history["steps"].append({"step": "2_Inquiry", "generated_questions": "Skipped as per planner's decision."})

            # === STEP 3: Preliminary Analysis (Function Module) ===
            analysis_prompt = PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE.format(
                inquiry_context=inquiry_context,
                question=question,
                options_text=options_text,
                image_text=image_text
            )
            preliminary_response_str = self._call_llm(analysis_prompt, image_path, expect_json=True)
            case_history["steps"].append({"step": "3_Preliminary_Analysis", "response": preliminary_response_str})

            # === STEP 4: Safety Module ("Discuss" Phase) ===
            ethics_feedback = self._call_llm(SAFETY_ETHICS_PROMPT.format(preliminary_response=preliminary_response_str), expect_json=False)
            emergency_feedback = self._call_llm(SAFETY_EMERGENCY_PROMPT.format(preliminary_response=preliminary_response_str), expect_json=False)
            error_feedback = self._call_llm(SAFETY_ERROR_PROMPT.format(preliminary_response=preliminary_response_str), expect_json=False)
            case_history["steps"].append({
                "step": "4_Safety_Review",
                "ethics_feedback": ethics_feedback,
                "emergency_feedback": emergency_feedback,
                "error_feedback": error_feedback
            })

            # === STEP 5: Final Modification ("Modify" Phase) ===
            final_prompt = FINAL_MODIFICATION_PROMPT_TEMPLATE.format(
                question=question, options_text=options_text, image_text=image_text,
                preliminary_response=preliminary_response_str,
                ethics_feedback=ethics_feedback,
                emergency_feedback=emergency_feedback,
                error_feedback=error_feedback
            )
            final_response_str = self._call_llm(final_prompt, image_path, expect_json=True)
            case_history["steps"].append({"step": "5_Final_Modification", "response": final_response_str})

            # === STEP 6: Parse Final Result ===
            final_result_json = json.loads(preprocess_response_string(final_response_str))
            predicted_answer = final_result_json.get("answer", "Parsing Error")
            explanation = final_result_json.get("explanation", "Parsing Error")

        except Exception as e:
            print(f"FATAL ERROR during query processing for QID {qid}: {e}")
            predicted_answer = "Framework Error"
            explanation = str(e)
            case_history["error"] = str(e)

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s. Final Answer: {predicted_answer}")

        # Assemble final result object in the required format
        final_output = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "case_history": case_history,
            "processing_time": processing_time
        }
        return final_output

def main():
    parser = argparse.ArgumentParser(description="Run HealthcareAgent Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--model", type=str, default="qwen-vl-max", help="Model key to use for all agent steps")
    args = parser.parse_args()

    method_name = "HealthcareAgent"

    # Set up paths
    logs_dir = os.path.join("logs", "medqa", args.dataset, "multiple_choice" if args.qa_type == "mc" else "free-form", method_name)
    os.makedirs(logs_dir, exist_ok=True)
    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"

    # Load data
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Initialize the framework
    framework = HealthcareAgentFramework(model_key=args.model)

    # Process each item in the dataset
    for item in tqdm(data, desc=f"Running HealthcareAgent on {args.dataset}"):
        qid = item["qid"]
        result_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            result = framework.run_query(item)
            save_json(result, result_path)
        except Exception as e:
            print(f"CRITICAL MAIN LOOP ERROR processing item {qid}: {e}")
            # Save an error file
            error_result = {
                "qid": qid,
                "error": str(e),
                "timestamp": int(time.time())
            }
            save_json(error_result, result_path)

if __name__ == "__main__":
    main()