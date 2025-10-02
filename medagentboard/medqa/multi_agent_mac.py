"""
medagentboard/medqa/multi_agent_mac.py
"""

import os
import time
import argparse
import json
from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string

# --- Constants and Enums ---

class AgentRole(Enum):
    """Enumeration for different agent roles in the MAC framework."""
    DOCTOR = "Doctor"
    SUPERVISOR = "Supervisor"

# Default settings from the paper and for the framework
DEFAULT_DOCTOR_MODEL = "qwen-vl-max"
DEFAULT_SUPERVISOR_MODEL = "qwen-vl-max" # Supervisor might need strong reasoning
DEFAULT_NUM_DOCTORS = 4  # Optimal number identified in the paper
DEFAULT_MAX_ROUNDS = 5   # Paper mentions up to 13, but 5 is a practical starting point to balance performance and cost

# --- Base Agent Class (as provided) ---

class BaseAgent:
    """Base class for all agents in the MAC framework, adapted from your code."""

    def __init__(self,
                 agent_id: str,
                 role: Union[AgentRole, str],
                 model_key: str,
                 instruction: Optional[str] = None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent.
            role: The role of the agent (Doctor, Supervisor).
            model_key: Key for the LLM model configuration in LLM_MODELS_SETTINGS.
            instruction: System-level instruction defining the agent's persona and task.
        """
        self.agent_id = agent_id
        self.role = role if isinstance(role, str) else role.value
        self.model_key = model_key
        self.instruction = instruction or f"You are a helpful assistant playing the role of a {self.role}."

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.llm_client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        print(f"Initialized Agent: ID={self.agent_id}, Role={self.role}, Model={self.model_key} ({self.model_name})")

    def call_llm(self,
                 messages: List[Dict[str, Any]],
                 max_retries: int = 3) -> str:
        """
        Call the LLM with a list of messages and handle retries.

        Args:
            messages: List of message dictionaries.
            max_retries: Maximum number of retry attempts.

        Returns:
            LLM response text.
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM ({self.model_name}). Attempt {retries + 1}/{max_retries}.")
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                response = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response successfully.")
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error for agent {self.agent_id} (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed for agent {self.agent_id} after {max_retries} attempts: {e}")
                time.sleep(2)
        raise Exception(f"LLM call failed unexpectedly for agent {self.agent_id}.")

# --- MAC Framework Class ---

class MACFramework:
    """
    Orchestrates the Multi-Agent Conversation (MAC) workflow based on the paper.
    This framework facilitates a discussion between multiple Doctor agents and a Supervisor agent.
    """

    def __init__(self,
                 log_dir: str,
                 dataset_name: str,
                 doctor_model_key: str = DEFAULT_DOCTOR_MODEL,
                 supervisor_model_key: str = DEFAULT_SUPERVISOR_MODEL,
                 num_doctors: int = DEFAULT_NUM_DOCTORS,
                 max_rounds: int = DEFAULT_MAX_ROUNDS):
        """
        Initialize the MAC framework orchestrator.

        Args:
            log_dir: Directory to save logs and results.
            dataset_name: Name of the dataset being processed.
            doctor_model_key: Model key for all Doctor agents.
            supervisor_model_key: Model key for the Supervisor agent.
            num_doctors: The number of Doctor agents to use in the conversation.
            max_rounds: The maximum number of conversational rounds.
        """
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.num_doctors = num_doctors
        self.max_rounds = max_rounds
        os.makedirs(self.log_dir, exist_ok=True)

        # --- Initialize Agents based on paper's roles ---
        self.doctor_agents = [
            BaseAgent(
                agent_id=f"doctor_{i+1}",
                role=AgentRole.DOCTOR,
                model_key=doctor_model_key,
                instruction=(
                    "You are an expert medical professional. Your task is to analyze the provided medical case, which includes a question, optional multiple-choice options, and possibly an image. "
                    "You will participate in a multi-agent discussion. In each round, review the conversation history and the opinions of other doctors. "
                    "Then, provide your own updated analysis, clearly stating your reasoning and conclusion. "
                    "If you change your mind based on others' arguments, explain why. Your goal is to contribute to a correct and well-reasoned consensus. "
                    "Respond in JSON format with 'explanation' and 'answer' fields."
                )
            ) for i in range(num_doctors)
        ]

        self.supervisor_agent = BaseAgent(
            agent_id="supervisor",
            role=AgentRole.SUPERVISOR,
            model_key=supervisor_model_key,
            instruction=(
                "You are the Supervisor of a medical multi-agent discussion. Your role is to facilitate the conversation and drive towards a consensus. "
                "After each round of discussion among the Doctor agents, you will: "
                "1. Summarize the current state of the discussion, noting points of agreement and disagreement. "
                "2. Challenge the doctors' reasoning if it seems weak or contradictory. "
                "3. Evaluate if a consensus has been reached. A consensus is defined as strong agreement among the majority of doctors on both the answer and the core reasoning. "
                "4. If consensus is reached or this is the final round, provide the final definitive answer. "
                "Respond in JSON format with 'summary' (your analysis of the round), 'consensus_reached' (boolean), and 'final_answer' (your final concluded answer, which can be null if consensus is not yet reached)."
            )
        )

        print("MACFramework Initialized.")
        print(f" - Log Directory: {self.log_dir}")
        print(f" - Dataset: {self.dataset_name}")
        print(f" - Models: Doctors={doctor_model_key}, Supervisor={supervisor_model_key}")
        print(f" - Settings: Doctors={self.num_doctors}, Max Rounds={self.max_rounds}")

    def _format_initial_prompt(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formats the initial problem statement from the Admin Agent's perspective."""
        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")

        # The user message content can be a list (for VQA) or a string (for QA)
        user_content: Union[str, List[Dict[str, Any]]]

        prompt_text = f"A new case has been presented. Please begin the diagnostic discussion.\n\n--- Case Information ---\nQuestion: {question}\n"
        if options:
            options_str = "\n".join([f"({k}) {v}" for k, v in options.items()])
            prompt_text += f"Options:\n{options_str}\n"

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            base64_image = encode_image(image_path)
            user_content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        else:
            user_content = prompt_text

        return [{"role": "user", "content": user_content}]

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Converts the list of conversation turns into a readable string."""
        formatted_history = "--- Start of Conversation History ---\n"
        for turn in history:
            # Handle different content structures (string vs. list for VQA)
            content = turn.get("content")
            if isinstance(content, list):
                # Extract text part for history
                text_content = next((item['text'] for item in content if item['type'] == 'text'), "")
                content_str = f"{text_content} [Image was provided]"
            else:
                content_str = str(content)

            # The role is now a string like 'Doctor (doctor_1)'
            formatted_history += f"Role: {turn['role']}\n"
            formatted_history += f"Message: {content_str}\n"
            formatted_history += "-------------------------------------\n"
        formatted_history += "--- End of Conversation History ---\n"
        return formatted_history

    def run_query(self, data_item: Dict) -> Dict:
        """
        Processes a single data item through the MAC framework.

        Args:
            data_item: Dictionary containing query details.

        Returns:
            A dictionary containing the full results and conversation log.
        """
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()

        conversation_log = []
        final_answer_obj = {"answer": "Error", "explanation": "Processing failed to produce a final answer."}

        try:
            # The 'Admin Agent' provides the initial information
            initial_messages = self._format_initial_prompt(data_item)
            conversation_log.append({
                "role": "Admin",
                "content": initial_messages[0]['content']
            })

            for round_num in range(1, self.max_rounds + 1):
                print(f"\n--- Starting Round {round_num}/{self.max_rounds} for QID: {qid} ---")

                # --- Doctors' Turn ---
                round_doctor_responses = []
                for doctor in self.doctor_agents:
                    history_str = self._format_conversation_history(conversation_log)
                    doctor_prompt = (
                        f"{history_str}\n"
                        f"This is round {round_num}. Based on the full conversation history, provide your updated analysis. "
                        "If other doctors have provided compelling arguments, acknowledge them and refine your position. "
                        "State your current answer and explanation clearly."
                    )

                    # The message list for the LLM needs to include the initial prompt with the image
                    messages_for_llm = [
                        {"role": "system", "content": doctor.instruction},
                        *initial_messages,
                        {"role": "user", "content": doctor_prompt}
                    ]

                    response_str = doctor.call_llm(messages_for_llm)
                    round_doctor_responses.append({
                        "role": f"Doctor ({doctor.agent_id})",
                        "content": response_str
                    })

                conversation_log.extend(round_doctor_responses)

                # --- Supervisor's Turn ---
                print(f"\n--- Supervisor Turn for Round {round_num} ---")
                history_str = self._format_conversation_history(conversation_log)
                supervisor_prompt = (
                    f"{history_str}\n"
                    f"This is the end of round {round_num}. As the Supervisor, please analyze the doctors' latest inputs. "
                    "Provide your summary, challenge any weak points, and determine if consensus has been reached. "
                    f"If consensus is met or if this is the final round ({self.max_rounds}), you must provide the 'final_answer'."
                )

                # Supervisor does not need the image, only the text discussion
                messages_for_llm = [
                    {"role": "system", "content": self.supervisor_agent.instruction},
                    {"role": "user", "content": supervisor_prompt}
                ]

                supervisor_response_str = self.supervisor_agent.call_llm(messages_for_llm)
                conversation_log.append({
                    "role": f"Supervisor ({self.supervisor_agent.agent_id})",
                    "content": supervisor_response_str
                })

                # Parse supervisor's response to check for consensus
                try:
                    supervisor_json = json.loads(preprocess_response_string(supervisor_response_str))
                    consensus_reached = supervisor_json.get("consensus_reached", False)
                    final_answer_from_supervisor = supervisor_json.get("final_answer")

                    print(f"Supervisor Summary: {supervisor_json.get('summary', 'N/A')}")
                    print(f"Consensus Reached: {consensus_reached}")

                    if final_answer_from_supervisor:
                        # The final answer could be a string or a dict. We want a dict.
                        if isinstance(final_answer_from_supervisor, dict) and "answer" in final_answer_from_supervisor:
                            final_answer_obj = final_answer_from_supervisor
                        else:
                            # If it's not a dict, we wrap it. This is a fallback.
                            final_answer_obj = {"answer": final_answer_from_supervisor, "explanation": supervisor_json.get('summary', '')}

                    if consensus_reached:
                        print("Consensus reached. Ending conversation.")
                        break

                    if round_num == self.max_rounds and not final_answer_from_supervisor:
                        print("Max rounds reached. Supervisor did not provide a final answer. Forcing a final decision.")
                        # This would be a place to make one last call to the supervisor asking for a forced decision.
                        # For simplicity, we'll use the last summary as the basis for the answer.
                        final_answer_obj = {
                            "answer": "Inconclusive",
                            "explanation": f"Max rounds reached without a clear final answer. Last summary: {supervisor_json.get('summary', 'N/A')}"
                        }


                except json.JSONDecodeError:
                    print(f"Error: Could not parse supervisor's response in round {round_num}. Continuing.")
                except Exception as e:
                    print(f"An unexpected error occurred while processing supervisor response: {e}")

        except Exception as e:
            print(f"ERROR processing QID {qid}: {e}")
            final_answer_obj = {"answer": "Error", "explanation": str(e)}

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s")

        # Assemble final result object
        final_result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
            "ground_truth": data_item.get("answer"),
            "predicted_answer": final_answer_obj.get("answer", "Error"),
            "explanation": final_answer_obj.get("explanation", "N/A"),
            "processing_time_seconds": processing_time,
            "details": {
                "conversation_log": conversation_log
            }
        }

        return final_result

    def run_dataset(self, data: List[Dict]):
        """
        Runs the MAC framework over an entire dataset.

        Args:
            data: List of data items (dictionaries).
        """
        print(f"\nStarting MAC framework processing for {len(data)} items in dataset '{self.dataset_name}'.")

        for item in tqdm(data, desc=f"Running MAC on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")
            result_path = os.path.join(self.log_dir, f"{qid}-result.json")

            if os.path.exists(result_path):
                print(f"Skipping {qid} - result file already exists.")
                continue

            try:
                result = self.run_query(item)
                save_json(result, result_path)
            except Exception as e:
                print(f"FATAL ERROR during run_query for QID {qid}: {e}")
                # Save an error record
                error_result = {"qid": qid, "error": str(e)}
                save_json(error_result, result_path)

        print(f"Finished processing dataset '{self.dataset_name}'. Results saved in {self.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MAC Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name (e.g., vqa_rad, pathvqa, medqa)")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--doctor_model", type=str, default=DEFAULT_DOCTOR_MODEL, help="Model key for the Doctor agents")
    parser.add_argument("--supervisor_model", type=str, default=DEFAULT_SUPERVISOR_MODEL, help="Model key for the Supervisor agent")
    parser.add_argument("--num_doctors", type=int, default=DEFAULT_NUM_DOCTORS, help="Number of doctor agents to use")
    parser.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="Maximum number of discussion rounds")

    args = parser.parse_args()

    method_name = "MAC"

    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"
    logs_dir = os.path.join("./logs", "medqa", args.dataset,
                           "multiple_choice" if args.qa_type == "mc" else "free-form",
                           method_name)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Using Log Directory: {logs_dir}")

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    framework = MACFramework(
        log_dir=logs_dir,
        dataset_name=args.dataset,
        doctor_model_key=args.doctor_model,
        supervisor_model_key=args.supervisor_model,
        num_doctors=args.num_doctors,
        max_rounds=args.max_rounds
    )

    framework.run_dataset(data)

if __name__ == "__main__":
    main()