"""
medagentboard/medqa/multi_agent_reconcile.py

This module implements the Reconcile framework for multi-model,
multi-agent discussion. Each agent generates an answer with step-by-step
reasoning and an estimated confidence level. Then, the agents engage in
multi-round discussions and a confidence-weighted vote produces the final team answer.
"""

import os
import json
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
from tqdm import tqdm

# Import ColaCare utilities
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from medagentboard.utils.encode_image import encode_image


###############################################################################
# Discussion Phase Enumeration
###############################################################################
class DiscussionPhase(Enum):
    """Enumeration of discussion phases in the Reconcile framework."""
    INITIAL = "initial"        # Initial answer generation
    DISCUSSION = "discussion"  # Multi-round discussion
    FINAL = "final"            # Final team answer


###############################################################################
# ReconcileAgent: an LLM agent for the Reconcile framework
###############################################################################
class ReconcileAgent:
    """
    An agent participating in the Reconcile framework.

    Each agent uses a specified LLM model to generate an answer,
    detailed reasoning, and an estimated confidence level (between 0.0 and 1.0).

    Attributes:
        agent_id: Unique identifier for the agent
        model_key: Key of the LLM model in LLM_MODELS_SETTINGS
        model_name: Name of the model used by this agent
        client: OpenAI-compatible client for making API calls
        discussion_history: List of agent's responses throughout the discussion
        memory: Agent's memory of the case
    """
    def __init__(self, agent_id: str, model_key: str):
        """
        Initialize a Reconcile agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: Key of the LLM model in LLM_MODELS_SETTINGS

        Raises:
            ValueError: If model_key is not found in LLM_MODELS_SETTINGS
        """
        self.agent_id = agent_id
        self.model_key = model_key
        self.discussion_history = []
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not configured in LLM_MODELS_SETTINGS")
        self.model_config = LLM_MODELS_SETTINGS[model_key]

        # Set up the LLM client using the OpenAI-based client from ColaCare
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("OpenAI client is not installed. Please install it.") from e

        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"],
        )
        self.model_name = self.model_config["model_name"]
        print(f"Initialized agent {self.agent_id} with model {self.model_name}")

    def call_llm(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """
        Call the LLM with the provided messages and a retry mechanism.

        Args:
            messages: List of messages (each as a dictionary) to send to the LLM
            max_retries: Maximum number of retry attempts

        Returns:
            The text content from the LLM response
        """
        attempt = 0
        wait_time = 1

        while attempt < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM with model {self.model_name} (attempt {attempt+1}/{max_retries})")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                response_text = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response: {response_text[:100]}...")
                return response_text
            except Exception as e:
                attempt += 1
                print(f"Agent {self.agent_id} LLM call attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        # If all retries fail, return an error JSON message
        print(f"Agent {self.agent_id} all LLM call attempts failed, returning default response")
        return json.dumps({
            "reasoning": "LLM call failed after multiple attempts",
            "answer": "",
            "confidence": 0.0
        })

    def generate_initial_response(self,
                                question: str,
                                options: Optional[Dict[str, str]] = None,
                                image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an initial response for a given question.

        Args:
            question: The input question text
            options: Optional multiple choice options
            image_path: Optional path to an image for MedVQA

        Returns:
            A dictionary containing reasoning, answer, and confidence
        """
        print(f"Agent {self.agent_id} generating initial response")

        # Construct system message
        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert assistant. Analyze the following medical question "
                "and provide a clear answer along with detailed step-by-step reasoning. "
                "Based on your understanding, estimate your confidence in your answer "
                "on a scale from 0.0 to 1.0, where 1.0 means complete certainty."
            )
        }

        # Construct user message
        user_content = []

        # Add image if provided
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                print(f"Agent {self.agent_id} added image from {image_path}")
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        # Format question with options if provided
        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"

        # Add formatted question
        prompt_text = (
            f"{question_text}\n\n"
            f"Provide your response in JSON format with the following fields:\n"
            f"- 'reasoning': your detailed step-by-step analysis\n"
            f"- 'answer': your final answer"
        )

        if options:
            prompt_text += " (specify just the option letter)"

        prompt_text += (
            f"\n- 'confidence': a number between 0.0 and 1.0 representing your confidence level\n\n"
            f"Ensure your JSON is properly formatted."
        )

        user_content.append({
            "type": "text",
            "text": prompt_text
        })

        user_message = {
            "role": "user",
            "content": user_content
        }

        # Call LLM and parse response
        response_text = self.call_llm([system_message, user_message])
        result = self._parse_response(response_text)

        # Store in agent's memory
        self.memory.append({
            "phase": DiscussionPhase.INITIAL.value,
            "response": result
        })

        return result

    def generate_discussion_response(self,
                                  question: str,
                                  discussion_prompt: str,
                                  options: Optional[Dict[str, str]] = None,
                                  image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response during the discussion phase.

        Args:
            question: The original question
            discussion_prompt: The formatted discussion prompt with other agents' responses
            options: Optional multiple choice options
            image_path: Optional path to an image for MedVQA

        Returns:
            A dictionary containing reasoning, answer, and confidence
        """
        print(f"Agent {self.agent_id} generating discussion response")

        # Construct system message
        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert participating in a multi-agent discussion. "
                "Review the opinions from other experts, then provide your updated analysis. "
                "You may change your opinion if others' reasoning convinces you, or defend your position "
                "with clear explanations. Estimate your confidence in your answer on a scale from 0.0 to 1.0."
            )
        }

        # Construct user message
        user_content = []

        # Add image if provided
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        # Format question with options if provided
        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"

        # Add question and discussion prompt
        prompt_text = (
            f"Original question: {question_text}\n\n"
            f"Discussion from other experts:\n{discussion_prompt}\n\n"
            f"Based on this discussion, provide your updated analysis in JSON format with the following fields:\n"
            f"- 'reasoning': your detailed step-by-step analysis\n"
            f"- 'answer': your final answer"
        )

        if options:
            prompt_text += " (specify just the option letter)"

        prompt_text += (
            f"\n- 'confidence': a number between 0.0 and 1.0 representing your confidence level\n\n"
            f"Ensure your JSON is properly formatted."
        )

        user_content.append({
            "type": "text",
            "text": prompt_text
        })

        user_message = {
            "role": "user",
            "content": user_content
        }

        # Call LLM and parse response
        response_text = self.call_llm([system_message, user_message])
        result = self._parse_response(response_text)

        # Determine the current round number
        current_round = sum(1 for mem in self.memory if mem["phase"] == DiscussionPhase.DISCUSSION.value) + 1

        # Store in agent's memory
        self.memory.append({
            "phase": DiscussionPhase.DISCUSSION.value,
            "round": current_round,
            "response": result
        })

        return result

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.

        Args:
            response_text: The raw response text from the LLM

        Returns:
            A dictionary with reasoning, answer, and confidence
        """
        try:
            result = json.loads(preprocess_response_string(response_text))

            # Validate required fields
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"

            if "answer" not in result:
                result["answer"] = ""

            if "confidence" not in result:
                result["confidence"] = 0.0
            else:
                # Ensure confidence is a float between 0 and 1
                try:
                    result["confidence"] = float(result["confidence"])
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except (ValueError, TypeError):
                    result["confidence"] = 0.0

            return result

        except json.JSONDecodeError:
            print(f"Agent {self.agent_id} failed to parse JSON response: {response_text[:100]}...")

            # Attempt to extract with simple parsing
            reasoning = ""
            answer = ""
            confidence = 0.0

            lines = response_text.split('\n')
            for line in lines:
                if line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, IndexError):
                        confidence = 0.0

            # If basic parsing doesn't work, use the raw text
            if not reasoning:
                reasoning = response_text

            return {
                "reasoning": reasoning,
                "answer": answer,
                "confidence": confidence
            }


###############################################################################
# ReconcileCoordinator: orchestrates the multi-agent discussion process
###############################################################################
class ReconcileCoordinator:
    """
    The coordinator for the Reconcile framework.

    This class orchestrates the following phases:
    1. Initial Response Generation: Each agent generates an initial response.
    2. Multi-Round Discussion: Agents update their responses based on the grouped responses.
    3. Team Answer Generation: A confidence-weighted vote decides the final answer.

    Attributes:
        agents: List of ReconcileAgent objects participating in the discussion
        max_rounds: Maximum number of discussion rounds
    """
    def __init__(self, agent_configs: List[Dict[str, str]], max_rounds: int = 3):
        """
        Initialize the Reconcile coordinator.

        Args:
            agent_configs: List of agent configurations (each with agent_id and model_key)
            max_rounds: Maximum number of discussion rounds
        """
        # Instantiate Reconcile agents using provided configurations
        self.agents = [
            ReconcileAgent(cfg["agent_id"], cfg["model_key"])
            for cfg in agent_configs
        ]
        self.max_rounds = max_rounds
        print(f"Initialized ReconcileCoordinator with {len(self.agents)} agents, max_rounds={max_rounds}")

    def _group_answers(self, answers: List[Dict[str, Any]]) -> str:
        """
        Group and summarize responses from agents.

        Args:
            answers: List of agent response dictionaries

        Returns:
            A formatted string with grouped answers and their supporting explanations
        """
        groups = {}

        # Group answers and explanations
        for ans in answers:
            answer_text = ans.get("answer", "").strip().lower()
            confidence = ans.get("confidence", 0.0)

            if answer_text not in groups:
                groups[answer_text] = {
                    "count": 0,
                    "explanations": [],
                    "confidence_sum": 0.0
                }

            groups[answer_text]["count"] += 1
            groups[answer_text]["explanations"].append(ans.get("reasoning", ""))
            groups[answer_text]["confidence_sum"] += confidence

        # Format grouped answers
        grouped_str = ""
        for ans_text, data in groups.items():
            # Calculate average confidence
            avg_confidence = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0

            grouped_str += f"Answer: {ans_text}\n"
            grouped_str += f"Supporters: {data['count']}\n"
            grouped_str += f"Average confidence: {avg_confidence:.2f}\n"
            grouped_str += f"Explanations:\n"

            # Add each explanation with a bullet point
            for i, exp in enumerate(data["explanations"]):
                # Truncate very long explanations
                if len(exp) > 500:
                    exp = exp[:500] + "... (truncated)"
                grouped_str += f"• Expert {i+1}: {exp}\n"

            grouped_str += "\n"

        return grouped_str.strip()

    def _recalibrate(self, confidence: float) -> float:
        """
        Recalibrate a confidence score for better voting weights.

        Args:
            confidence: The original confidence score (0.0 to 1.0)

        Returns:
            Recalibrated confidence score
        """
        if confidence == 1.0:
            return 1.0
        elif confidence >= 0.9:
            return 0.8
        elif confidence >= 0.8:
            return 0.5
        elif confidence > 0.6:
            return 0.3
        else:
            return 0.1

    def _weighted_vote(self, answers: List[Dict[str, Any]]) -> str:
        """
        Compute the final team answer using a confidence-weighted vote.

        Args:
            answers: List of response dictionaries from agents

        Returns:
            The final answer string
        """
        vote_weights = {}

        # Calculate weights for each answer
        for ans in answers:
            answer = ans.get("answer", "").strip()
            if not answer:
                continue

            confidence = ans.get("confidence", 0.0)
            weight = self._recalibrate(confidence)

            # Normalize answer to lowercase for vote counting, but preserve original case
            key = answer.lower()

            if key not in vote_weights:
                vote_weights[key] = {"weight": 0, "original": answer}

            vote_weights[key]["weight"] += weight

        if not vote_weights:
            return ""

        # Find the answer with the highest weight
        winner_key = max(vote_weights, key=lambda k: vote_weights[k]["weight"])
        final_decision = vote_weights[winner_key]["original"]

        return final_decision

    def _check_consensus(self, answers: List[Dict[str, Any]]) -> bool:
        """
        Check if all agents provided the same answer (consensus reached).

        Args:
            answers: List of response dictionaries from agents

        Returns:
            True if consensus reached, False otherwise
        """
        # Extract valid answers and convert to lowercase for comparison
        valid_answers = [ans.get("answer", "").strip().lower() for ans in answers if ans.get("answer", "").strip()]

        # Check if there's at least one valid answer and all are the same
        return len(valid_answers) > 0 and len(set(valid_answers)) == 1

    def run_discussion(self,
                      question: str,
                      options: Optional[Dict[str, str]] = None,
                      image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete discussion process.

        Args:
            question: The input question
            options: Optional multiple choice options
            image_path: Optional path to an image for MedVQA

        Returns:
            Dictionary with the final team answer and discussion history
        """
        print(f"Starting discussion with {len(self.agents)} agents on question: {question}")
        start_time = time.time()

        discussion_history = []

        # Phase 1: Initial responses
        print("Phase 1: Generating initial responses")
        current_answers = []

        for agent in self.agents:
            resp = agent.generate_initial_response(question, options, image_path)
            current_answers.append(resp)

            # Add to discussion history
            discussion_history.append({
                "phase": DiscussionPhase.INITIAL.value,
                "agent_id": agent.agent_id,
                "response": resp
            })

            print(f"Agent {agent.agent_id} initial answer: {resp.get('answer', '')} (confidence: {resp.get('confidence', 0.0):.2f})")

        # Phase 2: Multi-round discussion
        round_num = 0
        consensus_reached = False

        while round_num < self.max_rounds and not consensus_reached:
            round_num += 1
            print(f"Phase 2: Discussion round {round_num}/{self.max_rounds}")

            # Prepare the discussion prompt based on previous answers
            discussion_prompt = self._group_answers(current_answers)

            # Each agent generates a new response
            new_answers = []
            for agent in self.agents:
                resp = agent.generate_discussion_response(
                    question, discussion_prompt, options, image_path
                )
                new_answers.append(resp)

                # Add to discussion history
                discussion_history.append({
                    "phase": DiscussionPhase.DISCUSSION.value,
                    "round": round_num,
                    "agent_id": agent.agent_id,
                    "response": resp
                })

                print(f"Agent {agent.agent_id} round {round_num} answer: {resp.get('answer', '')} (confidence: {resp.get('confidence', 0.0):.2f})")

            # Update current answers for next round
            current_answers = new_answers

            # Check if consensus is reached
            consensus_reached = self._check_consensus(current_answers)
            print(f"Round {round_num} consensus reached: {consensus_reached}")

            if consensus_reached:
                print("Consensus reached, ending discussion")
                break

        # Phase 3: Final team answer via weighted vote
        print("Phase 3: Generating final team answer")
        final_decision = self._weighted_vote(current_answers)

        # Add final decision to history
        discussion_history.append({
            "phase": DiscussionPhase.FINAL.value,
            "final_decision": final_decision,
            "consensus_reached": consensus_reached,
            "rounds_completed": round_num,
            "confidence_scores": [ans.get("confidence", 0.0) for ans in current_answers]
        })

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Discussion completed in {processing_time:.2f} seconds. Final answer: {final_decision}")

        return {
            "final_decision": final_decision,
            "discussion_history": discussion_history,
            "processing_time": processing_time
        }


###############################################################################
# Process a Single QA Item with the Reconcile Framework
###############################################################################
def process_item(item: Dict[str, Any],
               agent_configs: List[Dict[str, str]],
               max_rounds: int = 3) -> Dict[str, Any]:
    """
    Process a single QA item with the Reconcile framework.

    Args:
        item: Input QA item dictionary (with qid, question, etc.)
        agent_configs: List of agent configurations (each with agent_id and model_key)
        max_rounds: Maximum number of discussion rounds

    Returns:
        Processed QA result with the final predicted answer and discussion history
    """
    qid = item.get("qid", "unknown")
    question = item.get("question", "")
    options = item.get("options")
    image_path = item.get("image_path")
    ground_truth = item.get("answer")

    print(f"Processing item {qid}")

    # Create coordinator and run discussion
    coordinator = ReconcileCoordinator(agent_configs, max_rounds)
    discussion_result = coordinator.run_discussion(question, options, image_path)

    # Compile results
    result = {
        "qid": qid,
        "timestamp": int(time.time()),
        "question": question,
        "options": options,
        "image_path": image_path,
        "ground_truth": ground_truth,
        "predicted_answer": discussion_result["final_decision"],
        "case_history": discussion_result,
    }

    return result


###############################################################################
# Main Entry Point for the Reconcile Framework
###############################################################################
def main():
    """
    Main entry point for running the Reconcile framework from command line.
    """
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on medical QA datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True,
                       help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--agents", nargs='+', default=["qwen-max-latest", "deepseek-v3-ark", "qwen-vl-max"],
                       help="List of agent model keys (e.g., deepseek-v3-ark, qwen-max-latest, qwen-vl-max)")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of discussion rounds")

    args = parser.parse_args()
    method = "ReConcile"

    # Extract dataset name
    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")

    # Determine QA format (multiple choice or free-form)
    qa_type = args.qa_type
    print(f"QA Format: {qa_type}")

    # Create logs directory structure
    logs_dir = os.path.join("logs", dataset_name, "multiple_choice" if qa_type == "mc" else "free-form", method)
    os.makedirs(logs_dir, exist_ok=True)

    # Construct the data path
    data_path = os.path.join("my_datasets", "processed", args.dataset, f"medqa_{args.qa_type}.json")

    # Load the dataset
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Configure agents: each agent is assigned an ID and a model key
    agent_configs = []
    for idx, model_key in enumerate(args.agents, 1):
        agent_configs.append({"agent_id": f"agent_{idx}", "model_key": model_key})

    print(f"Configured {len(agent_configs)} agents: {[cfg['model_key'] for cfg in agent_configs]}")


    # Process each item in the dataset
    for item in tqdm(data, desc=f"Processing {dataset_name} ({qa_type})"):
        qid = item.get("qid")
        result_path = os.path.join(logs_dir, f"{qid}-result.json")

        # Skip already processed items
        if os.path.exists(result_path):
            print(f"Skipping {qid} (already processed)")
            continue

        try:
            # Process the item
            result = process_item(item, agent_configs, args.max_rounds)
            # Save result
            save_json(result, result_path)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")

if __name__ == "__main__":
    main()