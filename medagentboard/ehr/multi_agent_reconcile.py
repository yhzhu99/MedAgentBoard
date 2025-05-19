"""
medagentboard/ehr/multi_agent_reconcile.py

This module implements the Reconcile framework for multi-model,
multi-agent discussion for EHR predictive modeling tasks. Each agent generates
a prediction with step-by-step reasoning and an estimated confidence level.
Then, the agents engage in multi-round discussions and a confidence-weighted
aggregation produces the final team prediction.
"""

import os
import json
import time
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
from tqdm import tqdm

# Import utilities
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


###############################################################################
# Discussion Phase Enumeration
###############################################################################
class DiscussionPhase(Enum):
    """Enumeration of discussion phases in the Reconcile framework."""
    INITIAL = "initial"        # Initial prediction generation
    DISCUSSION = "discussion"  # Multi-round discussion
    FINAL = "final"            # Final team prediction


###############################################################################
# ReconcileAgent: an LLM agent for the Reconcile framework
###############################################################################
class ReconcileAgent:
    """
    An agent participating in the Reconcile framework for EHR prediction.

    Each agent uses a specified LLM model to generate a prediction,
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

        # Set up the LLM client using the OpenAI-based client
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
            "prediction": 0.5,
            "confidence": 0.0
        })

    def generate_initial_response(self, question: str) -> Dict[str, Any]:
        """
        Generate an initial prediction for the EHR time series data.

        Args:
            question: The input question containing EHR data and prediction task

        Returns:
            A dictionary containing reasoning, prediction, and confidence
        """
        print(f"Agent {self.agent_id} generating initial response")

        # Construct system message
        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert specializing in analyzing electronic health records (EHR) "
                "and making clinical predictions. Analyze the following patient data "
                "and provide a clear prediction along with detailed step-by-step reasoning. "
                "Based on your understanding, estimate your confidence in your prediction "
                "on a scale from 0.0 to 1.0, where 1.0 means complete certainty."
            )
        }

        # Construct user message
        prompt_text = (
            f"{question}\n\n"
            f"Provide your response in JSON format with the following fields:\n"
            f"- 'reasoning': your detailed step-by-step analysis of the patient data\n"
            f"- 'prediction': a floating-point number between 0 and 1 representing the predicted probability\n"
            f"- 'confidence': a number between 0.0 and 1.0 representing your confidence level in your prediction\n\n"
            f"Ensure your JSON is properly formatted."
        )

        user_message = {
            "role": "user",
            "content": prompt_text
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

    def generate_discussion_response(self, question: str, discussion_prompt: str) -> Dict[str, Any]:
        """
        Generate a response during the discussion phase.

        Args:
            question: The original question with EHR data
            discussion_prompt: The formatted discussion prompt with other agents' responses

        Returns:
            A dictionary containing reasoning, prediction, and confidence
        """
        print(f"Agent {self.agent_id} generating discussion response")

        # Construct system message
        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert participating in a multi-agent discussion about "
                "electronic health records (EHR) analysis. Review the opinions from other experts, "
                "then provide your updated analysis. You may adjust your prediction if others' "
                "reasoning convinces you, or defend your position with clear explanations. "
                "Estimate your confidence in your prediction on a scale from 0.0 to 1.0."
            )
        }

        # Construct user message
        prompt_text = (
            f"Original patient data and task:\n{question}\n\n"
            f"Discussion from other experts:\n{discussion_prompt}\n\n"
            f"Based on this discussion, provide your updated analysis in JSON format with the following fields:\n"
            f"- 'reasoning': your detailed step-by-step analysis of the patient data\n"
            f"- 'prediction': a floating-point number between 0 and 1 representing the predicted probability\n"
            f"- 'confidence': a number between 0.0 and 1.0 representing your confidence level in your prediction\n\n"
            f"Ensure your JSON is properly formatted."
        )

        user_message = {
            "role": "user",
            "content": prompt_text
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
            A dictionary with reasoning, prediction, and confidence
        """
        try:
            result = json.loads(preprocess_response_string(response_text))

            # Validate required fields
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"

            if "prediction" not in result:
                result["prediction"] = 0.5
            else:
                # Ensure prediction is a float between 0 and 1
                try:
                    result["prediction"] = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, result["prediction"]))
                except (ValueError, TypeError):
                    result["prediction"] = 0.5

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
            prediction = 0.5
            confidence = 0.0

            lines = response_text.split('\n')
            for line in lines:
                if line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.lower().startswith("prediction:"):
                    try:
                        prediction = float(line.split(":", 1)[1].strip())
                        prediction = max(0.0, min(1.0, prediction))
                    except (ValueError, IndexError):
                        prediction = 0.5
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
                "prediction": prediction,
                "confidence": confidence
            }


###############################################################################
# ReconcileCoordinator: orchestrates the multi-agent discussion process
###############################################################################
class ReconcileCoordinator:
    """
    The coordinator for the Reconcile framework in EHR prediction tasks.

    This class orchestrates the following phases:
    1. Initial Prediction Generation: Each agent generates an initial prediction.
    2. Multi-Round Discussion: Agents update their predictions based on the grouped responses.
    3. Team Prediction Generation: A confidence-weighted aggregation produces the final prediction.

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

    def _group_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """
        Group and summarize predictions from agents.

        Args:
            predictions: List of agent response dictionaries

        Returns:
            A formatted string with grouped predictions and their supporting explanations
        """
        # Define groups based on prediction ranges
        groups = {
            "low_risk": {"range": (0.0, 0.33), "count": 0, "explanations": [], "avg_pred": 0.0, "confidence_sum": 0.0},
            "medium_risk": {"range": (0.33, 0.67), "count": 0, "explanations": [], "avg_pred": 0.0, "confidence_sum": 0.0},
            "high_risk": {"range": (0.67, 1.0), "count": 0, "explanations": [], "avg_pred": 0.0, "confidence_sum": 0.0}
        }

        # Group predictions and explanations
        for pred in predictions:
            prediction_value = pred.get("prediction", 0.5)
            confidence = pred.get("confidence", 0.0)
            reasoning = pred.get("reasoning", "")

            # Determine which group this prediction belongs to
            for group_name, group_data in groups.items():
                lower, upper = group_data["range"]
                if lower <= prediction_value < upper or (group_name == "high_risk" and prediction_value == upper):
                    group_data["count"] += 1
                    group_data["explanations"].append(reasoning)
                    group_data["avg_pred"] += prediction_value
                    group_data["confidence_sum"] += confidence
                    break

        # Format grouped predictions
        grouped_str = ""
        for group_name, data in groups.items():
            if data["count"] > 0:
                avg_pred = data["avg_pred"] / data["count"]
                avg_confidence = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0

                grouped_str += f"Prediction Group: {group_name.replace('_', ' ').title()} (Range: {data['range'][0]:.2f}-{data['range'][1]:.2f})\n"
                grouped_str += f"Number of experts in this group: {data['count']}\n"
                grouped_str += f"Average prediction: {avg_pred:.3f}\n"
                grouped_str += f"Average confidence: {avg_confidence:.2f}\n"
                grouped_str += f"Explanations from this group:\n"

                # Add each explanation with a bullet point
                for i, exp in enumerate(data["explanations"]):
                    # Truncate very long explanations
                    if len(exp) > 500:
                        exp = exp[:500] + "... (truncated)"
                    grouped_str += f"• Expert {i+1}: {exp}\n"

                grouped_str += "\n"

        return grouped_str.strip()

    def _consensus_threshold(self, predictions: List[float]) -> bool:
        """
        Check if predictions have reached a reasonable consensus.

        Args:
            predictions: List of prediction values

        Returns:
            True if consensus reached, False otherwise
        """
        if not predictions:
            return False

        # Calculate standard deviation of predictions
        std_dev = np.std(predictions)

        # If standard deviation is below threshold, consider it a consensus
        return std_dev < 0.1  # Threshold can be adjusted based on desired sensitivity

    def _weighted_average(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Compute the final team prediction using a confidence-weighted average.

        Args:
            predictions: List of prediction dictionaries from agents

        Returns:
            The final prediction value
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for pred in predictions:
            prediction = pred.get("prediction", 0.5)
            confidence = pred.get("confidence", 0.0)

            # Square the confidence to give more weight to high-confidence predictions
            weight = confidence ** 2

            weighted_sum += prediction * weight
            total_weight += weight

        # If no valid weights, return simple average
        if total_weight == 0:
            valid_predictions = [p.get("prediction", 0.5) for p in predictions]
            return sum(valid_predictions) / len(valid_predictions) if valid_predictions else 0.5

        return weighted_sum / total_weight

    def run_discussion(self, question: str) -> Dict[str, Any]:
        """
        Run the complete discussion process for an EHR prediction task.

        Args:
            question: The input question containing EHR data

        Returns:
            Dictionary with the final team prediction and discussion history
        """
        print(f"Starting EHR prediction discussion with {len(self.agents)} agents")
        start_time = time.time()

        discussion_history = []

        # Phase 1: Initial predictions
        print("Phase 1: Generating initial predictions")
        current_predictions = []

        for agent in self.agents:
            resp = agent.generate_initial_response(question)
            current_predictions.append(resp)

            # Add to discussion history
            discussion_history.append({
                "phase": DiscussionPhase.INITIAL.value,
                "agent_id": agent.agent_id,
                "response": resp
            })

            print(f"Agent {agent.agent_id} initial prediction: {resp.get('prediction', 0.5):.3f} (confidence: {resp.get('confidence', 0.0):.2f})")

        # Phase 2: Multi-round discussion
        round_num = 0
        consensus_reached = False

        while round_num < self.max_rounds and not consensus_reached:
            round_num += 1
            print(f"Phase 2: Discussion round {round_num}/{self.max_rounds}")

            # Prepare the discussion prompt based on previous predictions
            discussion_prompt = self._group_predictions(current_predictions)

            # Each agent generates a new response
            new_predictions = []
            for agent in self.agents:
                resp = agent.generate_discussion_response(question, discussion_prompt)
                new_predictions.append(resp)

                # Add to discussion history
                discussion_history.append({
                    "phase": DiscussionPhase.DISCUSSION.value,
                    "round": round_num,
                    "agent_id": agent.agent_id,
                    "response": resp
                })

                print(f"Agent {agent.agent_id} round {round_num} prediction: {resp.get('prediction', 0.5):.3f} (confidence: {resp.get('confidence', 0.0):.2f})")

            # Update current predictions for next round
            current_predictions = new_predictions

            # Check if consensus is reached
            prediction_values = [p.get("prediction", 0.5) for p in current_predictions]
            consensus_reached = self._consensus_threshold(prediction_values)
            print(f"Round {round_num} consensus reached: {consensus_reached}")

            if consensus_reached:
                print("Consensus reached, ending discussion")
                break

        # Phase 3: Final team prediction via weighted average
        print("Phase 3: Generating final team prediction")
        final_prediction = self._weighted_average(current_predictions)

        # Add final prediction to history
        discussion_history.append({
            "phase": DiscussionPhase.FINAL.value,
            "final_prediction": final_prediction,
            "consensus_reached": consensus_reached,
            "rounds_completed": round_num,
            "individual_predictions": [p.get("prediction", 0.5) for p in current_predictions],
            "confidence_scores": [p.get("confidence", 0.0) for p in current_predictions]
        })

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Discussion completed in {processing_time:.2f} seconds. Final prediction: {final_prediction:.3f}")

        return {
            "final_prediction": final_prediction,
            "discussion_history": discussion_history,
            "processing_time": processing_time
        }


###############################################################################
# Process a Single EHR Item with the Reconcile Framework
###############################################################################
def process_item(item: Dict[str, Any],
               agent_configs: List[Dict[str, str]],
               max_rounds: int = 3) -> Dict[str, Any]:
    """
    Process a single EHR item with the Reconcile framework.

    Args:
        item: Input EHR item dictionary (with qid, question, etc.)
        agent_configs: List of agent configurations (each with agent_id and model_key)
        max_rounds: Maximum number of discussion rounds

    Returns:
        Processed EHR result with the final predicted probability and discussion history
    """
    qid = item.get("qid", "unknown")
    question = item.get("question", "")
    ground_truth = item.get("answer")

    print(f"Processing EHR item {qid}")

    # Create coordinator and run discussion
    coordinator = ReconcileCoordinator(agent_configs, max_rounds)
    discussion_result = coordinator.run_discussion(question)

    # Compile results
    result = {
        "qid": qid,
        "timestamp": int(time.time()),
        "question": question,
        "ground_truth": ground_truth,
        "predicted_value": discussion_result["final_prediction"],
        "case_history": discussion_result,
    }

    return result


###############################################################################
# Main Entry Point for the Reconcile Framework on EHR data
###############################################################################
def main():
    """
    Main entry point for running the Reconcile framework on EHR datasets.
    """
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on EHR predictive modeling tasks")
    parser.add_argument("--dataset", type=str, choices=["mimic-iv", "tjh"], required=True, help="Dataset name")
    parser.add_argument("--task", type=str, choices=["mortality", "readmission"], required=True, help="Prediction task")
    parser.add_argument("--agents", nargs='+', default=["qwen-max-latest", "deepseek-v3-official", "qwen-vl-max"],
                       help="List of agent model keys (e.g., deepseek-v3-official, qwen-max-latest, qwen-vl-max)")
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum number of discussion rounds")

    args = parser.parse_args()
    method = "ReConcile"

    # Extract dataset name and task
    dataset_name = args.dataset
    task_name = args.task
    print(f"Dataset: {dataset_name}, Task: {task_name}")

    # Create logs directory structure
    logs_dir = os.path.join("logs", "ehr", dataset_name, task_name, method)
    os.makedirs(logs_dir, exist_ok=True)

    # Construct the data path
    data_path = os.path.join("my_datasets", "processed", "ehr", dataset_name, f"ehr_timeseries_{task_name}_test.json")

    # Load the dataset
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Configure agents: each agent is assigned an ID and a model key
    agent_configs = []
    for idx, model_key in enumerate(args.agents, 1):
        agent_configs.append({"agent_id": f"agent_{idx}", "model_key": model_key})

    print(f"Configured {len(agent_configs)} agents: {[cfg['model_key'] for cfg in agent_configs]}")

    # Process each item in the dataset
    for item in tqdm(data, desc=f"Processing {dataset_name} ({task_name})"):
        qid = item.get("qid")
        result_path = os.path.join(logs_dir, f"ehr_timeseries_{qid}-result.json")

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