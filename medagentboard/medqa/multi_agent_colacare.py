"""
medagentboard/medqa/multi_agent_colacare.py

"""

from openai import OpenAI
import os
import json
from enum import Enum
from typing import Dict, Any, Optional, List
import time
import argparse
from tqdm import tqdm

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json


class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"


class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor or Coordinator)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up OpenAI client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

    def call_llm(self,
                system_message: Dict[str, str],
                user_message: Dict[str, Any],
                max_retries: int = 3) -> str:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing question and optional image
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    response_format={"type": "json_object"}
                )
                response = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief pause before retrying


class DoctorAgent(BaseAgent):
    """Doctor agent with a medical specialty."""

    def __init__(self,
                 agent_id: str,
                 specialty: MedicalSpecialty,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize a doctor agent.

        Args:
            agent_id: Unique identifier for the doctor
            specialty: Doctor's medical specialty
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent, ID: {agent_id}, Model: {model_key}")

    def analyze_case(self,
                    question: str,
                    options: Optional[Dict[str, str]] = None,
                    image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a medical case.

        Args:
            question: Question about the case
            options: Optional multiple choice options
            image_path: Optional path to medical image

        Returns:
            Dictionary containing analysis results
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing case with model: {self.model_key}")

        # Prepare system message to guide the doctor's analysis
        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}. "
                      f"Analyze the medical case and provide your professional opinion on the question. "
                      f"Your output should be in JSON format, including 'explanation' (detailed reasoning) and "
                      f"'answer' (clear conclusion) fields."
        }

        # For multiple choice questions, instruct to choose one option
        if options:
            system_message["content"] += (
                f" For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                f"that best matches your conclusion. Be specific about which option you are selecting."
            )

        # Prepare user message content
        user_content = []

        # Add image if provided
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        # If options are provided, format them in the question
        if options:
            options_text = "\nOptions:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        user_content.append({
            "type": "text",
            "text": f"{question_with_options}\n\nProvide your analysis in JSON format, including 'explanation' and 'answer' fields."
        })

        user_message = {
            "role": "user",
            "content": user_content,
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(response_text)
            print(f"Doctor {self.agent_id} response successfully parsed")
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result

    def review_synthesis(self,
                        question: str,
                        synthesis: Dict[str, Any],
                        options: Optional[Dict[str, str]] = None,
                        image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis.

        Args:
            question: Original question
            synthesis: Meta agent's synthesis
            options: Optional multiple choice options
            image_path: Optional path to medical image

        Returns:
            Dictionary containing agreement status and possible rebuttal
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")

        # Get current round
        current_round = len(self.memory) // 2 + 1

        # Get doctor's own most recent analysis
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break

        # Prepare system message for review
        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}, participating in round {current_round} of a multidisciplinary team consultation. "
                      f"Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion. "
                      f"Consider your previous analysis and the MetaAgent's synthesized opinion to decide whether to agree or provide a different perspective. "
                      f"Your output should be in JSON format, including 'agree' (boolean or 'yes'/'no'), 'reason' (rationale for your decision), "
                      f"and 'answer' (your suggested answer if you disagree; if you agree, you can repeat the synthesized answer) fields."
        }

        # Prepare user message content
        user_content = []

        # Add image if provided
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        # If options are provided, format them in the question
        if options:
            options_text = "\nOptions:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        # Prepare text with own previous analysis
        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"

        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested answer: {synthesis.get('answer', '')}"

        user_content.append({
            "type": "text",
            "text": f"Original question: {question_with_options}\n\n"
                  f"{own_analysis_text}"
                  f"{synthesis_text}\n\n"
                  f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                  f"1. 'agree': 'yes'/'no'\n"
                  f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                  f"3. 'answer': Your supported answer (can be the synthesized answer if you agree, or your own suggested answer if you disagree)"
        })

        user_message = {
            "role": "user",
            "content": user_content,
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(response_text)
            print(f"Doctor {self.agent_id} review successfully parsed")

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "answer" not in result:
                # Default to own previous answer or synthesized answer
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "No answer provided")

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a meta agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use (defaults to text-only model since meta agent doesn't analyze images)
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                           question: str,
                           doctor_opinions: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_round: int = 1,
                           options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions.

        Args:
            question: Original question
            doctor_opinions: List of doctor opinions
            doctor_specialties: List of corresponding doctor specialties
            current_round: Current discussion round
            options: Optional multiple choice options

        Returns:
            Dictionary containing synthesized explanation and answer
        """
        print(f"Meta agent synthesizing round {current_round} opinions with model: {self.model_key}")

        # Prepare system message for synthesis
        system_message = {
            "role": "system",
            "content": f"You are a medical consensus coordinator facilitating round {current_round} of a multidisciplinary team consultation. "
                      "Synthesize the opinions of multiple specialist doctors into a coherent analysis and conclusion. "
                      "Consider each doctor's expertise and perspective, and weigh their opinions accordingly. "
                      "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and "
                      "'answer' (consensus conclusion) fields."
        }

        # For multiple choice questions, instruct to choose one option
        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that best represents the consensus view. Be specific about which option you are selecting."
            )

        # Format doctors' opinions as input
        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)

        opinions_text = "\n".join(formatted_opinions)

        # If options are provided, format them in the question
        if options:
            options_text = "\nOptions:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        # Prepare user message with all opinions
        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                      f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
                      f"Please synthesize these opinions into a consensus view. Provide your synthesis in JSON format, including "
                      f"'explanation' (comprehensive reasoning) and 'answer' (clear conclusion) fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(response_text)
            print("Meta agent synthesis successfully parsed")

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result

    def make_final_decision(self,
                           question: str,
                           doctor_reviews: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_synthesis: Dict[str, Any],
                           current_round: int,
                           max_rounds: int,
                           options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a final decision based on doctor reviews.

        Args:
            question: Original question
            doctor_reviews: List of doctor reviews
            doctor_specialties: List of corresponding doctor specialties
            current_synthesis: Current synthesized result
            current_round: Current round
            max_rounds: Maximum number of rounds
            options: Optional multiple choice options

        Returns:
            Dictionary containing final explanation and answer
        """
        print(f"Meta agent making round {current_round} decision with model: {self.model_key}")

        # Check if all doctors agree
        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds

        # Prepare system message for final decision
        system_message = {
            "role": "system",
            "content": "You are a medical consensus coordinator making a final decision. "
        }

        if all_agree:
            system_message["content"] += "All doctors agree with your synthesis, generate a final report."
        elif reached_max_rounds:
            system_message["content"] += (
                f"Maximum number of discussion rounds ({max_rounds}) reached without full consensus. "
                f"Make a final decision using majority opinion approach."
            )
        else:
            system_message["content"] += (
                "Not all doctors agree with your synthesis, but a decision for the current round is needed."
            )

        system_message["content"] += (
            " Your output should be in JSON format, including 'explanation' (final reasoning) and "
            "'answer' (final conclusion) fields."
        )

        # For multiple choice questions, instruct to choose one option
        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that represents the final decision. Be specific about which option you are selecting."
            )

        # Format doctor reviews
        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(doctor_reviews, doctor_specialties)):
            formatted_review = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_review += f"Agree: {'Yes' if review.get('agree', False) else 'No'}\n"
            formatted_review += f"Reason: {review.get('reason', '')}\n"
            formatted_review += f"Answer: {review.get('answer', '')}\n"
            formatted_reviews.append(formatted_review)

        reviews_text = "\n".join(formatted_reviews)

        # If options are provided, format them in the question
        if options:
            options_text = "\nOptions:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        # Prepare current synthesis text
        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested answer: {current_synthesis.get('answer', '')}"
        )

        decision_type = "final" if all_agree or reached_max_rounds else "current round"

        # Review previous rounds' syntheses from memory
        previous_syntheses = []
        for i, mem in enumerate(self.memory):
            if mem["type"] == "synthesis" and mem["round"] < current_round:
                prev = f"Round {mem['round']} synthesis:\n"
                prev += f"Explanation: {mem['content'].get('explanation', '')}\n"
                prev += f"Answer: {mem['content'].get('answer', '')}"
                previous_syntheses.append(prev)

        previous_syntheses_text = "\n\n".join(previous_syntheses) if previous_syntheses else "No previous syntheses available."

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                      f"{current_synthesis_text}\n\n"
                      f"Doctor Reviews:\n{reviews_text}\n\n"
                      f"Previous Rounds:\n{previous_syntheses_text}\n\n"
                      f"Please provide your {decision_type} decision, "
                      f"in JSON format, including 'explanation' and 'answer' fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(response_text)
            print("Meta agent final decision successfully parsed")

            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Meta agent final decision is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result


class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                max_rounds: int = 3,
                doctor_configs: List[Dict] = None,
                meta_model_key: str = "qwen-max-latest"):
        """
        Initialize MDT consultation.

        Args:
            max_rounds: Maximum number of discussion rounds
            doctor_configs: List of dictionaries specifying each doctor's specialty and model_key
            meta_model_key: LLM model for meta agent (can be text-only)
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.SURGERY, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.RADIOLOGY, "model_key": "qwen-vl-max"},
        ]
        self.meta_model_key = meta_model_key

        # Initialize doctor agents with different specialties and models
        self.doctor_agents = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            specialty = config["specialty"]
            model_key = config.get("model_key", "qwen-vl-max")
            doctor_agent = DoctorAgent(agent_id, specialty, model_key)
            self.doctor_agents.append(doctor_agent)

        # Initialize meta agent (using text-only model)
        self.meta_agent = MetaAgent("meta", meta_model_key)

        # Store doctor specialties for easy access
        self.doctor_specialties = [doctor.specialty for doctor in self.doctor_agents]

        # Prepare doctor info for logging
        doctor_info = ", ".join([
            f"{config['specialty'].value} ({config.get('model_key', 'default')})"
            for config in self.doctor_configs
        ])
        print(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")


    def run_consultation(self,
                        qid: str,
                        question: str,
                        options: Optional[Dict[str, str]] = None,
                        image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the MDT consultation process.

        Args:
            qid: Question ID
            question: Question about the case
            options: Optional multiple choice options
            image_path: Optional path to medical image

        Returns:
            Dictionary containing final consultation result
        """

        print(f"Starting MDT consultation for case {qid}")
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")

        # Case consultation history
        case_history = {
            "rounds": []
        }

        current_round = 0
        final_decision = None
        consensus_reached = False

        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}

            # Step 1: Each doctor analyzes the case
            doctor_opinions = []
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                opinion = doctor.analyze_case(question, options, image_path)
                doctor_opinions.append(opinion)
                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "opinion": opinion
                })

                print(f"Doctor {i+1} opinion: {opinion.get('answer', '')}")

            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis = self.meta_agent.synthesize_opinions(
                question, doctor_opinions, self.doctor_specialties,
                current_round, options
            )
            round_data["synthesis"] = synthesis

            print(f"Meta agent synthesis: {synthesis.get('answer', '')}")

            # Step 3: Doctors review synthesis
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review = doctor.review_synthesis(question, synthesis, options, image_path)
                doctor_reviews.append(review)
                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "review": review
                })

                agrees = review.get('agree', False)
                all_agree = all_agree and agrees

                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            # Add round data to history
            case_history["rounds"].append(round_data)

            # Step 4: Meta agent makes decision based on reviews
            decision = self.meta_agent.make_final_decision(
                question, doctor_reviews, self.doctor_specialties,
                synthesis, current_round, self.max_rounds, options
            )

            # Check if consensus reached
            if all_agree:
                consensus_reached = True
                final_decision = decision
                print("Consensus reached")
            else:
                print("No consensus reached, continuing to next round")
                if current_round == self.max_rounds:
                    # If max rounds reached, use the last round's decision as final
                    final_decision = decision

        # If no final decision yet, use the last decision
        if not final_decision:
            final_decision = decision

        print(f"Final decision: {final_decision.get('answer', '')}")

        # Add final decision to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round

        return case_history


def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response to extract structured output.

    Args:
        response_text: Text response from LLM

    Returns:
        Dictionary containing structured fields
    """
    try:
        # Try parsing as JSON
        parsed = json.loads(response_text)
        return parsed
    except json.JSONDecodeError:
        # If not valid JSON, extract from text
        # This is a fallback for when the model doesn't format JSON correctly
        lines = response_text.strip().split('\n')
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                result[key] = value

        # Ensure explanation and answer fields exist
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result


def process_input(item, doctor_configs=None, meta_model_key="qwen-max-latest"):
    """
    Process input data.

    Args:
        item: Input data dictionary with question, options, etc.
        doctor_configs: List of doctor configurations (specialty and model_key)
        meta_model_key: Model key for the meta agent

    Returns:
        Processed result from MDT consultation
    """
    # Required fields
    qid = item.get("qid")
    question = item.get("question")

    # Optional fields
    options = item.get("options")
    image_path = item.get("image_path")

    # Initialize consultation
    mdt = MDTConsultation(
        max_rounds=3,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
    )

    # Run consultation
    result = mdt.run_consultation(
        qid=qid,
        question=question,
        options=options,
        image_path=image_path,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True,
                       help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--meta_model", type=str, default="qwen-max-latest",
                       help="Model used for meta agent")
    parser.add_argument("--doctor_models", nargs='+', default=["qwen-vl-max", "qwen-vl-max", "qwen-vl-max"],
                       help="Models used for doctor agents. Provide one model name per doctor.")
    args = parser.parse_args()

    method = "ColaCare"



    # Extract dataset name from path
    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")

    # Determine format (multiple choice or free-form)
    qa_type = args.qa_type
    print(f"QA Format: {qa_type}")

    # Create logs directory structure
    logs_dir = os.path.join("logs", dataset_name, "multiple_choice" if qa_type == "mc" else "free-form", method)
    os.makedirs(logs_dir, exist_ok=True)

    # Set up data path
    data_path = f"./my_datasets/processed/{dataset_name}/medqa_{qa_type}.json"

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Configure doctor models and roles
    # Allow variable number of doctors based on provided models
    doctor_specialties = [
        MedicalSpecialty.INTERNAL_MEDICINE,
        MedicalSpecialty.SURGERY,
        MedicalSpecialty.RADIOLOGY
    ]

    # Make sure we have enough specialties for all provided models
    if len(args.doctor_models) > len(doctor_specialties):
        print(f"Warning: More doctor models ({len(args.doctor_models)}) provided than specialties ({len(doctor_specialties)}). "
              f"Extra models will not be used.")

    # Create doctor configurations
    doctor_configs = []
    for i, model in enumerate(args.doctor_models[:len(doctor_specialties)]):
        doctor_configs.append({
            "specialty": doctor_specialties[i],
            "model_key": model
        })

    print(f"Configuring {len(doctor_configs)} doctors with models: {args.doctor_models[:len(doctor_configs)]}")

    # Process each item
    for item in tqdm(data, desc=f"Running MDT consultation on {dataset_name}"):
        pid = item["qid"]

        # Skip if already processed
        if os.path.exists(os.path.join(logs_dir, f"{pid}-result.json")):
            print(f"Skipping {pid} - already processed")
            continue

        try:
            # Process the item
            result = process_input(
                item,
                doctor_configs=doctor_configs,
                meta_model_key=args.meta_model
            )

            # Add output to the original item and save
            item_result = {
                "qid": pid,
                "timestamp": int(time.time()),
                "question": item["question"],
                "options": item.get("options"),
                "ground_truth": item.get("answer"),
                "predicted_answer": result["final_decision"]["answer"],
                "case_history": result
            }

            # Save individual result
            save_json(item_result, os.path.join(logs_dir, f"{pid}-result.json"))

        except Exception as e:
            print(f"Error processing item {pid}: {e}")


if __name__ == "__main__":
    main()