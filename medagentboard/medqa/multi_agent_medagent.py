"""
medagentboard/medqa/multi_agent_medagent.py
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
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    PEDIATRICS = "Pediatrics"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    NEONATOLOGY = "Neonatology"
    GENETICS = "Genetics"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    DECISION_MAKER = "Decision Maker"
    EXPERT_GATHERER = "Expert Gatherer"


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
            agent_type: Type of agent (Doctor, Coordinator, or Decision Maker)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key

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
                print(f"Agent {self.agent_id} calling LLM, attempt {retries+1}/{max_retries}")
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


class ExpertGathererAgent(BaseAgent):
    """Agent responsible for gathering domain experts based on medical questions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max"):
        """
        Initialize the expert gatherer agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.EXPERT_GATHERER, model_key)
        print(f"Initializing expert gatherer agent, ID: {agent_id}, Model: {model_key}")

    def gather_question_domain_experts(self, question: str) -> List[MedicalSpecialty]:
        """
        Gather relevant domain experts for a medical question.

        Args:
            question: Medical question to analyze

        Returns:
            List of MedicalSpecialty enums representing relevant experts
        """
        print(f"Expert gatherer {self.agent_id} gathering question domain experts")

        # Prepare system message based on paper's prompt design
        system_message = {
            "role": "system",
            "content": "You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine. "
                      "You need to complete the following steps: "
                      "1. Carefully read the medical scenario presented in the question. "
                      "2. Based on the medical scenario in it, classify the question into three different subfields of medicine. "
                      "3. You should output in JSON format with a 'fields' array containing the medical specialties."
        }

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Review this medical question and determine the three most appropriate medical specialties required to provide the answer. Ensure the selected specialties are distinct and cover different aspects of the problem:\n\n{question}"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            specialties = result.get("fields", [])

            # Convert string specialties to MedicalSpecialty enum
            valid_specialties = []
            for spec in specialties:
                try:
                    # Handle variations in specialty names
                    spec_lower = spec.lower().strip()
                    if "pediatr" in spec_lower:
                        valid_specialties.append(MedicalSpecialty.PEDIATRICS)
                    elif "cardio" in spec_lower:
                        valid_specialties.append(MedicalSpecialty.CARDIOLOGY)
                    elif "pulmon" in spec_lower:
                        valid_specialties.append(MedicalSpecialty.PULMONOLOGY)
                    elif "neonat" in spec_lower:
                        valid_specialties.append(MedicalSpecialty.NEONATOLOGY)
                    elif "genet" in spec_lower:
                        valid_specialties.append(MedicalSpecialty.GENETICS)
                except ValueError:
                    print(f"Warning: Could not map specialty '{spec}' to known MedicalSpecialty")

            # Ensure we have exactly 3 specialties as per paper
            if len(valid_specialties) < 3:
                # Fill with default specialties if needed
                default_specialties = [
                    MedicalSpecialty.PEDIATRICS,
                    MedicalSpecialty.CARDIOLOGY,
                    MedicalSpecialty.PULMONOLOGY,
                    MedicalSpecialty.NEONATOLOGY,
                    MedicalSpecialty.GENETICS
                ]
                valid_specialties = valid_specialties + default_specialties[len(valid_specialties):3]

            return valid_specialties[:3]  # Return exactly 3 specialties

        except json.JSONDecodeError:
            print("Expert gatherer response is not valid JSON, using fallback specialties")
            # Return default specialties if parsing fails
            return [
                MedicalSpecialty.CARDIOLOGY,
                MedicalSpecialty.PULMONOLOGY,
                MedicalSpecialty.NEONATOLOGY,
            ]


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
                f" For multiple choice questions, ensure your 'answer' field contains only the option letter (A, B, C, etc.) "
                f"that best matches your conclusion. Be specific and concise about which option you are selecting."
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
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            return result

    def review_synthesis(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis.

        Args:
            synthesis: Meta agent's synthesis

        Returns:
            Dictionary containing agreement status and reason
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")

        # Prepare system message for review
        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}, participating in a multidisciplinary team consultation. "
                      f"Review the synthesis report and determine if you agree with it. "
                      f"Your output should be in JSON format, including 'agree' (yes/no), 'reason' (rationale for your decision), "
                      f"and 'answer' (your suggested answer if you disagree) fields."
        }

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Synthesized report:\n{synthesis.get('explanation', '')}\n\n"
                      f"Do you agree with this synthesized report? Provide your response in JSON format with the following fields:\n"
                      f"1. 'agree': 'yes' or 'no'\n"
                      f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                      f"3. 'answer': If you disagree, provide your suggested answer"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

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

            return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max"):
        """
        Initialize a meta agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use (defaults to text-only model since meta agent doesn't analyze images)
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                           doctor_opinions: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_round: int = 1) -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions without providing an answer.

        Args:
            doctor_opinions: List of doctor opinions
            doctor_specialties: List of corresponding doctor specialties
            current_round: Current discussion round

        Returns:
            Dictionary containing synthesized explanation only
        """
        print(f"Meta agent synthesizing round {current_round} opinions with model: {self.model_key}")

        # Prepare system message for synthesis
        system_message = {
            "role": "system",
            "content": f"You are a medical consensus coordinator facilitating round {current_round} of a multidisciplinary team consultation. "
                      "Your task is to synthesize the different medical opinions into a coherent summary report WITHOUT providing a final answer or conclusion. "
                      "Consider each doctor's expertise and perspective and create a comprehensive synthesis of their reasoning. "
                      "Your output should be in JSON format with ONLY an 'explanation' field that summarizes all perspectives in a balanced way."
        }

        # Format doctors' opinions as input
        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)

        opinions_text = "\n".join(formatted_opinions)

        # Prepare user message with all opinions
        user_message = {
            "role": "user",
            "content": f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
                      f"Please synthesize these opinions into a coherent summary report. Do NOT provide a final answer or conclusion. "
                      f"Provide your synthesis in JSON format with ONLY an 'explanation' field that summarizes the medical perspectives."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")

            # Ensure there's no answer in the result
            if "answer" in result:
                del result["answer"]

            # Ensure there's an explanation field
            if "explanation" not in result:
                result["explanation"] = "No explanation provided in the synthesis."

            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Meta agent synthesis is not valid JSON, using fallback parsing")

            # Extract all text as explanation
            result = {"explanation": response_text.strip()}

            return result


class DecisionMakingAgent(BaseAgent):
    """Decision making agent that gives final answers based on synthesized opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a decision making agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.DECISION_MAKER, model_key)
        print(f"Initializing decision making agent, ID: {agent_id}, Model: {model_key}")

    def make_decision(self,
                     question: str,
                     synthesis: Dict[str, Any],
                     options: Optional[Dict[str, str]] = None,
                     image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a final decision based on synthesized report and doctor reviews.

        Args:
            question: Original question
            synthesis: Meta agent's synthesis
            options: Optional multiple choice options
            image_path: Optional path to medical image

        Returns:
            Dictionary containing final explanation and answer
        """
        print(f"Decision making agent generating final answer")

        # Prepare system message
        system_message = {
            "role": "system",
            "content": "You are a medical decision making agent responsible for providing the final answer based on expert opinions. "
        }

        system_message["content"] += (
            "Your task is to examine the doctors' opinions and the synthesized report to determine the best final answer. "
            "Your output should be in JSON format, including 'explanation' (final reasoning) and "
            "'answer' (final conclusion) fields."
        )

        # For multiple choice questions, instruct to choose one option
        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains ONLY the option letter (A, B, C, etc.) "
                "that represents the final decision. Be specific and concise."
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


        # Prepare synthesis text
        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}"

        # Add text content to user message
        user_content.append({
            "type": "text",
            "text": f"Question: {question_with_options}\n\n"
                  f"Synthesized Report:\n{synthesis_text}\n\n"
                  f"Please provide your final answer in JSON format, including 'explanation' and 'answer' fields."
        })

        user_message = {
            "role": "user",
            "content": user_content,
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Decision making agent response successfully parsed")
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Decision making agent response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            return result


class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                max_rounds: int = 3,
                model_key: str = "qwen-max-latest",
                meta_model_key: str = "qwen-max-latest",
                decision_model_key: str = "qwen-max-latest"):
        """
        Initialize MDT consultation.

        Args:
            max_rounds: Maximum number of discussion rounds
            model_key: LLM model for doctor agents
            meta_model_key: LLM model for meta agent
            decision_model_key: LLM model for decision making agent
        """
        self.max_rounds = max_rounds
        self.model_key = model_key
        self.meta_model_key = meta_model_key
        self.decision_model_key = decision_model_key

        # Initialize expert gatherer agent
        self.expert_gatherer = ExpertGathererAgent("expert_gatherer", model_key)

        # Initialize other agents (doctors will be initialized dynamically)
        self.doctor_agents = []
        self.doctor_specialties = []

        # Initialize meta agent
        self.meta_agent = MetaAgent("meta", meta_model_key)

        # Initialize decision making agent
        self.decision_agent = DecisionMakingAgent("decision", decision_model_key)

        print(f"Initialized MDT consultation, max_rounds={max_rounds}, model={model_key}")

    def _initialize_doctor_agents(self, specialties: List[MedicalSpecialty]):
        """Initialize doctor agents with the given specialties."""
        self.doctor_agents = []
        for idx, specialty in enumerate(specialties, 1):
            agent_id = f"doctor_{idx}"
            doctor_agent = DoctorAgent(agent_id, specialty, self.model_key)
            self.doctor_agents.append(doctor_agent)
        self.doctor_specialties = specialties

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
        start_time = time.time()

        print(f"Starting MDT consultation for case {qid}")
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")

        # Step 1: Gather relevant domain experts for this question
        specialties = self.expert_gatherer.gather_question_domain_experts(question)
        print(f"Gathered specialties for question: {[s.value for s in specialties]}")

        # Initialize doctor agents with these specialties
        self._initialize_doctor_agents(specialties)

        # Case consultation history
        case_history = {
            "selected_specialties": [s.value for s in specialties],
            "rounds": []
        }

        current_round = 0
        final_decision = None
        consensus_reached = False

        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}

            # Step 2: Each doctor analyzes the case
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

            # Step 3: Meta agent synthesizes opinions without providing an answer
            print("Meta agent synthesizing opinions")
            synthesis = self.meta_agent.synthesize_opinions(
                doctor_opinions, self.doctor_specialties, current_round
            )
            round_data["synthesis"] = synthesis

            print("Meta agent synthesis created")

            # Step 4: Doctors review synthesis
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review = doctor.review_synthesis(synthesis)
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

            # If all doctors agree or this is the last round, make final decision
            if all_agree or current_round == self.max_rounds:
                # Step 5: Decision making agent provides final answer
                print("Decision making agent generating final answer")
                final_decision = self.decision_agent.make_decision(question, synthesis, options, image_path)

                consensus_reached = all_agree

                if all_agree:
                    print("Consensus reached")
                else:
                    print("Max rounds reached without consensus")

                break

            print("No consensus reached, continuing to next round")

        print(f"Final answer: {final_decision.get('answer', '')}")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Add final decision to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time

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
        parsed = json.loads(preprocess_response_string(response_text))
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

        # Ensure explanation field exists
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"

        # Ensure answer field exists if not MetaAgent synthesis
        if "synthesized" not in response_text.lower() and "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result


def process_input(item, model_key="qwen-vl-max", meta_model_key="qwen-max-latest", decision_model_key="qwen-max-latest"):
    """
    Process input data based on its structure.

    Args:
        item: Input data dictionary with question, options, etc.
        model_key: Model key for the doctor agents
        meta_model_key: Model key for the meta agent
        decision_model_key: Model key for the decision making agent

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
        model_key=model_key,
        meta_model_key=meta_model_key,
        decision_model_key=decision_model_key
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
    parser.add_argument("--model", type=str, default="qwen-vl-max",
                       help="Model used for doctor agents")
    parser.add_argument("--meta_model", type=str, default="deepseek-v3-ark",
                       help="Model used for meta agent")
    parser.add_argument("--decision_model", type=str, default="qwen-vl-max",
                       help="Model used for decision making agent")
    args = parser.parse_args()

    method = "MedAgent"

    # Extract dataset name
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

    # Process each item
    for item in tqdm(data, desc=f"Running MDT consultation on {dataset_name}"):
        qid = item["qid"]

        # Skip if already processed
        if os.path.exists(os.path.join(logs_dir, f"{qid}-result.json")):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            # Process the item
            result = process_input(
                item,
                model_key=args.model,
                meta_model_key=args.meta_model,
                decision_model_key=args.decision_model
            )

            # Add output to the original item and save
            item_result = {
                "qid": qid,
                "timestamp": int(time.time()),
                "question": item["question"],
                "options": item.get("options"),
                "image_path": item.get("image_path"),
                "ground_truth": item.get("answer"),
                "predicted_answer": result["final_decision"]["answer"],
                "case_history": result
            }

            # Save individual result
            save_json(item_result, os.path.join(logs_dir, f"{qid}-result.json"))

        except Exception as e:
            print(f"Error processing item {qid}: {e}")


if __name__ == "__main__":
    main()