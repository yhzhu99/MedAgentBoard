"""
multi_agent_ehr_predictor.py - Multi-agent framework for EHR predictive modeling tasks
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
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


class MedicalSpecialty(Enum):
    """Medical specialty enumeration for EHR analysis."""
    CRITICAL_CARE = "Critical Care Medicine"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    INFECTIOUS_DISEASE = "Infectious Disease"
    NEPHROLOGY = "Nephrology"
    HEMATOLOGY = "Hematology"
    ENDOCRINOLOGY = "Endocrinology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    DECISION_MAKER = "Decision Maker"
    EXPERT_GATHERER = "Expert Gatherer"


class BaseAgent:
    """Base class for all agents in the EHR prediction framework."""

    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,
                 model_key: str = "deepseek-v3-official"):
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
            user_message: User message containing EHR data
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
    """Agent responsible for gathering domain experts based on EHR data."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        """
        Initialize the expert gatherer agent for EHR data analysis.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.EXPERT_GATHERER, model_key)
        print(f"Initializing EHR expert gatherer agent, ID: {agent_id}, Model: {model_key}")

    def gather_ehr_domain_experts(self, question: str, task_type: str) -> List[MedicalSpecialty]:
        """
        Gather relevant domain experts for EHR time-series analysis.

        Args:
            question: EHR data and question prompt
            task_type: Type of prediction task (mortality or readmission)

        Returns:
            List of MedicalSpecialty enums representing relevant experts
        """
        print(f"Expert gatherer {self.agent_id} identifying specialists for EHR {task_type} prediction")

        # Prepare system message for EHR expert gathering
        system_message = {
            "role": "system",
            "content": "You are a medical coordinator who specializes in determining which medical specialists are most appropriate "
                      "for analyzing time-series Electronic Health Record (EHR) data for clinical prediction tasks. "
                      "You need to complete the following steps: "
                      "1. Carefully analyze the clinical features and time-series data presented in the EHR. "
                      f"2. Based on the available data, determine which three medical specialties would be most relevant for a {task_type} prediction task. "
                      "3. You should output in JSON format with a 'fields' array containing the three most appropriate medical specialties."
        }

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Review this EHR data and determine the three most appropriate medical specialties needed to analyze the time-series patterns for {task_type} prediction:\n\n{question}"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            specialties = result.get("fields", [])

            # Map to our specialty enum
            valid_specialties = []
            for spec in specialties:
                spec_lower = spec.lower().strip()
                if "critical" in spec_lower or "intensive" in spec_lower or "icu" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.CRITICAL_CARE)
                elif "cardio" in spec_lower or "heart" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.CARDIOLOGY)
                elif "pulmon" in spec_lower or "respiratory" in spec_lower or "lung" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.PULMONOLOGY)
                elif "infect" in spec_lower or "microbi" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.INFECTIOUS_DISEASE)
                elif "nephro" in spec_lower or "kidney" in spec_lower or "renal" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.NEPHROLOGY)
                elif "hemat" in spec_lower or "blood" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.HEMATOLOGY)
                elif "endocrin" in spec_lower or "metabol" in spec_lower or "diabet" in spec_lower:
                    valid_specialties.append(MedicalSpecialty.ENDOCRINOLOGY)
                else:
                    # If specialty doesn't match, default to critical care for EHR analysis
                    valid_specialties.append(MedicalSpecialty.CRITICAL_CARE)

            # Remove duplicates while preserving order
            seen_specialties = set()
            valid_specialties = [s for s in valid_specialties if not (s in seen_specialties or seen_specialties.add(s))]

            # Ensure we have exactly 3 specialties
            if len(valid_specialties) < 3:
                # Fill with default specialties
                default_specialties = [
                    MedicalSpecialty.CRITICAL_CARE,
                    MedicalSpecialty.CARDIOLOGY,
                    MedicalSpecialty.PULMONOLOGY
                ]
                for specialty in default_specialties:
                    if specialty not in valid_specialties and len(valid_specialties) < 3:
                        valid_specialties.append(specialty)

            return valid_specialties[:3]  # Return exactly 3 specialties

        except json.JSONDecodeError:
            print("Expert gatherer response is not valid JSON, using default specialties")
            # Return default specialties for EHR analysis
            return [
                MedicalSpecialty.CRITICAL_CARE,
                MedicalSpecialty.CARDIOLOGY,
                MedicalSpecialty.PULMONOLOGY,
            ]


class DoctorAgent(BaseAgent):
    """Doctor agent specialized in analyzing EHR time-series data."""

    def __init__(self,
                 agent_id: str,
                 specialty: MedicalSpecialty,
                 model_key: str = "deepseek-v3-official"):
        """
        Initialize a doctor agent for EHR analysis.

        Args:
            agent_id: Unique identifier for the doctor
            specialty: Doctor's medical specialty
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent for EHR analysis, ID: {agent_id}, Model: {model_key}")

    def analyze_ehr(self, question: str, task_type: str) -> Dict[str, Any]:
        """
        Analyze EHR time-series data for clinical prediction.

        Args:
            question: EHR data formatted as a question
            task_type: Type of prediction task (mortality or readmission)

        Returns:
            Dictionary containing analysis results and probability prediction
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing EHR data with model: {self.model_key}")

        # Prepare system message specific to the doctor's specialty and task type
        system_message = {
            "role": "system",
            "content": f"You are a specialist in {self.specialty.value} with expertise in analyzing time-series Electronic Health Record (EHR) data. "
                      f"Your task is to analyze the provided clinical time-series data and predict the probability of patient {task_type}. "
                      f"Focus on patterns and trends in the data that are relevant to your {self.specialty.value} expertise. "
                      f"Your output should be in JSON format, including: "
                      f"1. 'explanation': Detailed analysis of the EHR data from your specialty perspective, focusing on time-series patterns. "
                      f"2. 'answer': A probability value between 0 and 1 representing the likelihood of {task_type} (higher value means higher probability)."
        }

        # Prepare user message for EHR analysis
        user_message = {
            "role": "user",
            "content": f"{question}\n\nAs a {self.specialty.value} specialist, analyze the time-series EHR data and provide your prediction in JSON format with 'explanation' and 'answer' fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")

            # Ensure answer is a float between 0 and 1
            if 'answer' in result:
                try:
                    # Convert answer to float and ensure it's between 0 and 1
                    answer_value = float(result['answer'])
                    result['answer'] = max(0.0, min(1.0, answer_value))
                except (ValueError, TypeError):
                    # If conversion fails, default to 0.5
                    print(f"Doctor {self.agent_id} provided invalid answer format, defaulting to 0.5")
                    result['answer'] = 0.5
            else:
                result['answer'] = 0.5

            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

            # Ensure answer is a float between 0 and 1
            if 'answer' in result:
                try:
                    # Convert answer to float and ensure it's between 0 and 1
                    answer_value = float(result['answer'])
                    result['answer'] = max(0.0, min(1.0, answer_value))
                except (ValueError, TypeError):
                    # If conversion fails, default to 0.5
                    result['answer'] = 0.5
            else:
                result['answer'] = 0.5

            return result

    def review_synthesis(self, synthesis: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis of EHR analysis.

        Args:
            synthesis: Meta agent's synthesis
            task_type: Type of prediction task (mortality or readmission)

        Returns:
            Dictionary containing agreement status and reason
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing EHR synthesis with model: {self.model_key}")

        # Prepare system message for review
        system_message = {
            "role": "system",
            "content": f"You are a {self.specialty.value} specialist participating in a multidisciplinary team analysis of EHR data. "
                      f"Review the synthesized analysis of the time-series data and determine if you agree with it. "
                      f"Your output should be in JSON format, including: "
                      f"1. 'agree': A boolean value (true/false) indicating whether you agree with the synthesis. "
                      f"2. 'reason': Your rationale for agreeing or disagreeing, based on your specialty expertise. "
                      f"3. 'answer': If you disagree, provide your own probability value between 0 and 1 for the {task_type} prediction."
        }

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Synthesized EHR analysis:\n{synthesis.get('explanation', '')}\n\n"
                      f"Do you agree with this synthesized analysis for {task_type} prediction? Provide your response in JSON format."
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

            # Ensure answer is a float between 0 and 1 if present
            if 'answer' in result:
                try:
                    answer_value = float(result['answer'])
                    result['answer'] = max(0.0, min(1.0, answer_value))
                except (ValueError, TypeError):
                    # If conversion fails, remove the answer field
                    del result['answer']

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
                    try:
                        answer_text = line.split(":", 1)[1].strip() if ":" in line else line
                        answer_value = float(answer_text)
                        result["answer"] = max(0.0, min(1.0, answer_value))
                    except (ValueError, TypeError):
                        # If conversion fails, don't add answer field
                        pass

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"

            return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple specialists' EHR analyses."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        """
        Initialize a meta agent for synthesizing EHR analyses.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent for EHR synthesis, ID: {agent_id}, Model: {model_key}")

    def synthesize_ehr_analyses(self,
                              doctor_opinions: List[Dict[str, Any]],
                              doctor_specialties: List[MedicalSpecialty],
                              task_type: str,
                              current_round: int = 1) -> Dict[str, Any]:
        """
        Synthesize multiple specialists' analyses of EHR data.

        Args:
            doctor_opinions: List of doctor opinions on the EHR data
            doctor_specialties: List of corresponding doctor specialties
            task_type: Type of prediction task (mortality or readmission)
            current_round: Current discussion round

        Returns:
            Dictionary containing synthesized explanation only
        """
        print(f"Meta agent synthesizing round {current_round} EHR analyses with model: {self.model_key}")

        # Prepare system message for EHR synthesis
        system_message = {
            "role": "system",
            "content": f"You are a clinical coordinator synthesizing multiple specialists' analyses of time-series EHR data for round {current_round} of a multidisciplinary team consultation. "
                      f"Your task is to create a comprehensive synthesis of their interpretations of the clinical time-series patterns WITHOUT providing a final probability prediction. "
                      f"Consider each specialist's expertise in analyzing different aspects of the EHR data for {task_type} prediction. "
                      f"Your output should be in JSON format with ONLY an 'explanation' field that summarizes all perspectives in a balanced way."
        }

        # Format doctors' opinions as input
        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Specialist {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Probability: {opinion.get('answer', 'Not provided')}\n"
            formatted_opinions.append(formatted_opinion)

        opinions_text = "\n".join(formatted_opinions)

        # Prepare user message with all opinions
        user_message = {
            "role": "user",
            "content": f"Round {current_round} Specialists' EHR Analyses for {task_type} prediction:\n{opinions_text}\n\n"
                      f"Please synthesize these analyses into a coherent summary that integrates the time-series interpretations. Do NOT provide a final probability prediction. "
                      f"Provide your synthesis in JSON format with ONLY an 'explanation' field."
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
    """Decision making agent that outputs final probability prediction for EHR data."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        """
        Initialize a decision making agent for EHR prediction.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.DECISION_MAKER, model_key)
        print(f"Initializing decision making agent for EHR prediction, ID: {agent_id}, Model: {model_key}")

    def make_prediction(self,
                      question: str,
                      synthesis: Dict[str, Any],
                      doctor_opinions: List[Dict[str, Any]],
                      task_type: str) -> Dict[str, Any]:
        """
        Make a final probability prediction based on synthesized EHR analyses.

        Args:
            question: Original EHR data question
            synthesis: Meta agent's synthesis
            doctor_opinions: List of doctor opinions
            task_type: Type of prediction task (mortality or readmission)

        Returns:
            Dictionary containing final explanation and probability prediction
        """
        print(f"Decision making agent generating final {task_type} probability prediction")

        # Prepare system message
        system_message = {
            "role": "system",
            "content": f"You are a clinical decision making agent responsible for providing the final probability prediction for patient {task_type} based on time-series EHR data. "
                      f"Your task is to examine the specialists' analyses and the synthesized report to determine the most accurate probability prediction. "
                      f"Your output should be in JSON format, including: "
                      f"1. 'explanation': Final reasoning that justifies your prediction based on the EHR data patterns. "
                      f"2. 'answer': A floating-point number between 0 and 1 representing the predicted probability of {task_type} (higher value means higher likelihood)."
        }

        # Format doctor opinions for context
        opinions_summary = []
        for i, opinion in enumerate(doctor_opinions):
            probability = opinion.get('answer', 'Not provided')
            opinions_summary.append(f"Specialist {i+1} probability: {probability}")

        opinions_text = "\n".join(opinions_summary)

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"EHR Data Question:\n{question}\n\n"
                      f"Synthesized Analysis:\n{synthesis.get('explanation', '')}\n\n"
                      f"Specialist Probability Predictions:\n{opinions_text}\n\n"
                      f"Please provide your final probability prediction for patient {task_type} in JSON format, including 'explanation' and 'answer' fields. "
                      f"The 'answer' must be a floating-point number between 0 and 1."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Decision making agent response successfully parsed")

            # Ensure answer is a float between 0 and 1
            if 'answer' in result:
                try:
                    answer_value = float(result['answer'])
                    result['answer'] = max(0.0, min(1.0, answer_value))
                except (ValueError, TypeError):
                    # If conversion fails, use average of doctor opinions
                    valid_opinions = [op.get('answer') for op in doctor_opinions if isinstance(op.get('answer'), (int, float))]
                    if valid_opinions:
                        result['answer'] = sum(valid_opinions) / len(valid_opinions)
                    else:
                        result['answer'] = 0.5
            else:
                # If no answer provided, use average of doctor opinions
                valid_opinions = [op.get('answer') for op in doctor_opinions if isinstance(op.get('answer'), (int, float))]
                if valid_opinions:
                    result['answer'] = sum(valid_opinions) / len(valid_opinions)
                else:
                    result['answer'] = 0.5

            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Decision making agent response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

            # Ensure answer is a float between 0 and 1
            if 'answer' in result:
                try:
                    answer_value = float(result['answer'])
                    result['answer'] = max(0.0, min(1.0, answer_value))
                except (ValueError, TypeError):
                    # If conversion fails, use average of doctor opinions
                    valid_opinions = [op.get('answer') for op in doctor_opinions if isinstance(op.get('answer'), (int, float))]
                    if valid_opinions:
                        result['answer'] = sum(valid_opinions) / len(valid_opinions)
                    else:
                        result['answer'] = 0.5
            else:
                # If no answer provided, use average of doctor opinions
                valid_opinions = [op.get('answer') for op in doctor_opinions if isinstance(op.get('answer'), (int, float))]
                if valid_opinions:
                    result['answer'] = sum(valid_opinions) / len(valid_opinions)
                else:
                    result['answer'] = 0.5

            return result


class MDTConsultation:
    """Multi-disciplinary team consultation for EHR time-series prediction."""

    def __init__(self,
                max_rounds: int = 2,
                model_key: str = "deepseek-v3-official",
                meta_model_key: str = "deepseek-v3-official",
                decision_model_key: str = "deepseek-v3-official"):
        """
        Initialize MDT consultation for EHR prediction.

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

        print(f"Initialized MDT consultation for EHR prediction, max_rounds={max_rounds}, model={model_key}")

    def _initialize_doctor_agents(self, specialties: List[MedicalSpecialty]):
        """Initialize doctor agents with the given specialties."""
        self.doctor_agents = []
        for idx, specialty in enumerate(specialties, 1):
            agent_id = f"doctor_{idx}"
            doctor_agent = DoctorAgent(agent_id, specialty, self.model_key)
            self.doctor_agents.append(doctor_agent)
        self.doctor_specialties = specialties

    def run_consultation(self, qid: str, question: str, task_type: str) -> Dict[str, Any]:
        """
        Run the MDT consultation process for EHR prediction.

        Args:
            qid: Question ID
            question: EHR data formatted as a question
            task_type: Type of prediction task (mortality or readmission)

        Returns:
            Dictionary containing final consultation result with probability prediction
        """
        start_time = time.time()

        print(f"Starting MDT consultation for EHR case {qid}, task: {task_type}")
        print(f"EHR Question length: {len(question)} characters")

        # Step 1: Gather relevant domain experts for this EHR data
        specialties = self.expert_gatherer.gather_ehr_domain_experts(question, task_type)
        print(f"Gathered specialties for EHR analysis: {[s.value for s in specialties]}")

        # Initialize doctor agents with these specialties
        self._initialize_doctor_agents(specialties)

        # Case consultation history
        case_history = {
            "qid": qid,
            "task_type": task_type,
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

            # Step 2: Each doctor analyzes the EHR data
            doctor_opinions = []
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing EHR data")
                opinion = doctor.analyze_ehr(question, task_type)
                doctor_opinions.append(opinion)
                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "opinion": opinion
                })

                print(f"Doctor {i+1} probability prediction: {opinion.get('answer', 'Not provided')}")

            # Step 3: Meta agent synthesizes EHR analyses without providing a probability
            print("Meta agent synthesizing EHR analyses")
            synthesis = self.meta_agent.synthesize_ehr_analyses(
                doctor_opinions, self.doctor_specialties, task_type, current_round
            )
            round_data["synthesis"] = synthesis

            print("Meta agent synthesis created")

            # Step 4: Doctors review synthesis
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review = doctor.review_synthesis(synthesis, task_type)
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
                # Step 5: Decision making agent provides final probability prediction
                print("Decision making agent generating final probability prediction")
                final_decision = self.decision_agent.make_prediction(
                    question, synthesis, doctor_opinions, task_type
                )

                consensus_reached = all_agree

                if all_agree:
                    print("Consensus reached")
                else:
                    print("Max rounds reached without consensus")

                break

            print("No consensus reached, continuing to next round")

        print(f"Final probability prediction: {final_decision.get('answer', 'Not provided')}")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Add final decision to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time

        return case_history


def parse_structured_output(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract structured output for EHR prediction.

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

                # For answer field, try to convert to float
                if key == "answer" or key == "probability":
                    try:
                        float_value = float(value)
                        result["answer"] = max(0.0, min(1.0, float_value))
                    except ValueError:
                        # If conversion fails, don't add the field
                        pass
                else:
                    result[key] = value

        # Ensure explanation field exists
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"

        # Ensure answer field exists with default value if not found
        if "answer" not in result:
            result["answer"] = 0.5  # Default to 50% probability if not found

        return result


def detect_task_type(question: str) -> str:
    """
    Detect whether the EHR prediction task is for mortality or readmission.

    Args:
        question: EHR question text

    Returns:
        Task type string ('mortality' or 'readmission')
    """
    question_lower = question.lower()
    if "readmission" in question_lower or "re-admission" in question_lower:
        return "readmission"
    else:
        return "mortality"


def process_ehr_item(item, model_key="deepseek-v3-official", meta_model_key="deepseek-v3-official", decision_model_key="deepseek-v3-official"):
    """
    Process EHR time-series data for prediction.

    Args:
        item: Input data dictionary with qid, question, and answer
        model_key: Model key for the doctor agents
        meta_model_key: Model key for the meta agent
        decision_model_key: Model key for the decision making agent

    Returns:
        Processed result from MDT consultation
    """
    # Required fields
    qid = str(item.get("qid"))
    question = item.get("question")

    # Detect task type from question content
    task_type = detect_task_type(question)
    print(f"Detected task type: {task_type}")

    # Initialize consultation
    mdt = MDTConsultation(
        max_rounds=2,  # Reduced to 2 rounds to optimize processing time for EHR data
        model_key=model_key,
        meta_model_key=meta_model_key,
        decision_model_key=decision_model_key
    )

    # Run consultation
    result = mdt.run_consultation(
        qid=qid,
        question=question,
        task_type=task_type
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent EHR predictions")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (mimic-iv or tjh)")
    parser.add_argument("--task", type=str, required=True, choices=["mortality", "readmission"],
                        help="Prediction task type")
    parser.add_argument("--model", type=str, default="deepseek-v3-official",
                        help="Model used for doctor agents")
    parser.add_argument("--meta_model", type=str, default="deepseek-v3-official",
                        help="Model used for meta agent")
    parser.add_argument("--decision_model", type=str, default="deepseek-v3-official",
                        help="Model used for decision making agent")
    args = parser.parse_args()

    method = "MedAgent"

    # Extract dataset name and task
    dataset_name = args.dataset
    task_name = args.task
    print(f"Dataset: {dataset_name}, Task: {task_name}")

    # Create logs directory structure
    logs_dir = os.path.join("logs", "ehr", dataset_name, task_name, method)
    os.makedirs(logs_dir, exist_ok=True)

    # Set up data path
    data_path = f"./my_datasets/processed/ehr/{dataset_name}/ehr_timeseries_{task_name}_test.json"

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Process each item
    for item in tqdm(data, desc=f"Running EHR predictions on {dataset_name}/{task_name}"):
        qid = item["qid"]
        result_path = os.path.join(logs_dir, f"ehr_timeseries_{qid}-result.json")

        # Skip if already processed
        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            # Process the item
            result = process_ehr_item(
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
                "ground_truth": item.get("answer"),
                "predicted_probability": result["final_decision"]["answer"],
                "case_history": result
            }

            # Save individual result
            save_json(item_result, result_path)
            print(f"Saved result for {qid} to {result_path}")

        except Exception as e:
            print(f"Error processing item {qid}: {e}")


if __name__ == "__main__":
    main()