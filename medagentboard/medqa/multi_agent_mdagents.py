import os
import time
import argparse
from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm
import json

# Utilities from the ColaCare framework
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string

# --- Constants and Enums ---

class ComplexityLevel(Enum):
    """Enumeration for medical query complexity levels."""
    BASIC = "basic"        # Maps to paper's "Low"
    INTERMEDIATE = "intermediate" # Maps to paper's "Moderate"
    ADVANCED = "advanced"    # Maps to paper's "High"

class AgentRole(Enum):
    """Enumeration for different agent roles in MDAgents."""
    MODERATOR = "Moderator"
    RECRUITER = "Recruiter"
    GENERAL_DOCTOR = "General Doctor"
    SPECIALIST = "Specialist"
    TEAM_LEAD = "Team Lead"
    DECISION_MAKER = "Decision Maker" # For final decision synthesis

# Default settings (can be overridden by arguments)
DEFAULT_MODERATOR_MODEL = "deepseek-v3-ark" # Text model for classification/recruitment
DEFAULT_RECRUITER_MODEL = "deepseek-v3-ark"
DEFAULT_AGENT_MODEL = "qwen-vl-max" # Default multimodal model for analysis agents
DEFAULT_MAX_ROUNDS_INTERMEDIATE = 3 # Max discussion rounds for intermediate
DEFAULT_MAX_TURNS_INTERMEDIATE = 3  # Max turns per round for intermediate
DEFAULT_NUM_EXPERTS_INTERMEDIATE = 5
DEFAULT_NUM_TEAMS_ADVANCED = 3
DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED = 3

# --- Base Agent Class ---

class BaseAgent:
    """Base class for all agents in the MDAgents framework, adapted from ColaCare."""

    def __init__(self,
                 agent_id: str,
                 role: Union[AgentRole, str], # Allow custom roles string
                 model_key: str,
                 instruction: Optional[str] = None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent.
            role: The role of the agent (e.g., Moderator, Specialist).
            model_key: Key for the LLM model configuration in LLM_MODELS_SETTINGS.
            instruction: System-level instruction defining the agent's persona and task.
        """
        self.agent_id = agent_id
        self.role = role if isinstance(role, str) else role.value
        self.model_key = model_key
        self.instruction = instruction or f"You are a helpful assistant playing the role of a {self.role}."
        self.memory = [] # List to store message history for this agent

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up OpenAI compatible client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.llm_client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

        print(f"Initialized Agent: ID={self.agent_id}, Role={self.role}, Model={self.model_key} ({self.model_name})")

    def call_llm(self,
                 messages: List[Dict[str, Any]],
                 response_format: Optional[Dict[str, str]] = None, # e.g., {"type": "json_object"}
                 max_retries: int = 3,
                 temperature: float = 0.7) -> str:
        """
        Call the LLM with a list of messages and handle retries.

        Args:
            messages: List of message dictionaries (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": ...}]).
            response_format: Optional dictionary specifying the desired response format (e.g., JSON).
            max_retries: Maximum number of retry attempts.
            temperature: Sampling temperature for the LLM call.

        Returns:
            LLM response text.

        Raises:
            Exception: If LLM call fails after all retries.
        """

        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM ({self.model_name}). Attempt {retries + 1}/{max_retries}.")

                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format:
                    completion_params["response_format"] = response_format

                completion = self.llm_client.chat.completions.create(**completion_params)

                response = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response successfully.")

                # Add user message and assistant response to this agent's memory
                # (Assuming the last message in 'messages' is the user prompt)
                if messages[-1]['role'] == 'user':
                    self.memory.append(messages[-1])
                    self.memory.append({"role": "assistant", "content": response})

                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error for agent {self.agent_id} (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed for agent {self.agent_id} after {max_retries} attempts: {e}")

                time.sleep(1)

        # Should not be reached if max_retries > 0
        raise Exception(f"LLM call failed unexpectedly for agent {self.agent_id}.")

    def chat(self,
             prompt: str,
             image_path: Optional[str] = None,
             use_memory: bool = True,
             response_format: Optional[Dict[str, str]] = None,
             temperature: float = 0.7) -> str:
        """
        Simplified chat interface for an agent.

        Args:
            prompt: The user's message/query to the agent.
            image_path: Optional path to an image file (for multimodal models).
            use_memory: Whether to include the agent's history in the LLM call.
            response_format: Optional dictionary specifying the desired response format.
            temperature: Sampling temperature.

        Returns:
            The assistant's response text.
        """
        system_message = {"role": "system", "content": self.instruction}

        # Prepare user message content (text + optional image)
        user_content: Union[str, List[Dict[str, Any]]]
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

            base64_image = encode_image(image_path)
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        else:
            user_content = prompt

        user_message = {"role": "user", "content": user_content}

        # Construct messages list
        messages = [system_message]
        if use_memory and self.memory:
            messages.extend(self.memory) # Add past interactions
        messages.append(user_message) # Add the current prompt

        # Call LLM
        response = self.call_llm(messages, response_format=response_format, temperature=temperature)

        return response

    def clear_memory(self):
        """Clears the agent's conversation memory."""
        self.memory = []
        print(f"Cleared memory for agent {self.agent_id}")

# --- MDAgents Group Class ---

class Group:
    """Represents a team of agents working towards a common goal."""

    def __init__(self,
                 group_id: str,
                 goal: str,
                 members: List[BaseAgent],
                 question_context: Dict[str, Any]):
        """
        Initialize a group of agents.

        Args:
            group_id: Unique identifier for the group.
            goal: The objective or purpose of this group.
            members: A list of BaseAgent instances that are part of this group.
            question_context: Dictionary containing 'question', optional 'options', 'image_path'.
        """
        self.group_id = group_id
        self.goal = goal
        self.members = members
        self.question_context = question_context
        self.internal_log = [] # Log interactions within the group

        print(f"Initialized Group: ID={self.group_id}, Goal='{self.goal}', Members={[m.agent_id for m in self.members]}")

        # Identify lead agent
        self.lead_agent = None
        # Look for 'lead' in role/id if specified during recruitment
        for member in members:
            if 'lead' in member.role.lower() or 'lead' in member.agent_id.lower():
                self.lead_agent = member
                break

        # Default to first member if no lead found
        if not self.lead_agent and members:
            print(f"Warning: No explicit lead found in group {self.group_id}. Assigning first member '{members[0].agent_id}' as lead.")
            self.lead_agent = members[0]
        elif not members:
            print(f"Warning: Group {self.group_id} created with no members.")

    def _log_interaction(self, message: str):
        """Adds a message to the internal group log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(f"[Group {self.group_id} Log] {message}")
        self.internal_log.append(log_entry)

    def perform_internal_discussion(self) -> str:
        """
        Simulates the internal discussion process within the group to achieve its goal.

        Returns:
            A string representing the group's synthesized report or conclusion.
        """
        if not self.members:
            self._log_interaction("No members in the group to perform discussion.")
            return "Error: Group has no members."

        if not self.lead_agent:
            self._log_interaction("No lead agent identified for coordination.")
            return "Error: Group has no lead agent."

        self._log_interaction(f"Starting internal discussion. Lead: {self.lead_agent.agent_id} ({self.lead_agent.role})")

        # Identify assistant members (all except lead)
        assist_members = [m for m in self.members if m != self.lead_agent]

        # 1. Lead asks assistants for investigations
        delivery_prompt = f"You are the lead of the medical group: '{self.group_id}', which aims to '{self.goal}'.\n"

        if assist_members:
            delivery_prompt += "Your assistant clinicians are:\n"
            for a_mem in assist_members:
                delivery_prompt += f"- {a_mem.role} (ID: {a_mem.agent_id})\n"
            delivery_prompt += "\nGiven the medical query below, what specific insights or analyses are needed from each assistant based on their expertise?\n"
        else:
            delivery_prompt += "\nGiven the medical query below, please provide your comprehensive analysis based on your expertise.\n"

        delivery_prompt += f"\n--- Medical Query ---\nQuestion: {self.question_context['question']}\n"

        if self.question_context.get('options'):
            options_str = "\n".join([f"{k}: {v}" for k, v in self.question_context['options'].items()])
            delivery_prompt += f"Options:\n{options_str}\n"

        if self.question_context.get('image_path'):
            delivery_prompt += f"An associated image is provided.\n"

        delivery_prompt += "--- End Query ---\n\nProvide a concise summary of required investigations or your direct analysis if no assistants."

        # Lead agent's request to assistants
        lead_request = self.lead_agent.chat(
            prompt=delivery_prompt,
            image_path=self.question_context.get('image_path'),
            temperature=0.3  # Lower temperature for more focused request
        )

        self._log_interaction(f"Lead ({self.lead_agent.agent_id}) requested investigations/analysis:\n{lead_request}")

        # 2. Assistants provide their investigations/analysis
        investigations = []
        for a_mem in assist_members:
            investigation_prompt = (
                f"You are {a_mem.role} (ID: {a_mem.agent_id}) in medical group '{self.group_id}' with the goal: '{self.goal}'.\n"
                f"Your group lead ({self.lead_agent.role}, ID: {self.lead_agent.agent_id}) requires your input based on the following request:\n'{lead_request}'\n\n"
                f"Please provide your investigation summary or analysis focusing on your expertise regarding the medical query:\n"
                f"Question: {self.question_context['question']}\n"
            )

            if self.question_context.get('options'):
                options_str = "\n".join([f"{k}: {v}" for k, v in self.question_context['options'].items()])
                investigation_prompt += f"Options:\n{options_str}\n"

            if self.question_context.get('image_path'):
                investigation_prompt += f"(Image provided)\n"

            investigation_prompt += "\nKeep your response focused and relevant to the group's goal."

            # Get the assistant's response
            investigation = a_mem.chat(
                prompt=investigation_prompt,
                image_path=self.question_context.get('image_path'),
                temperature=0.3  # Lower temperature for more analytical response
            )

            investigations.append({"role": a_mem.role, "id": a_mem.agent_id, "report": investigation})
            self._log_interaction(f"Assistant ({a_mem.agent_id} - {a_mem.role}) provided report:\n{investigation[:150]}...")

        # 3. Lead synthesizes the information
        gathered_investigation = ""
        if investigations:
            gathered_investigation += "Gathered insights from assistant clinicians:\n"
            for inv in investigations:
                gathered_investigation += f"--- Report from {inv['role']} (ID: {inv['id']}) ---\n{inv['report']}\n---\n"
        else:
            gathered_investigation = "No assistant reports were generated. Relying solely on lead's analysis."

        synthesis_prompt = f"{gathered_investigation}\n\n"
        synthesis_prompt += f"As the lead ({self.lead_agent.role}) of group '{self.group_id}' aiming to '{self.goal}', synthesize the gathered information (including your own initial thoughts if applicable) "
        synthesis_prompt += f"to provide a comprehensive report or final answer for the group regarding the medical query:\n"
        synthesis_prompt += f"Question: {self.question_context['question']}\n"

        if self.question_context.get('options'):
            options_str = "\n".join([f"{k}: {v}" for k, v in self.question_context['options'].items()])
            synthesis_prompt += f"Options:\n{options_str}\n"
            synthesis_prompt += "Respond in JSON format with 'answer' (letter for multiple choice) and 'explanation' fields.\n"
        else:
            synthesis_prompt += "Respond in JSON format with 'answer' and 'explanation' fields.\n"

        if self.question_context.get('image_path'):
            synthesis_prompt += f"(Image provided)\n"

        synthesis_prompt += "\n--- Group Report ---\n"

        # Generate the final group report
        final_report = self.lead_agent.chat(
            prompt=synthesis_prompt,
            image_path=self.question_context.get('image_path'),
            response_format={"type": "json_object"},
            temperature=0.2  # Lower temperature for more definitive answer
        )

        self._log_interaction(f"Lead ({self.lead_agent.agent_id}) generated final group report:\n{final_report[:200]}...")

        return final_report

# --- MDAgents Framework Class ---

class MDAgentsFramework:
    """
    Orchestrates the MDAgents workflow: complexity check, recruitment,
    and query processing based on complexity.
    """

    def __init__(self,
                 log_dir: str,
                 dataset_name: str,
                 model_config: Dict[str, str], # Keys like 'moderator', 'recruiter', 'default_agent'
                 max_rounds_intermediate: int = DEFAULT_MAX_ROUNDS_INTERMEDIATE,
                 max_turns_intermediate: int = DEFAULT_MAX_TURNS_INTERMEDIATE,
                 num_experts_intermediate: int = DEFAULT_NUM_EXPERTS_INTERMEDIATE,
                 num_teams_advanced: int = DEFAULT_NUM_TEAMS_ADVANCED,
                 num_agents_per_team_advanced: int = DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED):
        """
        Initialize the MDAgents framework orchestrator.

        Args:
            log_dir: Directory to save logs and results.
            dataset_name: Name of the dataset being processed.
            model_config: Dictionary mapping roles ('moderator', 'recruiter', 'default_agent') to model keys.
            max_rounds_intermediate: Max discussion rounds for intermediate complexity.
            max_turns_intermediate: Max turns per round for intermediate complexity.
            num_experts_intermediate: Number of experts to recruit for intermediate.
            num_teams_advanced: Number of teams to recruit for advanced.
            num_agents_per_team_advanced: Number of agents per team for advanced.
        """
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.max_rounds_intermediate = max_rounds_intermediate
        self.max_turns_intermediate = max_turns_intermediate
        self.num_experts_intermediate = num_experts_intermediate
        self.num_teams_advanced = num_teams_advanced
        self.num_agents_per_team_advanced = num_agents_per_team_advanced

        os.makedirs(self.log_dir, exist_ok=True)

        # --- Initialize Core Agents ---
        self.moderator_agent = BaseAgent(
            agent_id="moderator",
            role=AgentRole.MODERATOR,
            model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
            instruction="You are a medical expert who conducts initial assessment. Your job is to decide the difficulty/complexity of the medical query based on the provided definitions. Respond in JSON format."
        )

        self.recruiter_agent = BaseAgent(
            agent_id="recruiter",
            role=AgentRole.RECRUITER,
            model_key=model_config.get('recruiter', DEFAULT_RECRUITER_MODEL),
            instruction="You are an experienced medical expert who recruits appropriate specialists based on the medical query and its complexity level. Respond in JSON format."
        )

        self.decision_maker_agent = BaseAgent(
            agent_id="final_decision_maker",
            role=AgentRole.DECISION_MAKER,
            model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL), # Reuse moderator model for final synthesis
            instruction="You are a final medical decision maker. Review all provided information (opinions, reports, discussions) and make the final, consolidated answer to the original medical query. Respond in JSON format."
        )

        print("MDAgentsFramework Initialized.")
        print(f" - Log Directory: {self.log_dir}")
        print(f" - Dataset: {self.dataset_name}")
        print(f" - Models: Moderator={self.moderator_agent.model_key}, Recruiter={self.recruiter_agent.model_key}, DefaultAgent={model_config.get('default_agent', DEFAULT_AGENT_MODEL)}")
        print(f" - Intermediate Settings: Max Rounds={self.max_rounds_intermediate}, Max Turns={self.max_turns_intermediate}, Experts={self.num_experts_intermediate}")
        print(f" - Advanced Settings: Teams={self.num_teams_advanced}, Agents Per Team={self.num_agents_per_team_advanced}")

    def _determine_complexity(self, question: str, options: Optional[Dict] = None, image_path: Optional[str] = None) -> ComplexityLevel:
        """
        Uses the Moderator agent to classify the query complexity.

        Args:
            question: The medical question.
            options: Optional multiple-choice options.
            image_path: Optional path to a medical image.

        Returns:
            The determined ComplexityLevel.
        """
        print("\n--- Determining Complexity ---")
        self.moderator_agent.clear_memory() # Ensure fresh context

        # Prepare question context for complexity check
        query_context = f"Medical Query:\n{question}\n"
        if options:
            options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            query_context += f"Options:\n{options_str}\n"
        if image_path:
            query_context += "This query includes a medical image.\n"

        # Prompt based on paper Appendix Figure: Complexity check prompt
        prompt = (
            f"Given the medical query below, decide its difficulty/complexity.\n\n"
            f"{query_context}\n"
            f"Complexity Guidelines:\n"
            f"1) basic: A single medical agent (like a PCP or general physician) can likely answer this knowledge question or simple case directly.\n"
            f"2) intermediate: Requires discussion among a team of medical experts with different specialties to reach a consensus.\n"
            f"3) advanced: A complex case requiring multiple teams (e.g., initial assessment, diagnostics, final review) collaborating across departments.\n\n"
            f"Respond with a JSON object containing a 'complexity' field with one of these values: 'basic', 'intermediate', or 'advanced'."
        )

        # For image-based queries, we can optionally include the image in the complexity check
        response = self.moderator_agent.chat(
            prompt=prompt,
            image_path=None,  # Don't use image for complexity check (relying on text description)
            response_format={"type": "json_object"},
            temperature=0.1   # Low temp for more consistent classification
        )

        try:
            # Clean and parse the JSON response
            response_clean = preprocess_response_string(response)
            response_json = json.loads(response_clean)
            complexity_str = response_json.get("complexity", "").lower()

            if complexity_str == "basic":
                print("Complexity: BASIC")
                return ComplexityLevel.BASIC
            elif complexity_str == "intermediate":
                print("Complexity: INTERMEDIATE")
                return ComplexityLevel.INTERMEDIATE
            elif complexity_str == "advanced":
                print("Complexity: ADVANCED")
                return ComplexityLevel.ADVANCED
            else:
                print(f"Warning: Invalid complexity value '{complexity_str}'. Defaulting to INTERMEDIATE.")
                return ComplexityLevel.INTERMEDIATE

        except Exception as e:
            print(f"Error parsing complexity response: {e}. Raw response: {response}. Defaulting to INTERMEDIATE.")
            return ComplexityLevel.INTERMEDIATE

    def _recruit_experts(self,
                       question: str,
                       options: Optional[Dict],
                       complexity: ComplexityLevel,
                       image_path: Optional[str] = None) -> Union[List[Dict], List[Dict[str, Any]]]:
        """
        Uses the Recruiter agent to identify necessary experts or teams.

        Args:
            question: The medical question.
            options: Optional multiple-choice options.
            complexity: The determined complexity level.
            image_path: Optional path to a medical image.

        Returns:
            - For BASIC: Empty list (single agent is used)
            - For INTERMEDIATE: A list of agent config dictionaries [{"role": str, "expertise": str, "hierarchy": str}].
            - For ADVANCED: A list of group config dictionaries [{"group_id": str, "goal": str, "members": [{"role": str, "expertise": str}]}].
        """
        print("\n--- Recruiting Experts/Teams ---")
        self.recruiter_agent.clear_memory()

        # Skip recruitment for BASIC complexity
        if complexity == ComplexityLevel.BASIC:
            print("Basic complexity - no recruitment needed.")
            return []

        # Prepare query context
        query_context = f"Medical Query:\n{question}\n"
        if options:
            options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            query_context += f"Options:\n{options_str}\n"
        if image_path:
            query_context += "This query includes a medical image.\n"

        # --- INTERMEDIATE RECRUITMENT ---
        if complexity == ComplexityLevel.INTERMEDIATE:
            # Update recruiter instruction for intermediate complexity
            recruitment_instruction = (
                f"You are an experienced medical expert. Your task is to recruit a team of {self.num_experts_intermediate} experts "
                f"with diverse specialties and expertise to discuss and solve the given medical query. "
                f"Specify their role, a brief expertise description, and optionally a communication hierarchy (e.g., 'Cardiologist > Nurse', 'Independent'). "
                f"Respond in JSON format."
            )

            self.recruiter_agent.instruction = recruitment_instruction

            prompt = (
                f"{query_context}\n"
                f"Recruit {self.num_experts_intermediate} experts for this moderately complex query.\n"
                f"Respond with a JSON array 'experts' where each expert object has fields: 'role', 'expertise', and 'hierarchy'.\n"
                f"Example response structure:\n"
                f"{{\"experts\": [\n"
                f"  {{\"role\": \"Pediatrician\", \"expertise\": \"Specializes in child healthcare\", \"hierarchy\": \"Independent\"}},\n"
                f"  {{\"role\": \"Cardiologist\", \"expertise\": \"Focuses on heart conditions\", \"hierarchy\": \"Pediatrician > Cardiologist\"}},\n"
                f"  {{\"role\": \"Pulmonologist\", \"expertise\": \"Specializes in respiratory disorders\", \"hierarchy\": \"Independent\"}}\n"
                f"]}}"
            )

            # Get the recruiter's response
            recruitment_response = self.recruiter_agent.chat(
                prompt=prompt,
                image_path=None,  # No need for image in recruitment
                response_format={"type": "json_object"},
                temperature=0.5   # Medium temperature for creativity in team composition
            )

            print(f"Recruiter Response (Intermediate):\n{recruitment_response}")

            # Parse the response to extract expert configurations
            try:
                # Clean and parse the JSON response
                response_clean = preprocess_response_string(recruitment_response)
                response_json = json.loads(response_clean)
                experts = response_json.get("experts", [])

                # Validate the structure
                validated_experts = []
                for expert in experts:
                    if isinstance(expert, dict) and 'role' in expert:
                        validated_expert = {
                            "role": expert.get("role", "Unknown Role"),
                            "expertise": expert.get("expertise", "General expertise related to the role."),
                            "hierarchy": expert.get("hierarchy", "Independent")
                        }
                        validated_experts.append(validated_expert)

                experts = validated_experts

            except Exception as e:
                print(f"Error parsing expert recruitment response: {e}. Raw response: {recruitment_response}")
                # Fallback to default roles
                print("Warning: Failed to parse experts. Using default roles.")
                default_roles = ["Internal Medicine Specialist", "Radiologist", "Surgeon", "Pathologist", "Pharmacist"]
                experts = [{"role": r, "expertise": f"Expertise in {r}", "hierarchy": "Independent"} for r in default_roles[:self.num_experts_intermediate]]

            print(f"Recruited Experts: {[e['role'] for e in experts]}")
            return experts

        # --- ADVANCED RECRUITMENT ---
        elif complexity == ComplexityLevel.ADVANCED:
            # Update recruiter instruction for advanced complexity
            recruitment_instruction = (
                f"You are an experienced medical director. Your task is to organize {self.num_teams_advanced} Multidisciplinary Teams (MDTs) "
                f"for a complex medical query. Each MDT should have around {self.num_agents_per_team_advanced} clinicians. Define the purpose (goal) of each team "
                f"and list its members with their roles and expertise. Ensure you include an 'Initial Assessment Team (IAT)' and a 'Final Review and Decision Team (FRDT)'. "
                f"Respond in JSON format."
            )

            self.recruiter_agent.instruction = recruitment_instruction

            prompt = (
                f"{query_context}\n"
                f"Organize {self.num_teams_advanced} MDTs, each with ~{self.num_agents_per_team_advanced} members, for this complex query.\n"
                f"Include an IAT and an FRDT.\n"
                f"Respond with a JSON object containing a 'teams' array where each team has: 'group_id', 'goal', and 'members' array.\n"
                f"Each member should have 'role', 'expertise', and optionally 'is_lead' (boolean) fields.\n"
                f"Example structure:\n"
                f"{{\"teams\": [\n"
                f"  {{\"group_id\": \"Group 1\", \"goal\": \"Initial Assessment Team (IAT)\", \"members\": [\n"
                f"    {{\"role\": \"Emergency Physician\", \"expertise\": \"Handles acute assessment\", \"is_lead\": true}},\n"
                f"    {{\"role\": \"Triage Nurse\", \"expertise\": \"Gathers initial patient data\"}}\n"
                f"  ]}},\n"
                f"  {{\"group_id\": \"Group 2\", \"goal\": \"Final Review and Decision Team (FRDT)\", \"members\": [\n"
                f"    {{\"role\": \"Senior Consultant\", \"expertise\": \"Oversees final decision\", \"is_lead\": true}},\n"
                f"    {{\"role\": \"Clinical Pharmacist\", \"expertise\": \"Medication review\"}}\n"
                f"  ]}}\n"
                f"]}}"
            )

            # Get the recruiter's response
            recruitment_response = self.recruiter_agent.chat(
                prompt=prompt,
                image_path=None,  # No need for image in recruitment
                response_format={"type": "json_object"},
                temperature=0.5   # Medium temperature for creativity in team composition
            )

            print(f"Recruiter Response (Advanced):\n{recruitment_response}")

            # Parse the response into groups and members
            try:
                # Clean and parse the JSON response
                response_clean = preprocess_response_string(recruitment_response)
                response_json = json.loads(response_clean)
                teams = response_json.get("teams", [])

                # Validate the structure
                validated_teams = []
                for team in teams:
                    if isinstance(team, dict) and 'group_id' in team and 'members' in team:
                        validated_members = []
                        for member in team.get('members', []):
                            if isinstance(member, dict) and 'role' in member:
                                validated_member = {
                                    "role": member.get("role", "Unknown Role"),
                                    "expertise": member.get("expertise", "General expertise for the role.")
                                }
                                # Only include is_lead if it's true
                                if member.get("is_lead") is True:
                                    validated_member["is_lead"] = True
                                validated_members.append(validated_member)

                        validated_team = {
                            "group_id": team.get("group_id", f"Group {len(validated_teams)+1}"),
                            "goal": team.get("goal", f"Goal for Group {len(validated_teams)+1}"),
                            "members": validated_members
                        }
                        validated_teams.append(validated_team)

                teams = validated_teams

            except Exception as e:
                print(f"Error parsing team recruitment response: {e}. Raw response: {recruitment_response}")
                # Fallback if no teams were successfully parsed
                print("Warning: Failed to parse any teams. Using default structure.")
                # Create default teams with standard structure
                teams = [
                    {"group_id": "Group 1", "goal": "Initial Assessment Team (IAT)", "members": [
                        {"role": "Emergency Physician", "expertise": "Acute assessment", "is_lead": True},
                        {"role": "Radiologist", "expertise": "Initial imaging"}
                    ]},
                    {"group_id": "Group 2", "goal": "Diagnostic Team", "members": [
                        {"role": "Cardiologist", "expertise": "Heart conditions", "is_lead": True},
                        {"role": "Neurologist", "expertise": "Nervous system"}
                    ]},
                    {"group_id": "Group 3", "goal": "Final Review and Decision Team (FRDT)", "members": [
                        {"role": "Senior Consultant", "expertise": "Oversees decision", "is_lead": True},
                        {"role": "Clinical Pharmacist", "expertise": "Medication review"}
                    ]}
                ]
                teams = teams[:self.num_teams_advanced]  # Adjust to requested number

            print(f"Recruited Teams: {[t['goal'] for t in teams]}")
            return teams

        else:  # BASIC complexity (should not reach here as we check earlier)
            return []

    def _process_basic_query(self, data_item: Dict) -> Dict:
        """
        Handles low complexity queries using a single agent.

        Args:
            data_item: Dictionary containing the query details (qid, question, options, image_path, answer).

        Returns:
            A dictionary containing the result ('predicted_answer', 'explanation').
        """
        print("\n--- Processing Basic Query ---")

        # Create a single general doctor agent
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        agent = BaseAgent(
            agent_id="basic_solver",
            role=AgentRole.GENERAL_DOCTOR,
            model_key=agent_model_key,
            instruction="You are a helpful medical assistant. Answer the following medical question accurately. Respond in JSON format."
        )

        # Prepare the main prompt
        main_prompt = f"Question: {data_item['question']}\n"
        options = data_item.get('options')

        if options:
            options_str = "Options:\n" + "\n".join([f"({k}) {v}" for k, v in options.items()])
            main_prompt += f"{options_str}\n"
            main_prompt += "\nProvide your answer as a JSON object with 'answer' (letter for multiple-choice) and 'explanation' fields."
        else:  # Free form
            main_prompt += "\nProvide your answer as a JSON object with 'answer' and 'explanation' fields."

        # Get the agent's response
        response = agent.chat(
            prompt=main_prompt,
            image_path=data_item.get('image_path'),
            response_format={"type": "json_object"},
            temperature=0.2  # Lower temperature for more factual recall
        )

        # Parse the response
        try:
            # Clean and parse the JSON response
            response_clean = preprocess_response_string(response)
            response_json = json.loads(response_clean)

            predicted_answer = response_json.get("answer", "")
            explanation = response_json.get("explanation", "No explanation provided.")

            # Clean up common formats for multiple choice answers if needed
            if options and isinstance(predicted_answer, str):
                # Format (B)
                if predicted_answer.startswith('(') and predicted_answer.endswith(')'):
                    predicted_answer = predicted_answer[1:-1].strip()
                # Format (B) Option Text
                elif predicted_answer.startswith('(') and len(predicted_answer) > 2 and predicted_answer[1].isalpha() and predicted_answer[2] == ')':
                    predicted_answer = predicted_answer[1]  # Extract just the letter
                # Format "B. Option Text" or "B)"
                elif len(predicted_answer) > 1 and predicted_answer[0].isalpha() and (predicted_answer[1] == '.' or predicted_answer[1] == ')'):
                    predicted_answer = predicted_answer[0]  # Extract just the letter

        except Exception as e:
            print(f"Error parsing basic response: {e}. Raw response: {response}")
            predicted_answer = "Could not parse answer."
            explanation = "Error parsing model response."

        print(f"Basic Query Result: Answer='{predicted_answer}', Explanation='{explanation[:100]}...'")

        return {
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "complexity": ComplexityLevel.BASIC.value
        }

    def _process_intermediate_query(self, data_item: Dict, expert_configs: List[Dict]) -> Dict:
        """
        Handles intermediate complexity queries using a recruited team and discussion.

        Args:
            data_item: Query details.
            expert_configs: List of configurations for recruited experts.

        Returns:
            A dictionary containing the result ('predicted_answer', 'explanation', 'interaction_log').
        """
        print("\n--- Processing Intermediate Query ---")

        # Get default agent model for experts
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)

        # 1. Create Agent Instances
        agents = []
        agent_dict = {}  # For easy lookup by ID

        for i, config in enumerate(expert_configs):
            # Create a unique ID based on role
            agent_id = f"expert_{i+1}_{config['role'].replace(' ','_').lower()}"

            # Create the agent with appropriate instruction
            agent = BaseAgent(
                agent_id=agent_id,
                role=config['role'],  # Use recruited role
                model_key=agent_model_key,  # Use default agent model for all experts
                instruction=f"You are a {config['role']} with expertise in {config['expertise']}. Collaborate with other medical experts to answer the medical query. Maintain your persona and provide insights based on your specialty. Respond in JSON format."
            )

            agents.append(agent)
            agent_dict[agent_id] = agent

        if not agents:
            print("Error: No expert agents created for intermediate query. Aborting.")
            return {"predicted_answer": "Error", "explanation": "Failed to create expert agents."}

        print(f"Created {len(agents)} expert agents: {[a.agent_id for a in agents]}")

        # Prepare question context
        question = data_item['question']
        options = data_item.get('options')
        image_path = data_item.get('image_path')

        question_context = f"Question: {question}\n"
        if options:
            options_str = "Options:\n" + "\n".join([f"({k}) {v}" for k, v in options.items()])
            question_context += f"{options_str}\n"
        if image_path:
            question_context += "(Image provided separately)\n"

        # 2. Initial Opinions (Round 1)
        print("\n-- Round 1: Initial Opinions --")
        round_opinions = {1: {}}  # Round -> Agent ID -> Opinion Dict
        initial_report_parts = []

        for agent in agents:
            agent.clear_memory()  # Start fresh for the query

            prompt = (
                f"{question_context}\n"
                f"Based on your expertise as a {agent.role}, provide your initial analysis and answer.\n"
                f"Respond with a JSON object containing 'answer' and 'explanation' fields."
            )

            response = agent.chat(
                prompt=prompt,
                image_path=image_path,
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for more focused analysis
            )

            # Parse response to extract answer and explanation
            try:
                # Clean and parse the JSON response
                response_clean = preprocess_response_string(response)
                response_json = json.loads(response_clean)

                ans = response_json.get("answer", "")
                expl = response_json.get("explanation", "No explanation provided.")

                # MC Answer cleanup if needed
                if options and isinstance(ans, str):
                    if ans.startswith('(') and ans.endswith(')'):
                        ans = ans[1:-1].strip()
                    elif ans.startswith('(') and len(ans) > 2 and ans[1].isalpha() and ans[2] == ')':
                        ans = ans[1]  # Extract just the letter
                    elif len(ans) > 1 and ans[0].isalpha() and (ans[1] == '.' or ans[1] == ')'):
                        ans = ans[0]  # Extract just the letter

            except Exception as e:
                print(f"Error parsing initial opinion from {agent.agent_id}: {e}. Raw response: {response}")
                ans = "Could not parse answer."
                expl = "Error parsing model response."

            round_opinions[1][agent.agent_id] = {"answer": ans, "explanation": expl}
            initial_report_parts.append(f"Expert {agent.role} ({agent.agent_id}):\nAnswer: {ans}\nExplanation: {expl[:200]}...\n---")
            print(f"Agent {agent.agent_id} ({agent.role}) Initial Answer: {ans}")

        # Setup interaction log structure
        interaction_log = {"rounds": []}
        current_round_data = {"round": 1, "initial_opinions": round_opinions[1], "turns": []}
        interaction_log["rounds"].append(current_round_data)

        # 3. Synthesize Final Decision
        print("\n-- Synthesizing Final Decision --")

        # Use the dedicated decision maker agent
        self.decision_maker_agent.clear_memory()

        synthesis_prompt = (
            f"You need to make a final decision for the following medical query based on initial opinions from a team of experts:\n\n"
            f"{question_context}\n\n"
            f"--- Expert Opinions (Round 1) ---\n"
            f"{''.join(initial_report_parts)}\n"
            f"--- End Opinions ---\n\n"
            f"Review these opinions carefully. Consider the different expert perspectives and their specific expertise.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )

        # Get the decision maker's synthesis
        final_response = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            temperature=0.2  # Low temperature for decisive answer
        )

        # Parse final response
        try:
            # Clean and parse the JSON response
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)

            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")

            # MC Answer cleanup if needed
            if options and isinstance(final_answer, str):
                if final_answer.startswith('(') and final_answer.endswith(')'):
                    final_answer = final_answer[1:-1].strip()
                elif final_answer.startswith('(') and len(final_answer) > 2 and final_answer[1].isalpha() and final_answer[2] == ')':
                    final_answer = final_answer[1]  # Extract just the letter
                elif len(final_answer) > 1 and final_answer[0].isalpha() and (final_answer[1] == '.' or final_answer[1] == ')'):
                    final_answer = final_answer[0]  # Extract just the letter

        except Exception as e:
            print(f"Error parsing final decision: {e}. Raw response: {final_response}")
            final_answer = "Could not parse answer."
            final_explanation = "Error parsing model response."

        print(f"Intermediate Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "interaction_log": interaction_log,  # Contains initial opinions
            "complexity": ComplexityLevel.INTERMEDIATE.value,
            "expert_configs": expert_configs  # Store the configurations for reference
        }

    def _process_advanced_query(self, data_item: Dict, team_configs: List[Dict]) -> Dict:
        """
        Handles high complexity queries using multiple recruited teams (ICT structure).

        Args:
            data_item: Query details.
            team_configs: List of configurations for recruited teams [{group_id, goal, members:[{role, expertise}]}].

        Returns:
            A dictionary containing the result ('predicted_answer', 'explanation', 'team_reports').
        """
        print("\n--- Processing Advanced Query ---")

        # Get default agent model
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)

        # Setup question context
        question_context = {
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
        }

        # 1. Create Group Instances
        groups = []
        group_dict = {}
        all_agents_in_groups = []  # Keep track of all agents created

        for i, config in enumerate(team_configs):
            members = []
            for j, member_config in enumerate(config['members']):
                # Create a unique identifier for each agent
                agent_id = f"{config['group_id'].replace(' ','_').lower()}_member_{j+1}_{member_config['role'].replace(' ','_').lower()}"

                # Determine if this agent is the lead
                is_lead = member_config.get('is_lead', False)
                instruction_prefix = f"You are {member_config['role']} ({member_config['expertise']}) in team '{config['goal']}'."
                if is_lead:
                    instruction_prefix += " You are the LEAD of this team."

                # Create the agent
                agent = BaseAgent(
                    agent_id=agent_id,
                    role=f"{member_config['role']}{' (Lead)' if is_lead else ''}",  # Indicate lead in role string
                    model_key=agent_model_key,  # Use default model for all team members
                    instruction=f"{instruction_prefix} Collaborate within your team to achieve the goal: '{config['goal']}'. Respond in JSON format."
                )

                members.append(agent)
                all_agents_in_groups.append(agent)

            # Create the group with its members
            group = Group(
                group_id=config['group_id'],
                goal=config['goal'],
                members=members,
                question_context=question_context
            )

            groups.append(group)
            group_dict[group.group_id] = group

        print(f"Created {len(groups)} teams: {[g.group_id for g in groups]}")

        # 2. Process Teams Sequentially (ICT Flow)
        # A real ICT has a pipeline: IAT -> Other Teams -> FRDT
        team_reports = {}  # Store report from each team

        # Initialize report variables
        initial_assessment_report = ""
        diagnostic_reports = ""
        final_review_report = ""

        # Process IAT (Initial Assessment Team) first
        for group in groups:
            if "initial assessment" in group.goal.lower() or "iat" in group.goal.lower():
                print(f"\n-- Processing Team: {group.group_id} ({group.goal}) --")
                raw_report = group.perform_internal_discussion()

                # Parse JSON report
                try:
                    response_clean = preprocess_response_string(raw_report)
                    report_json = json.loads(response_clean)
                    team_reports[group.group_id] = report_json
                    initial_assessment_report = f"--- Report from {group.group_id} ({group.goal}) ---\n"
                    initial_assessment_report += f"Answer: {report_json.get('answer', 'N/A')}\n"
                    initial_assessment_report += f"Explanation: {report_json.get('explanation', 'No explanation provided.')}\n---\n"
                except Exception as e:
                    print(f"Error parsing team report: {e}. Raw report: {raw_report}")
                    team_reports[group.group_id] = {"raw": raw_report}
                    initial_assessment_report = f"--- Report from {group.group_id} ({group.goal}) ---\n{raw_report}\n---\n"

                break  # Assume only one IAT

        # Process other diagnostic/specialty teams
        for group in groups:
            if not ("initial assessment" in group.goal.lower() or "iat" in group.goal.lower() or
                    "final review" in group.goal.lower() or "frdt" in group.goal.lower()):
                print(f"\n-- Processing Team: {group.group_id} ({group.goal}) --")
                # These teams can benefit from the IAT report (could pass it in future enhancement)
                raw_report = group.perform_internal_discussion()

                # Parse JSON report
                try:
                    response_clean = preprocess_response_string(raw_report)
                    report_json = json.loads(response_clean)
                    team_reports[group.group_id] = report_json
                    team_report = f"--- Report from {group.group_id} ({group.goal}) ---\n"
                    team_report += f"Answer: {report_json.get('answer', 'N/A')}\n"
                    team_report += f"Explanation: {report_json.get('explanation', 'No explanation provided.')}\n---\n"
                except Exception as e:
                    print(f"Error parsing team report: {e}. Raw report: {raw_report}")
                    team_reports[group.group_id] = {"raw": raw_report}
                    team_report = f"--- Report from {group.group_id} ({group.goal}) ---\n{raw_report}\n---\n"

                diagnostic_reports += team_report

        # Process FRDT (Final Review and Decision Team) last
        final_decision_report_from_frdt = {}
        for group in groups:
            if "final review" in group.goal.lower() or "frdt" in group.goal.lower():
                print(f"\n-- Processing Team: {group.group_id} ({group.goal}) --")

                # FRDT needs previous reports
                # In a future enhancement, we could modify the internal discussion to include previous reports
                raw_report = group.perform_internal_discussion()

                # Parse JSON report
                try:
                    response_clean = preprocess_response_string(raw_report)
                    report_json = json.loads(response_clean)
                    team_reports[group.group_id] = report_json
                    final_decision_report_from_frdt = report_json
                    final_review_report = f"--- Report from {group.group_id} ({group.goal}) ---\n"
                    final_review_report += f"Answer: {report_json.get('answer', 'N/A')}\n"
                    final_review_report += f"Explanation: {report_json.get('explanation', 'No explanation provided.')}\n---\n"
                except Exception as e:
                    print(f"Error parsing team report: {e}. Raw report: {raw_report}")
                    team_reports[group.group_id] = {"raw": raw_report}
                    final_decision_report_from_frdt = {"raw": raw_report}
                    final_review_report = f"--- Report from {group.group_id} ({group.goal}) ---\n{raw_report}\n---\n"

                break  # Assume only one FRDT

        # 3. Final Decision Synthesis (using the dedicated decision maker agent)
        print("\n-- Synthesizing Final Decision from Team Reports --")
        self.decision_maker_agent.clear_memory()

        # Compile all reports
        compiled_reports = ""
        if initial_assessment_report:
            compiled_reports += f"INITIAL ASSESSMENT:\n{initial_assessment_report}\n"
        if diagnostic_reports:
            compiled_reports += f"DIAGNOSTIC TEAMS:\n{diagnostic_reports}\n"
        if final_review_report:
            compiled_reports += f"FINAL REVIEW:\n{final_review_report}\n"

        if not compiled_reports:
            compiled_reports = "No team reports were generated."
            # Use FRDT report directly if available and others failed
            if final_decision_report_from_frdt:
                compiled_reports = f"Using FRDT report directly:\n{json.dumps(final_decision_report_from_frdt, indent=2)}"

        # Prepare the query context for the final decision
        options_str = ""
        if data_item.get('options'):
            options_str = "\nOptions: " + str(data_item['options'])

        # Generate the final synthesis prompt
        synthesis_prompt = (
            f"You need to make the ultimate final decision for the following complex medical query based on reports from multiple specialized teams:\n\n"
            f"Original Query Context:\nQuestion: {data_item['question']}{options_str}\n"
            f"{'(Image associated)' if data_item.get('image_path') else ''}\n\n"
            f"--- Compiled Team Reports ---\n"
            f"{compiled_reports}\n"
            f"--- End Reports ---\n\n"
            f"Synthesize all this information into one final, definitive answer and explanation.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )

        # Get the final decision
        final_response = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            temperature=0.2  # Low temperature for decisive answer
        )

        # Parse final response
        try:
            # Clean and parse the JSON response
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)

            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")

            # MC Answer cleanup if needed
            options = data_item.get('options')
            if options and isinstance(final_answer, str):
                if final_answer.startswith('(') and final_answer.endswith(')'):
                    final_answer = final_answer[1:-1].strip()
                elif final_answer.startswith('(') and len(final_answer) > 2 and final_answer[1].isalpha() and final_answer[2] == ')':
                    final_answer = final_answer[1]  # Extract just the letter
                elif len(final_answer) > 1 and final_answer[0].isalpha() and (final_answer[1] == '.' or final_answer[1] == ')'):
                    final_answer = final_answer[0]  # Extract just the letter

        except Exception as e:
            print(f"Error parsing final decision (advanced): {e}. Raw response: {final_response}")
            final_answer = "Could not parse answer."
            final_explanation = "Error parsing model response."

        print(f"Advanced Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "team_reports": team_reports,  # Reports from each team
            "complexity": ComplexityLevel.ADVANCED.value,
            "team_configs": team_configs  # Store the configurations for reference
        }

    def run_query(self, data_item: Dict) -> Dict:
        """
        Processes a single data item through the MDAgents framework.

        Args:
            data_item: Dictionary containing query details.

        Returns:
            A dictionary containing the full results and metadata for the query.
        """
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()

        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")
        ground_truth = data_item.get("answer")

        result_data = {}
        recruited_info = None

        try:
            # 1. Determine Complexity
            complexity = self._determine_complexity(question, options, image_path)

            # 2. Process based on complexity
            if complexity == ComplexityLevel.BASIC:
                result_data = self._process_basic_query(data_item)

            elif complexity == ComplexityLevel.INTERMEDIATE:
                expert_configs = self._recruit_experts(question, options, complexity, image_path)
                recruited_info = expert_configs  # Save for logging
                result_data = self._process_intermediate_query(data_item, expert_configs)

            elif complexity == ComplexityLevel.ADVANCED:
                team_configs = self._recruit_experts(question, options, complexity, image_path)
                recruited_info = team_configs  # Save for logging
                result_data = self._process_advanced_query(data_item, team_configs)

        except Exception as e:
            print(f"ERROR processing QID {qid}: {e}")
            # Create minimal result data in case of error
            result_data = {
                "predicted_answer": "Error occurred during processing",
                "explanation": f"Error: {str(e)}",
                "error": str(e),
                "complexity": "unknown"
            }

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s")

        # 3. Assemble final result object (similar to ColaCare)
        final_result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "complexity_level": result_data.get("complexity", "unknown"),
            "predicted_answer": result_data.get("predicted_answer", "Error"),
            "explanation": result_data.get("explanation", "N/A"),
            "recruited_info": recruited_info,  # Log experts/teams recruited
            "processing_time_seconds": processing_time,
            # Include specific logs/reports based on complexity
            "details": result_data  # Contains logs/reports specific to the complexity path
        }

        return final_result

    def run_dataset(self, data: List[Dict]):
        """
        Runs the MDAgents framework over an entire dataset.

        Args:
            data: List of data items (dictionaries).
        """
        print(f"\nStarting MDAgents processing for {len(data)} items in dataset '{self.dataset_name}'.")

        for item in tqdm(data, desc=f"Running MDAgents on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")
            # Check if result file exists
            if os.path.exists(os.path.join(self.log_dir, f"{qid}-result.json")):
                print(f"Skipping {qid} - result file already exists.")
                continue

            try:
                result = self.run_query(item)
                save_json(result, os.path.join(self.log_dir, f"{qid}-result.json"))
            except Exception as e:
                print(f"FATAL ERROR during run_query for QID {qid}: {e}")

        print(f"Finished processing dataset '{self.dataset_name}'. Results saved in {self.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MDAgents Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name (e.g., vqa_rad, pathvqa, medqa)")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--moderator_model", type=str, default=DEFAULT_MODERATOR_MODEL, help="Model key for the Moderator agent")
    parser.add_argument("--recruiter_model", type=str, default=DEFAULT_RECRUITER_MODEL, help="Model key for the Recruiter agent")
    parser.add_argument("--agent_model", type=str, default=DEFAULT_AGENT_MODEL, help="Default model key for solver agents")

    # Advanced settings
    parser.add_argument("--num_experts", type=int, default=DEFAULT_NUM_EXPERTS_INTERMEDIATE, help="Number of experts for intermediate complexity")
    parser.add_argument("--num_teams", type=int, default=DEFAULT_NUM_TEAMS_ADVANCED, help="Number of teams for advanced complexity")
    parser.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS_INTERMEDIATE, help="Maximum discussion rounds for intermediate complexity")

    args = parser.parse_args()

    method_name = "MDAgents"  # Identify the method

    # Format paths
    data_path = f"./my_datasets/processed/{args.dataset}/medqa_{args.qa_type}.json"

    # Create logs directory structure consistent with ColaCare
    logs_dir = os.path.join("./logs", args.dataset,
                           "multiple_choice" if args.qa_type == "mc" else "free-form",
                           method_name)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Using Log Directory: {logs_dir}")

    # Load the main dataset
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Configure models
    model_config = {
        "moderator": args.moderator_model,
        "recruiter": args.recruiter_model,
        "default_agent": args.agent_model
    }

    # Initialize MDAgents Framework
    framework = MDAgentsFramework(
        log_dir=logs_dir,
        dataset_name=args.dataset,
        model_config=model_config,
        num_experts_intermediate=args.num_experts,
        num_teams_advanced=args.num_teams,
        max_rounds_intermediate=args.max_rounds
    )

    # Run the framework on the dataset
    framework.run_dataset(data)


if __name__ == "__main__":
    main()