"""
Multi-agent system for medical literature simplification based on AgentSimp approach.
This implementation adapts the collaborative document simplification framework to
medical literature, creating accessible lay summaries through specialized agent roles.

File: medagentboard/laysummary/multi_agent_agentsimp.py
"""

import os
import json
import time
import argparse
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm
from openai import OpenAI

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


class AgentRole(Enum):
    """Enumeration of agent roles in the medical literature simplification process."""
    PROJECT_DIRECTOR = "Project Director"
    STRUCTURE_ANALYST = "Structure Analyst"
    CONTENT_SIMPLIFIER = "Content Simplifier"
    SIMPLIFY_SUPERVISOR = "Simplify Supervisor"
    METAPHOR_ANALYST = "Metaphor Analyst"
    TERMINOLOGY_INTERPRETER = "Terminology Interpreter"
    CONTENT_INTEGRATOR = "Content Integrator"
    ARTICLE_ARCHITECT = "Article Architect"
    PROOFREADER = "Proofreader"


class CommunicationStrategy(Enum):
    """Enumeration of communication strategies between agents."""
    PIPELINE = "Pipeline"
    SYNCHRONOUS = "Synchronous"


class ConstructionStrategy(Enum):
    """Enumeration of document construction strategies."""
    DIRECT = "Direct"
    ITERATIVE = "Iterative"


class BaseAgent:
    """Base class for all agents in the medical literature simplification process."""

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        model_key: str = "deepseek-v3-official"
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent in the simplification process
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.role = role
        self.model_key = model_key
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        print(f"Initialized {role.value} agent (ID: {agent_id}) with model: {model_key}")

    def call_llm(
        self,
        system_message: Dict[str, str],
        user_message: Dict[str, str],
        max_retries: int = 3
    ) -> str:
        """
        Call the LLM with retry mechanism.

        Args:
            system_message: System message for LLM
            user_message: User message for LLM
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM with role {self.role.value}")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    response_format={"type": "json_object"},
                    temperature=0.3,  # Lower temperature for more deterministic outputs
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

    def add_to_memory(self, content: Dict[str, Any], action_type: str) -> None:
        """
        Add content to agent's memory.

        Args:
            content: Content to add to memory
            action_type: Type of action performed
        """
        self.memory.append({
            "type": action_type,
            "timestamp": time.time(),
            "content": content
        })


class ProjectDirectorAgent(BaseAgent):
    """
    Project Director agent responsible for providing overall
    guidance for medical text simplification.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Project Director agent."""
        super().__init__(agent_id, AgentRole.PROJECT_DIRECTOR, model_key)

    def create_simplification_guideline(self, medical_text: str) -> Dict[str, Any]:
        """
        Create a guideline for simplifying the medical text.

        Args:
            medical_text: Medical text to be simplified

        Returns:
            Dictionary containing simplification guidelines
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert medical editor specializing in creating clear guidelines for "
                "simplifying complex medical literature into accessible content for general audiences. "
                "Your task is to analyze the provided medical text and create a comprehensive simplification "
                "guideline that will help other agents produce an effective single-paragraph lay summary. "
                "Your response should be in JSON format and include the following elements:\n"
                "1. 'summary': A brief summary of the main points in the medical text\n"
                "2. 'target_audience': Description of the intended audience for the simplified version\n"
                "3. 'key_medical_concepts': List of important medical concepts that must be preserved\n"
                "4. 'simplification_level': Suggested reading level (e.g., '8th grade', 'high school')\n"
                "5. 'tone': Recommended tone for the simplified text\n"
                "6. 'single_paragraph_focus': Specific instructions for creating a cohesive single paragraph\n"
                "7. 'terminology_guidance': Advice on handling specific medical terms\n"
            )
        }

        user_message = {
            "role": "user",
            "content": (
                f"Please create a simplification guideline for the following medical text. "
                f"Focus on making this accessible to a general audience in a single paragraph "
                f"while preserving medical accuracy:\n\n{medical_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Project Director created simplification guideline")
            self.add_to_memory(result, "create_guideline")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Project Director response as JSON")
            # Create a basic fallback structure
            fallback = {
                "summary": "Medical text summary",
                "target_audience": "General public",
                "key_medical_concepts": [],
                "simplification_level": "High school",
                "tone": "Informative and accessible",
                "single_paragraph_focus": "Create a cohesive single paragraph summary",
                "terminology_guidance": "Explain technical terms"
            }
            self.add_to_memory(fallback, "create_guideline_fallback")
            return fallback


class StructureAnalystAgent(BaseAgent):
    """
    Structure Analyst agent responsible for identifying key information
    for the lay summary of medical text.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Structure Analyst agent."""
        super().__init__(agent_id, AgentRole.STRUCTURE_ANALYST, model_key)

    def create_structural_outline(self, medical_text: str) -> Dict[str, Any]:
        """
        Identify key information for the lay summary.

        Args:
            medical_text: Medical text to be simplified

        Returns:
            Dictionary containing key information for the lay summary
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in analyzing medical literature and identifying the most important "
                "information for a lay audience. Your task is to analyze the provided medical text and "
                "identify the key elements that should be included in a single-paragraph lay summary. "
                "Focus on extracting information about the scope of the study/review, main interventions, "
                "key findings, and limitations. Your response should be in JSON format and include:\n"
                "1. 'title': A simple, clear title for the summary\n"
                "2. 'key_elements': A list of the most important information that should be included in the summary\n"
                "3. 'main_conclusion': The primary conclusion or finding of the medical text\n"
            )
        }

        user_message = {
            "role": "user",
            "content": (
                f"Please analyze the following medical text and identify the key information "
                f"that should be included in a single-paragraph lay summary:\n\n{medical_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Structure Analyst identified key information for lay summary")
            self.add_to_memory(result, "create_outline")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Structure Analyst response as JSON")
            # Create a basic fallback structure
            fallback = {
                "title": "Medical Text Summary",
                "key_elements": ["Key information from the medical text"],
                "main_conclusion": "Primary finding or conclusion"
            }
            self.add_to_memory(fallback, "create_outline_fallback")
            return fallback


class ContentSimplifierAgent(BaseAgent):
    """
    Content Simplifier agent responsible for the initial
    simplification of the medical text into a single paragraph.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Content Simplifier agent."""
        super().__init__(agent_id, AgentRole.CONTENT_SIMPLIFIER, model_key)

    def simplify_content(
        self,
        medical_text: str,
        guideline: Dict[str, Any],
        outline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform initial simplification of the medical text into a single paragraph.

        Args:
            medical_text: Medical text to be simplified
            guideline: Simplification guideline from Project Director
            outline: Key information from Structure Analyst

        Returns:
            Dictionary containing simplified content
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in simplifying complex medical content for general audiences. "
                "Your task is to transform the provided medical text into a simplified, single-paragraph "
                "lay summary that is accessible to non-experts while maintaining accuracy. "
                "The summary should include the scope of the study/review, main interventions, key findings, "
                "and any important limitations or uncertainties. Follow the provided guideline and "
                "focus on the key elements identified. Apply techniques such as:\n"
                "- Simplifying complex sentences\n"
                "- Replacing technical terms with plain language\n"
                "- Focusing on the most important information\n"
                "- Using active voice and direct language\n"
                "Your response should be in JSON format and include:\n"
                "1. 'lay_summary': A single-paragraph simplified version of the medical text\n"
                "2. 'simplification_techniques': Brief explanation of techniques used\n"
            )
        }

        # Convert guideline and outline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])
        outline_text = json.dumps(outline, indent=2)

        user_message = {
            "role": "user",
            "content": (
                f"Please simplify the following medical text into a single-paragraph lay summary "
                f"according to the guideline and focusing on the key elements identified:\n\n"
                f"MEDICAL TEXT:\n{medical_text}\n\n"
                f"GUIDELINE:\n{guideline_text}\n\n"
                f"KEY INFORMATION:\n{outline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Content Simplifier created single-paragraph lay summary")
            self.add_to_memory(result, "simplify_content")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Content Simplifier response as JSON")
            # Create a basic fallback result
            fallback = {
                "lay_summary": "Simplified version of the medical text.",
                "simplification_techniques": "Basic simplification applied"
            }
            self.add_to_memory(fallback, "simplify_content_fallback")
            return fallback


class SimplifySupervisorAgent(BaseAgent):
    """
    Simplify Supervisor agent responsible for reviewing
    and providing feedback on simplified content.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Simplify Supervisor agent."""
        super().__init__(agent_id, AgentRole.SIMPLIFY_SUPERVISOR, model_key)

    def review_simplification(
        self,
        original_text: str,
        simplified_text: str,
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review the simplified content and provide feedback.

        Args:
            original_text: Original medical text
            simplified_text: Simplified version of the text
            guideline: Simplification guideline

        Returns:
            Dictionary containing review and suggestions
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert editor specialized in reviewing simplified medical content. "
                "Your task is to critically review a single-paragraph lay summary of medical text and provide "
                "constructive feedback to improve its accessibility while ensuring medical accuracy. "
                "Consider aspects such as:\n"
                "- Accuracy of medical information\n"
                "- Appropriate language level for target audience\n"
                "- Clarity and logical flow in a single paragraph\n"
                "- Appropriate handling of medical terminology\n"
                "- Areas that need further simplification\n"
                "Your response should be in JSON format and include:\n"
                "1. 'feedback': Overall assessment of the simplified text\n"
                "2. 'accuracy_issues': Any inaccuracies or omissions of important information\n"
                "3. 'clarity_issues': Areas that could be clearer or more accessible\n"
                "4. 'suggested_improvements': Specific suggestions for improvement\n"
                "5. 'revised_summary': An improved version incorporating your suggestions, still as a single paragraph\n"
            )
        }

        # Convert guideline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])

        user_message = {
            "role": "user",
            "content": (
                f"Please review the following single-paragraph lay summary against the original "
                f"text and the simplification guideline:\n\n"
                f"ORIGINAL TEXT:\n{original_text}\n\n"
                f"LAY SUMMARY:\n{simplified_text}\n\n"
                f"GUIDELINE:\n{guideline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Simplify Supervisor provided review and suggestions")
            self.add_to_memory(result, "review_simplification")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Simplify Supervisor response as JSON")
            # Create a basic fallback result
            fallback = {
                "feedback": "Basic review of simplified text",
                "accuracy_issues": "None identified",
                "clarity_issues": "Some areas may need clarification",
                "suggested_improvements": "Consider simplifying further",
                "revised_summary": simplified_text
            }
            self.add_to_memory(fallback, "review_simplification_fallback")
            return fallback


class MetaphorAnalystAgent(BaseAgent):
    """
    Metaphor Analyst agent responsible for explaining complex
    medical concepts using metaphors and analogies.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Metaphor Analyst agent."""
        super().__init__(agent_id, AgentRole.METAPHOR_ANALYST, model_key)

    def analyze_and_simplify(
        self,
        medical_text: str,
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze complex concepts and provide metaphorical explanations.

        Args:
            medical_text: Medical text to analyze
            guideline: Simplification guideline

        Returns:
            Dictionary containing metaphorical analysis and simplified text
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in explaining complex medical concepts through metaphors and "
                "analogies in plain language. Your task is to identify complex concepts in "
                "the provided lay summary and enhance it with clear, relatable explanations using everyday "
                "comparisons that a general audience can understand. Maintain the single paragraph format. "
                "Your response should be in JSON format and include:\n"
                "1. 'identified_concepts': List of complex medical concepts identified\n"
                "2. 'metaphorical_explanations': For each concept, provide a metaphor or analogy\n"
                "3. 'enhanced_summary': The single-paragraph summary with metaphorical explanations integrated\n"
            )
        }

        # Convert guideline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])

        user_message = {
            "role": "user",
            "content": (
                f"Please analyze the following lay summary, identify complex concepts, "
                f"and enhance it with metaphorical explanations while maintaining a single paragraph "
                f"format according to the simplification guideline:\n\n"
                f"LAY SUMMARY:\n{medical_text}\n\n"
                f"GUIDELINE:\n{guideline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            # Convert identified_concepts to a list of strings if it's not already
            if 'identified_concepts' in result and not isinstance(result['identified_concepts'], list):
                result['identified_concepts'] = [str(result['identified_concepts'])]
            # Ensure metaphorical_explanations is a dictionary
            if 'metaphorical_explanations' in result and not isinstance(result['metaphorical_explanations'], dict):
                result['metaphorical_explanations'] = {'concept': str(result['metaphorical_explanations'])}

            print(f"Metaphor Analyst provided metaphorical explanations")
            self.add_to_memory(result, "metaphor_analysis")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Metaphor Analyst response as JSON")
            # Create a basic fallback result
            fallback = {
                "identified_concepts": ["medical concept"],
                "metaphorical_explanations": {"medical concept": "simplified explanation"},
                "enhanced_summary": medical_text
            }
            self.add_to_memory(fallback, "metaphor_analysis_fallback")
            return fallback


class TerminologyInterpreterAgent(BaseAgent):
    """
    Terminology Interpreter agent responsible for explaining
    medical terminology in plain language.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Terminology Interpreter agent."""
        super().__init__(agent_id, AgentRole.TERMINOLOGY_INTERPRETER, model_key)

    def interpret_terminology(
        self,
        medical_text: str,
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify and explain medical terminology in plain language.

        Args:
            medical_text: Medical text containing terminology
            guideline: Simplification guideline

        Returns:
            Dictionary containing terminology explanations and simplified text
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in medical terminology who specializes in translating technical "
                "terms into plain language for general audiences. Your task is to identify medical "
                "terms in the provided lay summary and ensure they are explained in clear, accessible language. "
                "Maintain the single paragraph format. Your response should be in JSON format and include:\n"
                "1. 'identified_terms': List of medical terms identified\n"
                "2. 'term_explanations': For each term, provide a plain language explanation\n"
                "3. 'clarified_summary': The single-paragraph summary with medical terms clarified\n"
            )
        }

        # Convert guideline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])

        user_message = {
            "role": "user",
            "content": (
                f"Please identify and explain medical terminology in the following lay summary "
                f"while maintaining a single paragraph format according to the simplification guideline:\n\n"
                f"LAY SUMMARY:\n{medical_text}\n\n"
                f"GUIDELINE:\n{guideline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            # Convert identified_terms to a list of strings if it's not already
            if 'identified_terms' in result and not isinstance(result['identified_terms'], list):
                result['identified_terms'] = [str(result['identified_terms'])]
            # Ensure term_explanations is a dictionary
            if 'term_explanations' in result and not isinstance(result['term_explanations'], dict):
                result['term_explanations'] = {'term': str(result['term_explanations'])}

            print(f"Terminology Interpreter provided terminology explanations")
            self.add_to_memory(result, "terminology_interpretation")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Terminology Interpreter response as JSON")
            # Create a basic fallback result
            fallback = {
                "identified_terms": ["medical term"],
                "term_explanations": {"medical term": "plain language explanation"},
                "clarified_summary": medical_text
            }
            self.add_to_memory(fallback, "terminology_interpretation_fallback")
            return fallback


class ContentIntegratorAgent(BaseAgent):
    """
    Content Integrator agent responsible for merging simplified
    content from different perspectives (for synchronous strategy).
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Content Integrator agent."""
        super().__init__(agent_id, AgentRole.CONTENT_INTEGRATOR, model_key)

    def integrate_content(
        self,
        original_text: str,
        simplified_text: str,
        metaphorical_text: str,
        terminology_text: str,
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate content from different simplification perspectives.

        Args:
            original_text: Original medical text
            simplified_text: Text from Content Simplifier
            metaphorical_text: Text from Metaphor Analyst
            terminology_text: Text from Terminology Interpreter
            guideline: Simplification guideline

        Returns:
            Dictionary containing integrated content
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in integrating different perspectives of simplified medical content. "
                "Your task is to merge different versions of a medical lay summary into a cohesive, "
                "accessible single paragraph that incorporates the strengths of each approach. Your response should "
                "be in JSON format and include:\n"
                "1. 'integrated_summary': The cohesive single-paragraph summary combining the best elements\n"
                "2. 'integration_approach': Brief explanation of how you combined the different versions\n"
            )
        }

        # Convert guideline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])

        user_message = {
            "role": "user",
            "content": (
                f"Please integrate the following versions of a medical lay summary "
                f"into a single cohesive paragraph according to the simplification guideline:\n\n"
                f"ORIGINAL TEXT:\n{original_text}\n\n"
                f"BASIC SIMPLIFIED VERSION:\n{simplified_text}\n\n"
                f"METAPHORICAL VERSION:\n{metaphorical_text}\n\n"
                f"TERMINOLOGY-EXPLAINED VERSION:\n{terminology_text}\n\n"
                f"GUIDELINE:\n{guideline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Content Integrator merged different simplification perspectives")
            self.add_to_memory(result, "content_integration")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Content Integrator response as JSON")
            # Create a basic fallback result
            fallback = {
                "integrated_summary": simplified_text,
                "integration_approach": "Basic integration of simplified versions"
            }
            self.add_to_memory(fallback, "content_integration_fallback")
            return fallback


class ArticleArchitectAgent(BaseAgent):
    """
    Article Architect agent responsible for ensuring the
    simplified content has a clear and coherent structure.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Article Architect agent."""
        super().__init__(agent_id, AgentRole.ARTICLE_ARCHITECT, model_key)

    def construct_summary(
        self,
        simplified_texts: List[str],
        guideline: Dict[str, Any],
        outline: Dict[str, Any],
        strategy: ConstructionStrategy = ConstructionStrategy.DIRECT
    ) -> Dict[str, Any]:
        """
        Construct a coherent single-paragraph summary from simplified text chunks.

        Args:
            simplified_texts: List of simplified text chunks
            guideline: Simplification guideline
            outline: Key information outline
            strategy: Construction strategy (Direct or Iterative)

        Returns:
            Dictionary containing the constructed summary
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in structuring medical content for general audiences. "
                "Your task is to organize simplified medical text chunks into a cohesive, "
                "well-structured single paragraph that follows a logical flow. Ensure the summary "
                "adheres to the provided guideline and covers the key information identified. Your response "
                "should be in JSON format and include:\n"
                "1. 'lay_summary': The complete, cohesive single-paragraph lay summary\n"
                "2. 'construction_approach': Brief explanation of how you structured the content\n"
            )
        }

        # Convert guideline and outline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])
        outline_text = json.dumps(outline, indent=2)

        # Format the text chunks based on the construction strategy
        if strategy == ConstructionStrategy.DIRECT:
            # Combine all text chunks at once
            combined_chunks = "\n\n".join(simplified_texts)
            strategy_text = "Direct construction (combine all chunks at once)"
        else:
            # Present chunks sequentially for iterative construction
            combined_chunks = "\n\n--- CHUNK SEPARATOR ---\n\n".join(simplified_texts)
            strategy_text = "Iterative construction (combine chunks sequentially)"

        user_message = {
            "role": "user",
            "content": (
                f"Please construct a cohesive single-paragraph lay summary from the following simplified text chunks "
                f"using a {strategy_text} approach. Follow the provided guideline and cover the key information:\n\n"
                f"SIMPLIFIED TEXT CHUNKS:\n{combined_chunks}\n\n"
                f"GUIDELINE:\n{guideline_text}\n\n"
                f"KEY INFORMATION:\n{outline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Article Architect constructed single-paragraph summary using {strategy.value} strategy")
            self.add_to_memory(result, "summary_construction")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Article Architect response as JSON")
            # Create a basic fallback result
            fallback = {
                "lay_summary": "\n\n".join(simplified_texts),
                "construction_approach": f"Basic {strategy.value} construction"
            }
            self.add_to_memory(fallback, "summary_construction_fallback")
            return fallback


class ProofreaderAgent(BaseAgent):
    """
    Proofreader agent responsible for the final review and
    refinement of the simplified medical content.
    """

    def __init__(
        self,
        agent_id: str,
        model_key: str = "deepseek-v3-official"
    ):
        """Initialize the Proofreader agent."""
        super().__init__(agent_id, AgentRole.PROOFREADER, model_key)

    def proofread(
        self,
        original_text: str,
        simplified_summary: str,
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform final proofreading of the simplified summary.

        Args:
            original_text: Original medical text
            simplified_summary: Simplified summary to proofread
            guideline: Simplification guideline

        Returns:
            Dictionary containing the proofread summary
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an expert medical editor specializing in proofreading simplified medical content. "
                "Your task is to review the simplified single-paragraph lay summary for accuracy, clarity, "
                "consistency, and adherence to the simplification guideline. Check for grammar, spelling, flow, "
                "and ensure medical accuracy is maintained while keeping the language accessible to general audiences. "
                "Your response should be in JSON format and include:\n"
                "1. 'issues_identified': List of issues found (accuracy, clarity, grammar, etc.)\n"
                "2. 'corrections_made': Description of corrections applied\n"
                "3. 'final_summary': The final, proofread single-paragraph lay summary\n"
            )
        }

        # Convert guideline to formatted text
        guideline_text = "\n".join([f"{k}: {v}" for k, v in guideline.items()])

        user_message = {
            "role": "user",
            "content": (
                f"Please proofread the following single-paragraph lay summary. "
                f"Ensure it accurately reflects the original text while maintaining "
                f"accessibility according to the guideline:\n\n"
                f"ORIGINAL TEXT:\n{original_text}\n\n"
                f"LAY SUMMARY:\n{simplified_summary}\n\n"
                f"GUIDELINE:\n{guideline_text}"
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Proofreader completed final review of lay summary")
            self.add_to_memory(result, "proofreading")
            return result
        except json.JSONDecodeError:
            print("Failed to parse Proofreader response as JSON")
            # Create a basic fallback result
            fallback = {
                "issues_identified": ["No major issues identified"],
                "corrections_made": "Minor grammar and clarity improvements",
                "final_summary": simplified_summary
            }
            self.add_to_memory(fallback, "proofreading_fallback")
            return fallback


class AgentSimpCoordinator:
    """
    Coordinator for the AgentSimp medical literature simplification process.
    Manages the workflow and communication between different agents.
    """

    def __init__(
        self,
        model_key: str = "deepseek-v3-official",
        communication_strategy: CommunicationStrategy = CommunicationStrategy.PIPELINE,
        construction_strategy: ConstructionStrategy = ConstructionStrategy.DIRECT
    ):
        """
        Initialize the AgentSimp coordinator.

        Args:
            model_key: LLM model to use for all agents
            communication_strategy: Strategy for agent communication (Pipeline or Synchronous)
            construction_strategy: Strategy for document construction (Direct or Iterative)
        """
        self.model_key = model_key
        self.communication_strategy = communication_strategy
        self.construction_strategy = construction_strategy

        # Initialize agents
        self.project_director = ProjectDirectorAgent("director", model_key)
        self.structure_analyst = StructureAnalystAgent("analyst", model_key)
        self.content_simplifier = ContentSimplifierAgent("simplifier", model_key)
        self.simplify_supervisor = SimplifySupervisorAgent("supervisor", model_key)
        self.metaphor_analyst = MetaphorAnalystAgent("metaphor", model_key)
        self.terminology_interpreter = TerminologyInterpreterAgent("terminology", model_key)
        self.content_integrator = ContentIntegratorAgent("integrator", model_key)
        self.article_architect = ArticleArchitectAgent("architect", model_key)
        self.proofreader = ProofreaderAgent("proofreader", model_key)

        print(f"Initialized AgentSimp coordinator with {communication_strategy.value} "
              f"communication and {construction_strategy.value} construction strategies")

    def simplify_document(self, medical_text: str, chunk_size: int = 2000) -> Dict[str, Any]:
        """
        Simplify a medical document into a single-paragraph lay summary.

        Args:
            medical_text: Medical text to be simplified
            chunk_size: Maximum size for text chunks (characters)

        Returns:
            Dictionary containing the simplified document and process details
        """
        start_time = time.time()
        process_log = {
            "original_text": medical_text,
            "communication_strategy": self.communication_strategy.value,
            "construction_strategy": self.construction_strategy.value,
            "steps": []
        }

        # Step 1: Overall Planning
        print("Step 1: Overall Planning")

        # 1.1: Project Director creates simplification guideline
        print("1.1: Creating simplification guideline")
        guideline = self.project_director.create_simplification_guideline(medical_text)
        process_log["steps"].append({
            "step": "1.1",
            "agent": "Project Director",
            "action": "Create Guideline",
            "output": guideline
        })

        # 1.2: Structure Analyst identifies key information
        print("1.2: Identifying key information")
        key_info = self.structure_analyst.create_structural_outline(medical_text)
        process_log["steps"].append({
            "step": "1.2",
            "agent": "Structure Analyst",
            "action": "Identify Key Information",
            "output": key_info
        })

        # Step 2: Initial Content Simplification
        print("Step 2: Initial Content Simplification")
        simplified_result = self.content_simplifier.simplify_content(medical_text, guideline, key_info)
        lay_summary = simplified_result.get("lay_summary", "")

        if not isinstance(lay_summary, str):
            lay_summary = str(lay_summary)

        process_log["steps"].append({
            "step": "2",
            "agent": "Content Simplifier",
            "action": "Create Lay Summary",
            "output": simplified_result
        })

        # Step 3: Review and Refinement
        print("Step 3: Review and Refinement")

        # 3.1: Simplify Supervisor reviews and improves
        review_result = self.simplify_supervisor.review_simplification(medical_text, lay_summary, guideline)
        supervised_summary = review_result.get("revised_summary", lay_summary)

        if not isinstance(supervised_summary, str):
            supervised_summary = str(supervised_summary)

        process_log["steps"].append({
            "step": "3.1",
            "agent": "Simplify Supervisor",
            "action": "Review Lay Summary",
            "output": review_result
        })

        # 3.2: Metaphor Analyst enhances accessibility
        metaphor_result = self.metaphor_analyst.analyze_and_simplify(supervised_summary, guideline)
        metaphorical_summary = metaphor_result.get("enhanced_summary", supervised_summary)

        if not isinstance(metaphorical_summary, str):
            metaphorical_summary = str(metaphorical_summary)

        process_log["steps"].append({
            "step": "3.2",
            "agent": "Metaphor Analyst",
            "action": "Enhance Accessibility",
            "output": metaphor_result
        })

        # 3.3: Terminology Interpreter ensures plain language
        terminology_result = self.terminology_interpreter.interpret_terminology(metaphorical_summary, guideline)
        interpreted_summary = terminology_result.get("clarified_summary", metaphorical_summary)

        if not isinstance(interpreted_summary, str):
            interpreted_summary = str(interpreted_summary)

        process_log["steps"].append({
            "step": "3.3",
            "agent": "Terminology Interpreter",
            "action": "Ensure Plain Language",
            "output": terminology_result
        })

        # Step 4: Final Review
        print("Step 4: Final Review")
        proofread_result = self.proofreader.proofread(medical_text, interpreted_summary, guideline)
        final_summary = proofread_result.get("final_summary", interpreted_summary)

        if not isinstance(final_summary, str):
            final_summary = str(final_summary)

        process_log["steps"].append({
            "step": "4",
            "agent": "Proofreader",
            "action": "Final Review",
            "output": proofread_result
        })

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Return final result
        result = {
            "original_text": medical_text,
            "lay_summary": final_summary,
            "processing_time": processing_time,
            "process_log": process_log
        }

        print(f"Lay summary generation completed in {processing_time:.2f} seconds")
        return result


def process_dataset(
    dataset_name: str,
    model_key: str = "deepseek-v3-official",
    communication_strategy: CommunicationStrategy = CommunicationStrategy.PIPELINE,
    construction_strategy: ConstructionStrategy = ConstructionStrategy.DIRECT
) -> None:
    """
    Process a medical literature dataset and generate lay summaries.

    Args:
        dataset_name: Name of the dataset to process
        model_key: LLM model to use
        communication_strategy: Strategy for agent communication
        construction_strategy: Strategy for document construction
    """
    # Set up file paths
    input_path = f"my_datasets/processed/laysummary/{dataset_name}/test.json"
    output_dir = f"logs/laysummary/{dataset_name}/AgentSimp"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    try:
        dataset = load_json(input_path)
        print(f"Loaded {len(dataset)} samples from {input_path}")
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return

    # Initialize coordinator
    coordinator = AgentSimpCoordinator(
        model_key=model_key,
        communication_strategy=communication_strategy,
        construction_strategy=construction_strategy
    )

    # Process each sample
    for sample in tqdm(dataset, desc=f"Processing {dataset_name}"):
        sample_id = sample.get("id")
        source_text = sample.get("source", "")
        target_text = sample.get("target", "")

        # Skip if already processed
        output_path = f"{output_dir}/laysummary_{sample_id}-result.json"
        if os.path.exists(output_path):
            print(f"Skipping sample {sample_id} - already processed")
            continue

        try:
            print(f"Processing sample {sample_id}")

            # Generate lay summary
            result = coordinator.simplify_document(source_text)

            # Prepare output with just the lay summary as a string
            output = {
                "id": sample_id,
                "source": source_text,
                "target": target_text,
                "pred": result["lay_summary"],  # Store just the lay summary as a string
                "metadata": {
                    "model": model_key,
                    "communication_strategy": communication_strategy.value,
                    "construction_strategy": construction_strategy.value,
                    "processing_time": result["processing_time"]
                }
            }

            # Save result
            save_json(output, output_path)
            print(f"Saved result for sample {sample_id}")

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Medical Literature Lay Summary Generation using AgentSimp")
    parser.add_argument("--dataset", type=str, required=True, choices=["PLABA", "cochrane", "elife", "med_easi", "plos_genetics"],
                       help="Dataset to process")
    parser.add_argument("--model", type=str, default="deepseek-v3-official", help="LLM model to use")
    parser.add_argument("--communication", type=str, default="pipeline", choices=["pipeline", "synchronous"],
                       help="Communication strategy")
    parser.add_argument("--construction", type=str, default="direct", choices=["direct", "iterative"],
                       help="Construction strategy")

    args = parser.parse_args()

    # Parse arguments
    dataset_name = args.dataset
    model_key = args.model
    communication_strategy = CommunicationStrategy.PIPELINE if args.communication == "pipeline" else CommunicationStrategy.SYNCHRONOUS
    construction_strategy = ConstructionStrategy.DIRECT if args.construction == "direct" else ConstructionStrategy.ITERATIVE

    # Process dataset
    process_dataset(
        dataset_name=dataset_name,
        model_key=model_key,
        communication_strategy=communication_strategy,
        construction_strategy=construction_strategy
    )


if __name__ == "__main__":
    main()