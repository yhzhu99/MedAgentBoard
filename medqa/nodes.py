from crewai import Crew, Process
from medqa.crew.tasks import MedicalTasks
from medqa.crew.agents import MedQAAgents
from medqa.state import AnswerState
# from medqa.utils import log_state_change
from typing import Dict, Any
import copy

def initial_answers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    tasks = MedicalTasks(agents).initial_answer_task() # Each task corresponds to a different doctor agent
    
    initial_answers = []
    # Generate the initial answers for each doctor agent
    for task in tasks:
        task.description = task.description.format(question=state['question'])
    
        answer_crew = Crew(
            agents=[task.agent], 
            tasks=[task]
        )
        
        answer_crew.kickoff() # store the output of each task in corresponding task object
        initial_answers.append(task.output.raw)
        
    # Log the state change to track the flow of the multi-agent collaboration
    current_round = state['round'] + 1
    new_log = copy.deepcopy(state['log'])
    new_log["Question"] = state['question']
    new_log[f"Round {current_round}"] = {
        'question_prompts': [task.description for task in tasks],
        'answers': initial_answers
    }                                   
    # Unpack the current state and update the answers and round number
    new_state = {
        **state,
        'current_answers': initial_answers,
        'round': current_round,
        'log': new_log
    }
    
    return new_state
    

def check_consensus_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    
    consensus_task = MedicalTasks(agents).consensus_check_task(state['current_answers'])
    
    consensus_crew = Crew(
        agents=[consensus_task.agent],
        tasks=[consensus_task]
    )
    
    consensus_crew.kickoff()
    print(f"The output from the meta_agent is: {consensus_task.output.raw}")
    consensus = consensus_task.output.raw
    
    current_round = state['round']
    new_log = copy.deepcopy(state['log'])
    new_log[f"Round {current_round}"].update({
        'consensus_prompt': consensus_task.description,
        'consensus': consensus,
    })
    
    new_state = {
        **state,
        'consensus_reached': consensus,
        'log': new_log
    }
    
    return new_state

def generate_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    
    # Generate feedback task for the meta agent
    feedback_task = MedicalTasks(agents).feedback_task(state)
    
    feedback_crew = Crew(
        agents=[feedback_task.agent],
        tasks=[feedback_task]
    )
    
    feedback_crew.kickoff()
    feedback = feedback_task.output.raw
    
    current_round = state['round']
    new_log = copy.deepcopy(state['log'])
    new_log[f"Round {current_round}"].update({
        'feedback_prompt': feedback_task.description,
        'feedback': feedback
    })
    
    new_state = {
        **state,
        'previous_answers': state['current_answers'],
        'feedback': feedback,
        'log': new_log
    }
    
    return new_state

def refine_answers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    tasks = MedicalTasks(agents).refinement_task(state['question'],state['previous_answers'],state['feedback'])
    
    refined_answers = []
    for task in tasks:
        
        refine_crew = Crew(
            agents=[task.agent],
            tasks=[task]
        )
        
        refine_crew.kickoff()
        refined_answers.append(task.output.raw)
    
    new_round = state['round'] + 1
    new_log = copy.deepcopy(state['log'])
    new_log[f"Round {new_round}"] = {
        'question_prompts': [task.description for task in tasks],
        'answers': refined_answers
    }
    
    new_state = {
        **state,
        'current_answers': refined_answers,
        'round': new_round,
        'log': new_log
    }
    
    return new_state

def final_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    
    # Generate the finalized and consensually agreed answer
    if state['consensus_reached']:
        task_description = (
            "Based on the final consensus from the 3 doctors:\n\n" +
            "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['current_answers'])]) +
            "\n\nSimply give the final choice without any additional explanation, e.g., 'A'"
        )
        
    # Summarize all perspectives if consensus was not reached and choose the final answer by the majority vote
    else:
        task_description = (
            "Based on all perspectives after multiple rounds of discussion:\n\n" +
            "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['current_answers'])]) +
            "\n\nIf there is a clear majority answer/choice among the doctors when disagreement persists among some doctors, select it as the final answer and simply give the final answer/choice without any additional explanation, e.g., 'A'"
        )
    
    final_task = MedicalTasks(agents).output_task(task_description)
    
    final_crew = Crew(
        agents=[final_task.agent],
        tasks=[final_task]
    )
    
    final_crew.kickoff()
    final_answer = final_task.output.raw
    
    new_log = copy.deepcopy(state['log'])
    new_log[f"Final"] = {
        'final_prompt': final_task.description,
        "final_answer": final_answer
    }
    
    new_state = {
        **state,
        'final_answer': final_answer,
        'log': new_log
    }
    
    return new_state