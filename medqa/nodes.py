from crewai import Crew, Process
from medqa.crew.tasks import MedicalTasks
from medqa.crew.agents import MedQAAgents
from medqa.crew.tools import ConsensusTool
from medqa.state import AnswerState
from medqa.utils import log_state_change
from typing import Dict, Any

def generate_answers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    tasks = MedicalTasks(agents).initial_answer_task() # Each task corresponds to a different doctor agent
    
    for task in tasks:
        task.description = task.description.format(question=state['question'])
    
    # Sequentially processing of tasks by all doctor agents
    crew = Crew(
        agents=[t.agent for t in tasks], 
        tasks=tasks,
        process=Process.sequential
    )
    
    # results = crew.kickoff()  # This results is not a list of results, but a single result from the last agent, check the Crew class
    # answers = [result.raw_output for result in results] 
    
    crew.kickoff() # store the output of each task in corresponding task object
    answers = [task.output.raw for task in tasks]

    # Unpack the current state and update the answers and round number
    new_state = {
        **state,
        'answers': answers,
        'round': state['round'] + 1
    }
    
    log_state_change(state, new_state) # Log the state change to track the flow of the multi-agent collaboration
    
    return new_state
    

def check_consensus_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tool = ConsensusTool()
    consensus = tool.check_consensus(state['answers'])
    
    new_state = {
        **state,
        'consensus_reached': consensus
    }
    
    log_state_change(state, new_state)
    
    return new_state

def generate_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedicalAgents()
    meta_agent = agents.meta_agent()
    
    # Generate feedback task for the meta agent
    feedback_task = MedicalTasks(agents).feedback_task()
    
    feedback_task_crew = Crew(
        agents=[feedback_task.agent],
        tasks=[feedback_task]
    )
    
    feedback_task_crew.kickoff()
    feedback = feedback_task.output.raw
    
    new_state = {
        **state,
        'feedback': state['feedback'] + [feedback]
    }
    
    log_state_change(state, new_state)
    
    return new_state

def refine_answers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedicalAgents()
    tasks = MedicalTasks(agents).refinement_task(state['feedback'][-1])
    
    crew = Crew(
        agents=[t.agent for t in tasks],
        tasks=tasks,
        process=Process.sequential
    )
    
    results = crew.kickoff()
    answers = [result.raw_output for result in results]
    
    new_state = {
        **state,
        'answers': answers,
        'round': state['round'] + 1
    }
    
    log_state_change(state, new_state)
    
    return new_state

def final_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agents = MedQAAgents()
    meta_agent = agents.meta_agent()
    
    # Generate the finalized and consensually agreed answer
    if state['consensus_reached']:
        task_description = (
            "Summarize final consensus from these answers:\n\n" +
            "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['answers'])]) +
            "\n\nMake sure the output is simply the final answer/choice without any additional explanation."
        )
        
    # Summarize all perspectives if consensus was not reached and choose the final answer by the majority vote
    else:
        task_description = (
            "Summarize all perspectives after multiple rounds of discussion:\n\n" +
            "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['answers'])]) +
            "\n\nIf there is a clear majority opinion among the doctors, select it as the final answer. "
            "Otherwise, provide a comprehensive summary of all perspectives."
        )
    
    final_task = MedicalTasks(agents).output_task(task_description)
    
    final_task_crew = Crew(
        agents=[final_task.agent],
        tasks=[final_task]
    )
    
    final_task_crew.kickoff()
    final_answer = final_task.output.raw
    
    new_state = {
        **state,
        'final_answer': final_answer
    }
    
    return new_state