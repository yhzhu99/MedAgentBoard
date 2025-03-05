import logging
from medqa.graph import MedQAGraph

# Set up logging to track the flow of the multi-agent collaboration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Run the inference: python -m medqa.inference
question = '''
There is only one correct option in the following question:
"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?
"options": {"A": "Inhibition of thymidine synthesis", "B": "Inhibition of proteasome", "C": "Hyperstabilization of microtubules", "D": "Generation of free radicals", "E": "Cross-linking of DNA"}
'''
qa_graph = MedQAGraph(question, max_rounds=5)
final_answer = qa_graph.run()
print(f"Final Answer:\n{final_answer}")