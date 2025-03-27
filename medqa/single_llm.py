from openai import OpenAI
import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from utils import LLM_MODELS_SETTINGS
# nohup python -m medqa.qa_model --dataset MedQA --prompt_type few_shot > MedQA_few_shot.log 2>&1 &

def zero_shot_prompt(question: str, options: dict):
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Please respond only with a single selected option's letter, like A, B, C... \n"
    return prompt

def few_shot_prompt(dataset: str, question: str, options: dict):
    if dataset == "MedQA":
        examples = "Example 1: Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? \n" \
            "Options: [A: Ampicillin, B: Ceftriaxone, C: Ciprofloxacin, D: Doxycycline, E: Nitrofurantoin] \n" \
            "Answer: E \n" \
            "Example 2: Question: A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?\n" \
            "Options: [A: Placing the infant in a supine position on a firm mattress while sleeping, B: Routine postnatal electrocardiogram (ECG), C: Keeping the infant covered and maintaining a high room temperature, D: Application of a device to maintain the sleeping position, E: Avoiding pacifier use during sleep] \n" \
            "Answer: A \n" \
            
    elif dataset == "PubMedQA":
        examples = "Example 1: Question: The reduced use of sugars-containing (SC) liquid medicines has increased the use of other dose forms, potentially resulting in more widespread dental effects, including tooth wear. The aim of this study was to assess the erosive potential of 97 paediatric medicines in vitro. The study took the form of in vitro measurement of endogenous pH and titratable acidity (mmol). Endogenous pH was measured using a pH meter, followed by titration to pH 7.0 with 0.1-M NaOH. Overall, 55 (57%) formulations had an endogenous pH of<5.5. The mean (+/- SD) endogenous pH and titratable acidity for 41 SC formulations were 5.26 +/- 1.30 and 0.139 +/- 0.133 mmol, respectively; for 56 sugars-free (SF) formulations, these figures were 5.73 +/- 1.53 and 0.413 +/- 1.50 mmol (P>0.05). Compared with their SC bioequivalents, eight SF medicines showed no significant differences for pH or titratable acidity, while 15 higher-strength medicines showed lower pH (P = 0.035) and greater titratable acidity (P = 0.016) than their lower-strength equivalents. Chewable and dispersible tablets (P<0.001), gastrointestinal medicines (P = 0.002) and antibiotics (P = 0.007) were significant predictors of higher pH. In contrast, effervescent tablets (P<0.001), and nutrition and blood preparations (P = 0.021) were significant predictors of higher titratable acidity. Are sugars-free medicines more erosive than sugars-containing medicines? \n" \
            "Options: [A: Yes, B: No, C: Maybe] \n" \
            "Answer: B \n" \
            "Example 2: Question: This investigation assesses the effect of platelet-rich plasma (PRP) gel on postoperative pain, swelling, and trismus as well as healing and bone regeneration potential on mandibular third molar extraction sockets. A prospective randomized comparative clinical study was undertaken over a 2-year period. Patients requiring surgical extraction of a single impacted third molar and who fell within the inclusion criteria and indicated willingness to return for recall visits were recruited. The predictor variable was application of PRP gel to the socket of the third molar in the test group, whereas the control group had no PRP. The outcome variables were pain, swelling, and maximum mouth opening, which were measured using a 10-point visual analog scale, tape, and millimeter caliper, respectively. Socket healing was assessed radiographically by allocating scores for lamina dura, overall density, and trabecular pattern. Quantitative data were presented as mean. Mann-Whitney test was used to compare means between groups for continuous variables, whereas Fischer exact test was used for categorical variables. Statistical significance was inferred at P<.05. Sixty patients aged 19 to 35 years (mean: 24.7 \u00b1 3.6 years) were divided into both test and control groups of 30 patients each. The mean postoperative pain score (visual analog scale) was lower for the PRP group at all time points and this was statistically significant (P<.05). Although the figures for swelling and interincisal mouth opening were lower in the test group, this difference was not statistically significant. Similarly, the scores for lamina dura, trabecular pattern, and bone density were better among patients in the PRP group. This difference was also not statistically significant. Can autologous platelet-rich plasma gel enhance healing after surgical extraction of mandibular third molars? \n" \
            "Options: [A: Yes, B: No, C: Maybe] \n" \
            "Answer: A \n"
        
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Please respond only with [the selected option's letter, like A, B, C...\n" \
        f"Here are some examples for you reference: '''{examples}''' "
        
    return prompt
    
def cot_prompt(question: str, options: dict):
    response_field = f"'Thought': [the step-by-step reasoning behind your answer] \n" \
        f"'Option': [the letter of the selected option, like A, B, C, D]" \

    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. " \
        f"Please give your answer in JSON format with the following two fields: '''{response_field}''' "
    return prompt

def inference(datatset, question, options, model_key, prompt_type):
    
    response_format = None
    n_samples = 1
    
    if prompt_type == "zero_shot":
        prompt = zero_shot_prompt(question, options)
    elif prompt_type == "few_shot":
        prompt = few_shot_prompt(dataset, question, options)
    elif prompt_type == "cot":
        prompt = cot_prompt(question, options)
        response_format = {"type": "json_object"}
    elif prompt_type == "cot-sc":
        prompt = cot_prompt(question, options)
        response_format = {"type": "json_object"}
        n_samples = 5
    
    model_settings = LLM_MODELS_SETTINGS[model_key]
    client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
    
    response = client.chat.completions.create(
                    model=model_settings["model_name"],
                    messages=[
                        {"role": "system", "content": "You will be answering a medical MCQ question."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=response_format,
                    n = n_samples,
                    stream=False
                )
    
    if prompt_type == "cot-sc":
        answers = [response.choices[i].message.content for i in range(n_samples)]
        options = [json.loads(answers[i])["Option"] for i in range(n_samples)]
        counter = Counter(options)
        
        return counter.most_common(1)[0][0] # return the most common option only
    
    else:
        return responses[0].choices[0].message.content
    
if __name__ == "__main__":
    # get the dataset and prompt type from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use for testing")
    parser.add_argument("--prompt_type", type=str, help="Prompt type to use for testing")
    parser.add_argument("--model_name", type=str, help="Model name to use for inference")
    parser.add_argument("--start_pos", type=int, help="Starting position of the file")
    parser.add_argument("--end_pos", type=int, help="Ending position of the file")
    args = parser.parse_args()
    
    dataset = args.dataset # ["MedQA", "PubMedQA"]
    prompt_type = args.prompt_type # ["zero_shot", "few_shot", "cot"]
    model_name = args.model_name
    start_pos = args.start_pos
    end_pos = args.end_pos
    
    data_path = f"./cleaned_datasets/{dataset.lower()}.json"
    output_path = f"./output/{dataset}/{prompt_type}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output file created at: {output_path}")
    
    # load the data
    with open(data_path, "r") as file:
        data = [json.loads(line) for line in file]
    
    # get the data based on the start and end position
    data = data[start_pos:end_pos] if end_pos != -1 else data[start_pos:]
    total_lines = len(data)
        
    with open(output_path, "a") as output_file:
        
        for datum in tqdm(data, total=total_lines, desc=f"Testing on {dataset} with {prompt_type}"):
            model_answer = inference(dataset, datum["question"], datum["options"], model_name, prompt_type)
            
            output = {
                "qid": datum["qid"],
                "question": datum["question"],
                "ground_truth": datum["answer"],
                "model_answer": model_answer
            }
            
            json.dump(output, output_file)
            output_file.write("\n")