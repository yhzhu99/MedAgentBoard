"""
medagentboard/laysummary/single_llm.py

Single LLM approach for medical literature lay summary generation.
Implements three prompting settings:
- Basic: Simple direct prompt
- Optimized: Detailed prompt with guidelines
- Few-shot: Optimized prompt plus examples

Works with medical literature datasets to generate accessible lay summaries.
"""

from openai import OpenAI
import os
import json
import argparse
import time
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


class SingleLLMLaySummary:
    """
    Class for generating lay summaries of medical literature using a single LLM.
    """

    def __init__(self, model_key: str = "deepseek-v3-official"):
        """
        Initialize the LLM for lay summary generation.

        Args:
            model_key: Key identifying the model in LLM_MODELS_SETTINGS
        """
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
        print(f"Initialized SingleLLMLaySummary with model: {model_key}")

        # Few-shot examples for each dataset (placeholders)
        self.few_shot_examples = {
            "PLABA": [
    {
    "source": "Exercise-Associated Muscle Cramps (EAMC) are a common painful condition of muscle spasms. Despite scientists tried to understand the physiological mechanism that underlies these common phenomena, the etiology is still unclear. From 1900 to nowadays, the scientific world retracted several times the original hypothesis of heat cramps. However, recent literature seems to focus on two potential mechanisms: the dehydration or electrolyte depletion mechanism, and the neuromuscular mechanism. The aim of this review is to examine the recent literature, in terms of physiological mechanisms of EAMC. A comprehensive search was conducted on PubMed and Google Scholar. The following terminology was applied: muscle cramps, neuromuscular hypothesis (or thesis), dehydration hypothesis, Exercise-Associated muscle cramps, nocturnal cramps, muscle spasm, muscle fatigue. From the initial literature of 424 manuscripts, sixty-nine manuscripts were included, analyzed, compared and summarized. Literature analysis indicates that neuromuscular hypothesis may prevails over the initial hypothesis of the dehydration as the trigger event of muscle cramps. New evidence suggests that the action potentials during a muscle cramp are generated in the motoneuron soma, likely accompanied by an imbalance between the rising excitatory drive from the muscle spindles (Ia) and the decreasing inhibitory drive from the Golgi tendon organs. In conclusion, from the latest investigations there seem to be a spinal involvement rather than a peripheral excitation of the motoneurons.",
    "target": "Exercise-Associated Muscle Cramps (EAMC) are a common type of muscle spasm, in which a muscle continually contracts without intention, causing pain. Scientists have tried to explain why these cramps happen, but have not been able to. An idea commonly returned to throughout the years is that EAMCs may be caused by heat. Recently, though, more likely explanations are thought to include dehydration, lack of electrolytes, or issues with the nerves connecting to the muscles. The aim of this review is to look at recent research into how and why EAMCs happen. We searched common online resources for papers. For search terms, we used: muscle cramps, neuromuscular hypothesis (or thesis), dehydration hypothesis, Exercise-Associated muscle cramps, nocturnal cramps, muscle spasm, muscle fatigue. The search returned 424 papers. We read and analyzed 69 of them. From the latest evidence, interactions of nerves with muscles explains muscle cramps better than dehydration. Muscle contraction is normally balanced by special cells in the muscles that sense how stretched or contracted the muscles are. Recent findings suggest malfunctions in these sensor cells result in spinal nerves sending unnecessary signals for the muscles to contract. In summary, the signal causing muscles to contract during a spasm seems to come from the spine rather than the nerve endings within the muscles."
  },
  {
    "source": "Background: Muscle cramp is a painful, involuntary muscle contraction, and that occurs during or following exercise is referred to as exercise-associated muscle cramp (EAMC). The causes of EAMC are likely to be multifactorial, but dehydration and electrolytes deficits are considered to be factors. This study tested the hypothesis that post-exercise muscle cramp susceptibility would be increased with spring water ingestion, but reduced with oral rehydration solution (ORS) ingestion during exercise. Methods: Ten men performed downhill running (DHR) in the heat (35-36 °C) for 40-60 min to reduce 1.5-2% of their body mass in two conditions (spring water vs ORS) in a cross-over design. The body mass was measured at 20 min and every 10 min thereafter during DHR, and 30 min post-DHR. The participants ingested either spring water or ORS for the body mass loss in each period. The two conditions were counter-balanced among the participants and separated by a week. Calf muscle cramp susceptibility was assessed by a threshold frequency (TF) of an electrical train stimulation to induce cramp before, immediately after, 30 and 65 min post-DHR. Blood samples were taken before, immediately after and 65 min after DHR to measure serum sodium, potassium, magnesium and chroride concentrations, hematocrit (Hct), hemoglobin (Hb), and serum osmolarity. Changes in these varaibles over time were compared between conditions by two-way repeated measures of analysis of variance. Results: The average (±SD) baseline TF (25.6 ± 0.7 Hz) was the same between conditions. TF decreased 3.8 ± 2.7 to 4.5 ± 1.7 Hz from the baseline value immediately to 65 min post-DHR for the spring water condition, but increased 6.5 ± 4.9 to 13.6 ± 6.0 Hz in the same time period for the ORS condition (P < 0.05). Hct and Hb did not change significantly (P > 0.05) for both conditions, but osmolarity decreased (P < 0.05) only for the spring water condition. Serum sodium and chloride concentrations decreased (< 2%) at immediately post-DHR for the spring water condition only (P < 0.05). Conclusions: These results suggest that ORS intake during exercise decreased muscle cramp susceptibility. It was concluded that ingesting ORS appeared to be effective for preventing EAMC.",
    "target": "Muscle cramps are unconscious contractions of muscles that are painful. When they happen during or after excercise, they are called Exercise-Associated Muscle Cramps (EAMC). There are probably many causes of EAMC, but dehydration and lack of electrolytes (mineral salts dissolved in the blood) are thought to play roles. This study tested whether drinking a special drink called Oral Rehydration Solution (ORS) while exercising made Exercise-Associated Muscle Cramps less likely than drinking spring water. Ten men ran downhill in hot conditions for 40-60 minutes, drinking either spring water or ORS. Their weights were measured 20 minutes into the exercise, then every 10 minutes, and then 30 minutes after the exercise was completed. After each measurement, the men drank enough of either the spring water or ORS to make up for the weight they lost. We did the experiment twice, a week apart, so that each person tried both drinks. We balanced which order they tried the drinks in.   Muscle cramps were easier to induce in participants who had drunk spring water than in participants who had drunk ORS. Some electrolytes in the blood decreased for participants who were drinking only spring water. The study showed that the drinking ORS (Oral Rehydration Solution) during exercise made muscle cramps less likely."
  },
            ],
            "cochrane": [
    {
    "source": "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers.",
    "target": "We found two small studies that presented data for 49 participants with arterial leg ulcers (search conducted January 2019). The studies also included participants with other kinds of ulcers, and it is not clear what proportion of participants were diabetic. Neither study described the methods fully, both presented limited results for the arterial ulcer participants, and one study did not provide information on the number of participants with an arterial ulcer in the control group. The follow-up periods (six and eight weeks) were too short to measure healing. Therefore, the data that were available were incomplete and cannot be generalised to the greater population of people who suffer from arterial leg ulcers. One study randomised participants to either 2% ketanserin ointment in polyethylene glycol (PEG) or PEG alone, administered twice a day over eight weeks. This study reported increased wound healing in the ketanserin group, when compared with the control group. It should be noted that ketanserin is not licensed for use in humans in all countries. The second study randomised participants to either topically-applied growth factors isolated from the participant's own blood (concentrated growth factors (CGF)), or standard dressing; both applied weekly for six weeks. This study reported that 66.6% of CGF-treated diabetic arterial ulcers showed more than a 50% decrease in ulcer size, compared to 6.7% of non-healing ulcers treated with standard dressing. Only one study mentioned side effects, and reported that no participant experienced side effects during follow-up (six weeks). Neither of the two studies reported time to ulcer healing, patient satisfaction or quality of life measures. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers. We downgraded the overall certainty of the available evidence to 'very low' and 'low', because the studies reported their methods poorly, there were only two studies and few participants with arterial disease, and because the studies were short and reported few results. This made it impossible to determine whether there was any real difference in the number of ulcers healed between the groups."
  },
  {
    "source": "We identified one RCT that involved 40 participants, and addressed the timing of surgery for people with recently symptomatic carotid artery stenosis. It compared very early surgery with surgery performed after 14 days of the last symptomatic event. The overall quality of the evidence was very low, due to the small number of participants from only one trial, and missing outcome data. We found no statistically significant difference between the effects of very early or delayed surgery in reducing the combined risk of stroke and death within 30 days of surgery (risk ratio (RR) 3.32; confidence interval (CI) 0.38 to 29.23; very low-quality evidence), or the combined risk of perioperative death and stroke (RR 0.47; CI 0.14 to 1.58; very low-quality evidence). To date, no results are available to confirm the optimal timing for surgery. There is currently no high-quality evidence available to support either very early or delayed cerebral revascularization after a recent ischemic stroke. Hence, further randomized trials to identify which patients should undergo very urgent revascularization are needed. Future studies should stratify participants by age group, sex, grade of ischemia, and degree of stenosis. Currently, there is one ongoing RCT that is examining the timing of cerebral revascularization.",
    "target": "The searches are up-to-date to 26 January 2016. We found only one randomized trial that assessed the effect of the timing of surgery. It included a total of 40 participants, ranging in age from 47 to 84 years. From the limited evidence available, we cannot tell if the timing of surgery is an important factor in determining the outcome for individuals with recent symptoms from carotid artery narrowing. There is not enough evidence on the best time for surgical treatment for people with recent symptoms from carotid artery narrowing. The overall quality of the evidence was very low, due to the small number of participants from only one trial and missing outcome data. Further studies with a larger number of patients are needed."
  },
            ],
            "elife": [
{
    "source": "In temperate climates, winter deaths exceed summer ones. However, there is limited information on the timing and the relative magnitudes of maximum and minimum mortality, by local climate, age group, sex and medical cause of death. We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions. Death rates in men and women \u2265 45 years peaked in December to February and were lowest in June to August, driven by cardiorespiratory diseases and injuries. In these ages, percent difference in death rates between peak and minimum months did not vary across climate regions, nor changed from 1980 to 2016. Under five years, seasonality of all-cause mortality largely disappeared after the 1990s. In adolescents and young adults, especially in males, death rates peaked in June/July and were lowest in December/January, driven by injury deaths.",
    "target": "In the USA, more deaths happen in the winter than the summer. But when deaths occur varies greatly by sex, age, cause of death, and possibly region. Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment. Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed, or whether existing ones are effective. Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death. Now, Parks et al. show that there are age and sex differences in which times of year most deaths occur. Parks et al. analyzed data on US deaths between 1980 and 2016. While overall deaths in a year were highest in winter and lowest in summer, a greater number of young men died during summer \u2013 mainly due to injuries \u2013 than during winter. Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller. Deaths among women and men aged 45 or older peaked between December and February \u2013 largely caused by respiratory and heart diseases, or injuries. Deaths in this older age group were lowest during the summer months. Death patterns in older people changed little over time. No regional differences were found in seasonal death patterns, despite large climate variation across the USA. The analysis by Parks et al. suggests public health and medical interventions have been successful in reducing seasonal deaths among many groups. But more needs to be done to address seasonal differences in deaths among older adults. For example, by boosting flu vaccination rates, providing warnings about severe weather and better insulation for homes. Using technology like hands-free communication devices or home visits to help keep vulnerable elderly people connected during the winter months may also help."
  },
  {
    "source": "Whether complement dysregulation directly contributes to the pathogenesis of peripheral nervous system diseases, including sensory neuropathies, is unclear. We addressed this important question in a mouse model of ocular HSV-1 infection, where sensory nerve damage is a common clinical problem. Through genetic and pharmacologic targeting, we uncovered a central role for C3 in sensory nerve damage at the morphological and functional levels. Interestingly, CD4 T cells were central in facilitating this complement-mediated damage. This same C3/CD4 T cell axis triggered corneal sensory nerve damage in a mouse model of ocular graft-versus-host disease (GVHD). However, this was not the case in a T-dependent allergic eye disease (AED) model, suggesting that this inflammatory neuroimmune pathology is specific to certain disease etiologies. Collectively, these findings uncover a central role for complement in CD4 T cell-dependent corneal nerve damage in multiple disease settings and indicate the possibility for complement-targeted therapeutics to mitigate sensory neuropathies.",
    "target": "Most people have likely experienced the discomfort of an eyelash falling onto the surface of their eye. Or that gritty sensation when dust blows into the eye and irritates the surface. These sensations are warnings from sensory nerves in the cornea, the transparent tissue that covers the iris and pupil. Corneal nerves help regulate blinking, and control production of the tear fluid that protects and lubricates the eye. But if the cornea suffers damage or infection, it can become inflamed. Long-lasting inflammation can damage the corneal nerves, leading to pain and vision loss. If scientists can identify how this happens, they may ultimately be able to prevent it. To this end, Royer et al. have used mice to study three causes of hard-to-treat corneal inflammation. The first is infection with herpes simplex virus (HSV-1), which also causes cold sores. The second is eye allergy, where the immune system overreacts to substances like pollen or pet dander. And the third is graft-versus-host disease (GVHD), an immune disorder that can affect people who receive a bone marrow transplant. Royer et al. showed that HSV-1 infection and GVHD \u2013 but not allergies \u2013 made the mouse cornea less sensitive to touch. Consistent with this, microscopy revealed damage to corneal nerves in the mice with HSV-1 infection and those with GVHD. Further experiments showed that immune cells called CD4 T cells and a protein called complement C3 were contributing to this nerve damage. Treating the mice with an experimental drug derived from cobra venom protected the cornea from the harmful effects of inflammation. It did so by blocking activation of complement C3 at the eye surface. Identifying factors such as complement C3 that are responsible for corneal nerve damage is an important first step in helping patients with inflammatory eye diseases. Many drugs that target the complement pathway are currently under development. Some of these drugs could potentially be adapted for delivery as eye drops. But first, experiments must test whether complement also contributes to corneal nerve damage in humans. If it does, work can then begin on testing these drugs for safety and efficacy in patients."
  },
            ],
            "med_easi": [
{
    "source": "75-90 % of the affected people have mild intellectual disability.",
    "target": "People with syndromic intellectual disabilities may have a `` typical look. ''"
  },
  {
    "source": "A bandlike tightness around the chest or abdomen, weakness, tingling, numbness of the feet and legs, and difficulty voiding develop over hours to a few days.",
    "target": "Within hours to a few days, tingling, numbness, and muscle weakness develop in the feet and move upward. Urinating becomes difficult, although some people feel an urgent need to urinate (urgency)."
  },
            ],
            "plos_genetics": [
{
    "source": "The guidance cue UNC-6/Netrin regulates both attractive and repulsive axon guidance. Our previous work showed that in C. elegans, the attractive UNC-6/Netrin receptor UNC-40/DCC stimulates growth cone protrusion, and that the repulsive receptor, an UNC-5: UNC-40 heterodimer, inhibits growth cone protrusion. We have also shown that inhibition of growth cone protrusion downstream of the UNC-5: UNC-40 repulsive receptor involves Rac GTPases, the Rac GTP exchange factor UNC-73/Trio, and the cytoskeletal regulator UNC-33/CRMP, which mediates Semaphorin-induced growth cone collapse in other systems. The multidomain flavoprotein monooxygenase (FMO) MICAL (Molecule Interacting with CasL) also mediates growth cone collapse in response to Semaphorin by directly oxidizing F-actin, resulting in depolymerization. The C. elegans genome does not encode a multidomain MICAL-like molecule, but does encode five flavin monooxygenases (FMO-1, -2, -3, -4, and 5) and another molecule, EHBP-1, similar to the non-FMO portion of MICAL. Here we show that FMO-1, FMO-4, FMO-5, and EHBP-1 may play a role in UNC-6/Netrin directed repulsive guidance mediated through UNC-40 and UNC-5 receptors. Mutations in fmo-1, fmo-4, fmo-5, and ehbp-1 showed VD/DD axon guidance and branching defects, and variably enhanced unc-40 and unc-5 VD/DD axon guidance defects. Developing growth cones in vivo of fmo-1, fmo-4, fmo-5, and ehbp-1 mutants displayed excessive filopodial protrusion, and transgenic expression of FMO-5 inhibited growth cone protrusion. Mutations suppressed growth cone inhibition caused by activated UNC-40 and UNC-5 signaling, and activated Rac GTPase CED-10 and MIG-2, suggesting that these molecules are required downstream of UNC-6/Netrin receptors and Rac GTPases. From these studies we conclude that FMO-1, FMO-4, FMO-5, and EHBP-1 represent new players downstream of UNC-6/Netrin receptors and Rac GTPases that inhibit growth cone filopodial protrusion in repulsive axon guidance.",
    "target": "Mechanisms that guide axons to their targets in the developing nervous system have been elucidated, but how these pathways affect behavior of the growth cone of the axon during outgrowth remains poorly understood. We previously showed that the guidance cue UNC-6/Netrin and its receptors UNC-40/DCC and UNC-5 inhibit lamellipodial and filopodial growth cone protrusion to mediate repulsion from UNC-6/Netrin in C. elegans. Here we report a new mechanism downstream of UNC-6/Netrin involving flavin monooxygenase redox enzymes (FMOs). We show that FMOs are normally required for axon guidance and to inhibit growth cone protrusion. Furthermore, we show that they are required for the anti-protrusive effects of activated UNC-40 and UNC-5 receptors, and that they can partially compensate for loss of molecules in the pathway, indicating that they act downstream of UNC-6/Netrin signaling. Based on the function of the FMO-containing MICAL molecules in Drosophila and vertebrates, we speculate that the FMOs might directly oxidize actin, leading to filament disassembly and collapse, and/or lead to the phosphorylation of UNC-33/CRMP, which we show also genetically interacts with the FMOs downstream of UNC-6/Netrin. In conclusion, this is the first evidence that FMOs might act downstream of UNC-6/Netrin signaling in growth cone protrusion and axon repulsion."
  },
  {
    "source": "Spontaneous canine head and neck squamous cell carcinoma (HNSCC) represents an excellent model of human HNSCC but is greatly understudied. To better understand and utilize this valuable resource, we performed a pilot study that represents its first genome-wide characterization by investigating 12 canine HNSCC cases, of which 9 are oral, via high density array comparative genomic hybridization and RNA-seq. The analyses reveal that these canine cancers recapitulate many molecular features of human HNSCC. These include analogous genomic copy number abnormality landscapes and sequence mutation patterns, recurrent alteration of known HNSCC genes and pathways (e. g., cell cycle, PI3K/AKT signaling), and comparably extensive heterogeneity. Amplification or overexpression of protein kinase genes, matrix metalloproteinase genes, and epithelial-mesenchymal transition genes TWIST1 and SNAI1 are also prominent in these canine tumors. This pilot study, along with a rapidly growing body of literature on canine cancer, reemphasizes the potential value of spontaneous canine cancers in HNSCC basic and translational research.",
    "target": "Head and neck squamous cell carcinoma (HNSCC) represents the sixth leading cancer by incidence in humans; thus, developing effective therapeutic interventions is important. Although great advance has been made in our understanding of the biology of HNSCC over the past several decades, translating the research findings into clinical success has been frustratingly slow, and anticancer drug development remains a lengthy and expensive process. A significant challenge is that drug effects in current preclinical cancer models often do not predict clinical results, and there lacks translational models that can bridge the gap between preclinical research and human clinical trials. Here we report a pilot study that represents the first genome-wide characterization of spontaneously occurring HNSCCs in pet dogs. The study reveals a strong dog-human molecular homology at various levels, indicating the likelihood that spontaneous canine HNSCC molecularly represents its human counterpart. If conclusions of this pilot study are validated with a large sample size and more efforts are put into building better resource and infrastructure for canine cancer research, spontaneous canine HNSCCs could effectively serve as a much-needed translational model that bridges the gap between preclinical research and human trials."
  },
            ]
        }

    def _call_llm(self,
                 system_message: str,
                 user_message: str,
                 max_retries: int = 3) -> str:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message with the medical text
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0
        while retries < max_retries:
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent outputs
                )

                response = completion.choices[0].message.content
                return response

            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief pause before retrying

    def basic_prompt(self, medical_text: str) -> str:
        """
        Create a basic prompt for lay summary generation.

        Args:
            medical_text: Medical text to be simplified

        Returns:
            Formatted basic prompt string
        """
        prompt = (
            f"Please create a lay summary of the following medical text. The summary should be "
            f"a single paragraph that explains the content in simple terms for a general audience:\n\n"
            f"{medical_text}"
        )
        return prompt

    def optimized_prompt(self, medical_text: str) -> str:
        """
        Create an optimized prompt with guidelines for lay summary generation.

        Args:
            medical_text: Medical text to be simplified

        Returns:
            Formatted optimized prompt string
        """
        prompt = (
            f"Please create a lay summary of the following medical text. The summary should:\n"
            f"- Use plain language that avoids jargon and technical terms\n"
            f"- Explain complex concepts in simple, accessible ways\n"
            f"- Focus on the most important findings and their implications\n"
            f"- Be written at approximately an 8th-grade reading level\n"
            f"- Use active voice and concrete examples or analogies when helpful\n"
            f"- Present findings truthfully without exaggeration\n"
            f"- Provide sufficient context to understand the significance\n"
            f"- Maintain accuracy while making the content accessible\n"
            f"- Be structured as a single coherent paragraph\n\n"
            f"Medical text:\n{medical_text}"
        )
        return prompt

    def few_shot_prompt(self, medical_text: str, dataset: str) -> str:
        """
        Create a few-shot prompt with examples for lay summary generation.

        Args:
            medical_text: Medical text to be simplified
            dataset: Dataset name to select appropriate examples

        Returns:
            Formatted few-shot prompt string
        """
        # Get examples for the specified dataset
        examples = self.few_shot_examples.get(dataset, [])
        if not examples or len(examples) < 2:
            print(f"Warning: Not enough examples for dataset {dataset}. Using placeholder examples.")
            examples = [
                {"source": "Example medical text 1", "target": "Example lay summary 1"},
                {"source": "Example medical text 2", "target": "Example lay summary 2"}
            ]

        # Format examples
        examples_text = ""
        for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
            examples_text += f"Example {i}:\n"
            examples_text += f"Medical text: {example['source']}\n"
            examples_text += f"Lay summary: {example['target']}\n\n"

        # Create the prompt with examples
        prompt = (
            f"Please create a lay summary of the following medical text, similar to the examples provided. "
            f"The summary should:\n"
            f"- Use plain language that avoids jargon and technical terms\n"
            f"- Explain complex concepts in simple, accessible ways\n"
            f"- Focus on the most important findings and their implications\n"
            f"- Be written at approximately an 8th-grade reading level\n"
            f"- Use active voice and concrete examples or analogies when helpful\n"
            f"- Present findings truthfully without exaggeration\n"
            f"- Provide sufficient context to understand the significance\n"
            f"- Maintain accuracy while making the content accessible\n"
            f"- Be structured as a single coherent paragraph\n\n"
            f"Here are examples of the expected style and format:\n\n"
            f"{examples_text}"
            f"Medical text to summarize:\n{medical_text}"
        )
        return prompt

    def generate_lay_summary(self, medical_text: str, prompt_type: str, dataset: str = None) -> str:
        """
        Generate a lay summary using the specified prompting technique.

        Args:
            medical_text: Medical text to be summarized
            prompt_type: Type of prompting to use (basic, optimized, few_shot)
            dataset: Dataset name (required for few_shot)

        Returns:
            Generated lay summary
        """
        # Set system message
        system_message = (
            "You are an expert medical writer specializing in creating accessible lay summaries "
            "that make complex medical information understandable to the general public."
        )

        # Generate user prompt based on technique
        if prompt_type == "basic":
            user_prompt = self.basic_prompt(medical_text)
        elif prompt_type == "optimized":
            user_prompt = self.optimized_prompt(medical_text)
        elif prompt_type == "few_shot":
            if dataset is None:
                raise ValueError("Dataset must be specified for few_shot prompt type")
            user_prompt = self.few_shot_prompt(medical_text, dataset)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Call LLM to get response
        response = self._call_llm(system_message, user_prompt)
        return response

    def process_item(self, item: Dict[str, Any], prompt_type: str, dataset: str) -> Dict[str, Any]:
        """
        Process a single item using the specified prompting technique.

        Args:
            item: Input data dictionary with source text
            prompt_type: Type of prompting to use (basic, optimized, few_shot)
            dataset: Dataset name

        Returns:
            Result dictionary with generated lay summary
        """
        start_time = time.time()

        # Extract item fields
        item_id = item.get("id", "unknown")
        source_text = item.get("source", "")
        target_text = item.get("target", "")

        print(f"Processing item {item_id} with {prompt_type} prompting")

        # Generate lay summary
        try:
            lay_summary = self.generate_lay_summary(source_text, prompt_type, dataset)
            processing_time = time.time() - start_time

            # Prepare the result structure
            result = {
                "id": item_id,
                "source": source_text,
                "target": target_text,
                "pred": lay_summary,
                "metadata": {
                    "model": self.model_key,
                    "prompt_type": prompt_type,
                    "processing_time": processing_time
                }
            }

            return result
        except Exception as e:
            print(f"Error generating lay summary for item {item_id}: {e}")
            # Return a partial result with error information
            return {
                "id": item_id,
                "source": source_text,
                "target": target_text,
                "pred": f"Error: {str(e)}",
                "metadata": {
                    "model": self.model_key,
                    "prompt_type": prompt_type,
                    "error": str(e)
                }
            }


def process_dataset(
    dataset_name: str,
    prompt_type: str,
    model_key: str = "deepseek-v3-official"
) -> None:
    """
    Process a medical literature dataset and generate lay summaries.

    Args:
        dataset_name: Name of the dataset to process
        prompt_type: Type of prompting to use (basic, optimized, few_shot)
        model_key: LLM model to use
    """
    # Set up file paths
    input_path = f"my_datasets/processed/laysummary/{dataset_name}/test.json"
    output_dir = f"logs/laysummary/{dataset_name}/SingleLLM_{prompt_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    try:
        dataset = load_json(input_path)
        print(f"Loaded {len(dataset)} samples from {input_path}")
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return

    # Initialize the model
    model = SingleLLMLaySummary(model_key=model_key)

    # Process each sample
    for sample in tqdm(dataset, desc=f"Processing {dataset_name} with {prompt_type}"):
        sample_id = sample.get("id")

        # Skip if already processed
        output_path = f"{output_dir}/laysummary_{sample_id}-result.json"
        if os.path.exists(output_path):
            print(f"Skipping sample {sample_id} - already processed")
            continue

        try:
            print(f"Processing sample {sample_id}")

            # Generate lay summary
            result = model.process_item(sample, prompt_type, dataset_name)

            # Save result
            save_json(result, output_path)
            print(f"Saved result for sample {sample_id}")

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Medical Literature Lay Summary Generation using Single LLM")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["PLABA", "cochrane", "elife", "med_easi", "plos_genetics"],
                       help="Dataset to process")
    parser.add_argument("--prompt_type", type=str, required=True,
                       choices=["basic", "optimized", "few_shot"],
                       help="Prompting technique to use")
    parser.add_argument("--model_key", type=str, default="deepseek-v3-official",
                       help="Model key from LLM_MODELS_SETTINGS")

    args = parser.parse_args()

    # Process dataset
    process_dataset(
        dataset_name=args.dataset,
        prompt_type=args.prompt_type,
        model_key=args.model_key
    )


if __name__ == "__main__":
    main()