# 🏥 *MedAgentBoard*

**🎉 Our paper has been accepted to the NeurIPS 2025 Datasets & Benchmarks Track! 🎉**

[![arXiv](https://img.shields.io/badge/arXiv-2505.12371-b31b1b.svg)](https://arxiv.org/abs/2505.12371)
[![Project Website](https://img.shields.io/badge/Project%20Website-MedAgentBoard-0066cc.svg)](https://medagentboard.netlify.app/)

📄 [**Read the Paper →**](https://arxiv.org/abs/2505.12371) **Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks**

**Authors:** Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu

## Overview

**MedAgentBoard** is a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional (non-LLM) approaches across diverse medical tasks. The rapid advancement of Large Language Models (LLMs) has spurred interest in multi-agent collaboration for complex medical challenges. However, the practical advantages of these multi-agent systems are not yet well understood. Existing evaluations often lack generalizability to diverse real-world clinical tasks and frequently omit rigorous comparisons against both advanced single-LLM baselines and established conventional methods.

MedAgentBoard addresses this critical gap by introducing a benchmark suite covering four distinct medical task categories, utilizing varied data modalities including text, medical images, and structured Electronic Health Records (EHRs):
1.  **Medical (Visual) Question Answering:** Evaluating systems on answering questions from medical texts and/or medical images.
2.  **Lay Summary Generation:** Assessing the ability to convert complex medical texts into easily understandable summaries for patients.
3.  **Structured EHR Predictive Modeling:** Benchmarking predictions of clinical outcomes (e.g., mortality, readmission) using structured patient data.
4.  **Clinical Workflow Automation:** Evaluating the automation of multi-step clinical data analysis workflows, from data extraction to reporting.

Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios (e.g., enhancing task completeness in clinical workflow automation), it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods, which generally maintain superior performance in tasks like medical VQA and EHR-based prediction.

MedAgentBoard serves as a vital resource, offering actionable insights for researchers and practitioners. It underscores the necessity of a task-specific, evidence-based approach when selecting and developing AI solutions in medicine, highlighting that the inherent complexity and overhead of multi-agent systems must be carefully weighed against tangible performance gains.

**All code, datasets, detailed prompts, and experimental results are open-sourced! If you have any questions about this paper, please feel free to contact Yinghao Zhu, yhzhu99@gmail.com.**

## Key Features & Contributions

*   **Comprehensive Benchmark:** Provides a platform for rigorous evaluation and extensive comparative analysis of multi-agent collaboration, single LLMs, and conventional methods across diverse medical tasks and data modalities.
*   **Addresses Critical Gaps:** Directly tackles limitations in current research concerning generalizability and the completeness of baselines by synthesizing prior work with LLM-era evaluations.
*   **Clarity on Multi-Agent Efficacy:** Offers a unified framework for adjudicating the often conflicting claims about the true advantages of multi-agent approaches in the rapidly evolving field of medical AI.
*   **Actionable Insights:** Distills experimental findings into practical guidance for researchers and practitioners to make informed decisions about selecting, developing, and deploying AI solutions in various medical settings.

## Related Multi-Agent Frameworks and Baselines

The MedAgentBoard benchmark evaluates various approaches, including adaptations or implementations based on principles from the following (and other) influential multi-agent frameworks and related research. The project structure reflects implementations for some of these:

- **WWW 2025** [ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration](https://dl.acm.org/doi/abs/10.1145/3696410.3714877)
- **NPJ Digital Medicine 2025** [Enhancing diagnostic capability with multi-agents conversational large language models](https://www.nature.com/articles/s41746-025-01550-0)
- **NPJ Artificial Intelligence 2025** [Healthcare agent: eliciting the power of large language models for medical consultation](https://www.nature.com/articles/s44387-025-00021-x)
- **ACL 2024** [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://aclanthology.org/2024.acl-long.381/)
- **NeurIPS 2024** [MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html)
- **ACL 2024 Findings** [MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning](https://aclanthology.org/2024.findings-acl.33/)
- Other frameworks like AgentSimp, SmolAgents, OpenManus, and Owl are also discussed and utilized for specific tasks within MedAgentBoard (see paper for details).

## Associated Repositories

*   [MedAgentBoard-playground](https://github.com/yhzhu99/MedAgentBoard-playground): Contains the complete code for the project website.
*   [MedAgentBoard-WorkflowAutomation](https://github.com/yhzhu99/MedAgentBoard-WorkflowAutomation): Contains the complete code and results for Task 4 (Clinical Workflow Automation).

## Project Structure

```
medagentboard/
├── ehr/                     # EHR-related multi-agent implementations
│   ├── multi_agent_colacare.py
│   ├── multi_agent_medagent.py
│   ├── multi_agent_reconcile.py
│   ├── preprocess_dataset.py
│   └── run.sh
├── laysummary/              # Lay summary generation components
│   ├── evaluation.py
│   ├── multi_agent_agentsimp.py
│   ├── preprocess_datasets.py
│   ├── run.sh
│   └── single_llm.py
├── medqa/                   # Medical QA system implementations
│   ├── evaluate.py
│   ├── multi_agent_colacare.py
│   ├── multi_agent_mdagents.py
│   ├── multi_agent_medagent.py
│   ├── multi_agent_reconcile.py
│   ├── preprocess_datasets.py
│   ├── run.sh
│   └── single_llm.py
└── utils/                   # Shared utility functions
    ├── encode_image.py
    ├── json_utils.py
    ├── llm_configs.py
    └── llm_scoring.py
```

## Getting Started

### Prerequisites

1. Python 3.10 or higher
2. [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install dependencies from uv.lock
uv sync
```

### Environment Setup

Please setup the .env file with your API keys:

```
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
ARK_API_KEY=sk-xxx
# Add other API keys as needed (e.g., for GPT-4, Gemini, etc.)
```

## Usage

### Running Medical QA

```bash
# Run all MedQA tasks (example from paper, may need specific setup)
bash medagentboard/medqa/run.sh

# Run specific MedQA task
python -m medagentboard.medqa.multi_agent_colacare --dataset PubMedQA --qa_type mc
# Refer to medqa/run.sh and run_colacare_diverse_llms.sh for more examples
```
*Note: Clinical Workflow Automation tasks involve more complex setups; please refer to the paper and codebase for detailed instructions on reproducing those experiments.*

### Running Lay Summary Generation

```bash
python -m medagentboard.laysummary.multi_agent_agentsimp
# Refer to laysummary/run.sh for more examples
```

### Running EHR Components

```bash
python -m medagentboard.ehr.multi_agent_colacare
# Refer to ehr/run.sh for more examples
```

## Citation

If you find MedAgentBoard useful in your research, please cite our paper:

```bibtex
@article{zhu2025medagentboard,
  title={{MedAgentBoard}: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks},
  author={Zhu, Yinghao and He, Ziyi and Hu, Haoran and Zheng, Xiaochen and Zhang, Xichen and Wang, Zixiang and Gao, Junyi and Ma, Liantao and Yu, Lequan},
  journal={arXiv preprint arXiv:2505.12371},
  year={2025}
}
```
