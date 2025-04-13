# MedAgentBoard

## Folder Structure

```
├── README.md
├── logs
│   └── medqa
│       ├── MedQA
│       ├── PathVQA
│       ├── PubMedQA
│       └── VQA-RAD
├── medagentboard
│   ├── medqa
│   │   ├── README.md
│   │   ├── multi_agent_colacare.py
│   │   ├── multi_agent_mdagents.py
│   │   ├── multi_agent_medagent.py
│   │   ├── multi_agent_reconcile.py
│   │   ├── preprocess_datasets.py
│   │   ├── run.sh
│   │   └── single_llm.py
│   └── utils
│       ├── encode_image.py
│       ├── json_utils.py
│       └── llm_configs.py
├── my_datasets
│   ├── processed
│   │   └── medqa
│   └── raw
│       └── medqa # raw datasets are stored here
├── pyproject.toml
└── uv.lock
```

Please setup the .env file with your API keys. E.g.,

```
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
ARK_API_KEY=sk-xxx
```

## Usage:

After cloning the project, follow these steps to set up and run the application.

### Environment Setup

We recommend using `uv` as the Python package management tool:

```bash
# Create the virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/Linux/MacOS
```

### Install Dependencies

```bash
uv pip sync
```

### Running the Application

Execute the script with your desired parameters. E.g.,

```bash
python -m medagentboard.medqa.multi_agent_colacare --dataset PubMedQA --qa_type mc
```

To run the all configs for MedQA tasks, use:

```bash
bash medagentboard/medqa/run.sh
```