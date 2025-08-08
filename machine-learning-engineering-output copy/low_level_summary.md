
# Machine Learning Engineering Pipeline Documentation

## 1. Pipeline Description

The Machine Learning Engineering (MLE) pipeline is designed to automate the process of developing and deploying machine learning models for a given task. It leverages a multi-agent system to initialize solutions, refine them through ablation studies and iterative improvement, ensemble multiple solutions for better performance, and generate submission files for competitions like Kaggle.

**Inputs:**

- Task description (e.g., `task_description.txt` in the task directory).
- Training data (e.g., `train.csv` in the task directory).
- Test data (e.g., `test.csv` in the task directory).
- Configuration parameters (defined in `config.py`).

**Processing:**

1.  **Initialization:** Generates initial candidate solutions by retrieving pre-trained models or suggesting model architectures based on the task description.
2.  **Refinement:** Iteratively refines the candidate solutions through ablation studies (identifying the most important code components), generating improvement plans and implementing them.
3.  **Ensemble:** Combines multiple refined solutions to create an ensemble model with potentially improved performance.
4.  **Submission:** Generates a submission file (`submission.csv`) containing predictions on the test data.

**Outputs:**

-   A `submission.csv` file located in the `./machine_learning_engineering/workspace/<task_name>/ensemble/final/` directory, ready for submission to a competition.
-   Intermediate code files, logs, and state information are saved in the `./machine_learning_engineering/workspace/` directory.

## 2. Major Components

The pipeline consists of a series of agents and sub-agents orchestrated by `agents.SequentialAgent` and `agents.ParallelAgent` from the Google ADK.

1.  **Root Agent (`mle_frontdoor_agent`):** The entry point of the pipeline. It receives the initial instruction from the user and delegates the task to the `mle_pipeline_agent`.
2.  **MLE Pipeline Agent (`mle_pipeline_agent`):** A sequential agent that executes the core stages of the pipeline: initialization, refinement, ensembling, and submission.
3.  **Initialization Agent (`initialization_agent`):** Generates initial candidate solutions.
4.  **Refinement Agent (`refinement_agent`):** Iteratively refines the candidate solutions.
5.  **Ensemble Agent (`ensemble_agent`):** Combines multiple refined solutions into an ensemble.
6.  **Submission Agent (`submission_agent`):** Generates the submission file.

## 3. Data Flow

The data flows through the pipeline via the `callback_context.state` object. This object is a dictionary-like structure that persists data between agent calls.

1.  The Root Agent receives user input (task).
2.  The user input is passed to the MLE Pipeline Agent.
3.  The MLE Pipeline Agent passes the input to the Initialization Agent.
4.  The Initialization Agent reads configuration from `config.py`, the task description from a file (e.g. `task_description.txt`), and generates initial code solutions. The solutions, along with their execution results, are stored in the `callback_context.state`.
5.  The Refinement Agent accesses the solutions from `callback_context.state`, performs ablation studies, refines the code, and updates the solutions in `callback_context.state`.
6.  The Ensemble Agent accesses the refined solutions from `callback_context.state`, creates an ensemble solution, and stores it in `callback_context.state`.
7.  The Submission Agent retrieves the final solution from `callback_context.state`, adds code for generating a submission file, and stores the final submission code in `callback_context.state`.

## 4. LLM Usage

LLMs are used extensively throughout the pipeline for:

-   **Task Summarization:** The `task_summarization_agent` in `initialization_agent` summarizes the task description for better model retrieval.
-   **Model Retrieval:** The `model_retriever_agent` in `initialization_agent` uses an LLM and Google Search to find appropriate models for the task.
-   **Code Generation:** The `model_eval` agent in `initialization_agent`, `ablation_agent` in `refinement_agent`, `plan_implement` agents in `refinement_agent`, `ensemble_plan_implement` agents in `ensemble_agent` and `submission_agent` use LLMs to generate Python code based on instructions and context.
-   **Code Integration:** The `merger` agent in `initialization_agent` uses an LLM to merge two code solutions into one.
-   **Code Debugging:** The `debug_agent` uses LLMs to understand and fix errors in generated code.
-   **Ablation Study Planning:** The `ablation_agent` in the `refinement_agent` uses an LLM to write an ablation study. The result of the ablation study is then used by the `init_plan_agent` to plan code improvements.
-   **Data Leakage Check:** The `check_leakage_agent` in `check_leakage_util.py` uses LLMs to determine if there is data leakage in the provided code, and refine it if there is any.
-   **Data Usage Check:** The `check_data_use` agent uses LLMs to verify if all the information in the task description and data is being used.

## 5. File Details

### 5.1. `machine_learning_engineering/__init__.py`

```python
"""Machine Learning Engineer: automate the implementation of ML models."""

from . import agent
```

-   **Purpose:** Initializes the `machine_learning_engineering` package.
-   **Content:** Imports the `agent` module, making its contents available within the package.

### 5.2. `machine_learning_engineering/agent.py`

```python
"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit"""

import os
import json
from typing import Optional
from google.genai import types
from google.adk.agents import callback_context as callback_context_module

from google.adk import agents
from machine_learning_engineering.sub_agents.initialization import agent as initialization_agent_module
from machine_learning_engineering.sub_agents.refinement import agent as refinement_agent_module
from machine_learning_engineering.sub_agents.ensemble import agent as ensemble_agent_module
from machine_learning_engineering.sub_agents.submission import agent as submission_agent_module

from machine_learning_engineering import prompt


def save_state(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    return None


mle_pipeline_agent = agents.SequentialAgent(
    name="mle_pipeline_agent",
    sub_agents=[
        initialization_agent_module.initialization_agent,
        refinement_agent_module.refinement_agent,
        ensemble_agent_module.ensemble_agent,
        submission_agent_module.submission_agent,
    ],
    description="Executes a sequence of sub-agents for solving the MLE task.",
    after_agent_callback=save_state,
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = agents.Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="mle_frontdoor_agent",
    instruction=prompt.FRONTDOOR_INSTRUCTION,
    global_instruction=prompt.SYSTEM_INSTRUCTION,
    sub_agents=[mle_pipeline_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
```

-   **Purpose:** Defines the main agents and orchestrates the pipeline.
-   **Key Components:**
    -   `save_state`: Function to save the current state of the callback context to a JSON file.
        -   **Input:** `callback_context` (`callback_context_module.CallbackContext`).
        -   **Output:** `Optional[types.Content]`. Returns `None`.
        -   **Process:** Retrieves the `workspace_dir` and `task_name` from the callback context's state.  Constructs the path to the `final_state.json` file and saves the entire `callback_context.state` to that file as a JSON dump with an indent of 2.
        -   **Purpose:** Used as an `after_agent_callback` to persist the agent's state after the `mle_pipeline_agent` finishes execution, allowing for inspection or continuation.

    -   `mle_pipeline_agent`: An instance of `agents.SequentialAgent`.
        -   **Purpose:** Executes the core sub-agents sequentially: `initialization_agent`, `refinement_agent`, `ensemble_agent`, and `submission_agent`.
        -   **Sub-agents:** The sub-agents are imported from their respective modules (e.g., `initialization_agent_module`).
        -   **`after_agent_callback`**: `save_state` function is called after `mle_pipeline_agent` finishes.

    -   `root_agent`: An instance of `agents.Agent`.
        -   **Purpose:** The root agent that handles the initial user input and delegates the task to the `mle_pipeline_agent`.
        -   **Model:** Uses the LLM specified by the `ROOT_AGENT_MODEL` environment variable.
        -   **Instruction:** Uses the `FRONTDOOR_INSTRUCTION` and `SYSTEM_INSTRUCTION` prompts defined in `prompt.py`.
        -   **Sub-agents:** Contains the `mle_pipeline_agent`.
        -   **`generate_content_config`**: Sets the temperature to 0.01 for content generation.

### 5.3. `machine_learning_engineering/prompt.py`

```python
"""Defines the prompts in the Machine Learning Engineering Agent."""


SYSTEM_INSTRUCTION ="""You are a Machine Learning Engineering Multi Agent System.
"""

FRONTDOOR_INSTRUCTION="""
You are a machine learning engineer given a machine learning task for which to engineer a solution.

    - If the user asks questions that can be answered directly, answer it directly without calling any additional agents.
    - In this example, the task is the California Housing Task.
    - If the user asks for a description of the task, then obtain the task, extract the description and return it. Do not execute the task.

    # **Workflow:**

    # 1. Obtain intent.

    # 2. Obtain task

    # 3. Carry out task


    # **Tool Usage Summary:**

    #   * **Greeting/Out of Scope:** answer directly.
"""


TASK_AGENT_INSTR = """# Introduction
- Your task is to be a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- You need to first obtain a absolute path to the local directory that contains the data of the Kaggle competition from the user.
"""
```

-   **Purpose:** Defines the prompts used by the agents.
-   **Key Components:**
    -   `SYSTEM_INSTRUCTION`: A high-level instruction for the multi-agent system.
    -   `FRONTDOOR_INSTRUCTION`: Instructions for the root agent to handle user input and delegate tasks.
    -   `TASK_AGENT_INSTR`: Instructions for an agent working on a Kaggle competition task.

### 5.4. `machine_learning_engineering/shared_libraries/__init__.py`

```python
""
```

-   **Purpose:** Initializes the `shared_libraries` package.
-   **Content:** Empty file.

### 5.5. `machine_learning_engineering/shared_libraries/check_leakage_util.py`

```python
"""Utility functions for leakage check agent."""

from typing import Optional
import json
import functools

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types

from machine_learning_engineering.shared_libraries import data_leakage_prompt
from machine_learning_engineering.shared_libraries import code_util
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config


def get_check_leakage_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the check leakage agent instruction."""
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = context.state.get(code_state_key, "")
    return data_leakage_prompt.CHECK_LEAKAGE_INSTR.format(
        code=code,
    )


def get_refine_leakage_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the refine leakage agent instruction."""
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = context.state.get(code_state_key, "")
    return data_leakage_prompt.LEAKAGE_REFINE_INSTR.format(
        code=code,
    )


def parse_leakage_status(text: str) -> tuple[str, str]:
    """Parses the leakage status from the text."""
    start_idx, end_idx = text.find("["), text.rfind("]")+1
    text = text[start_idx:end_idx]
    result = json.loads(text)[0]
    leakage_status = result["leakage_status"]
    code_block = result["code_block"].replace(f"```python", "").replace("```", "")
    return leakage_status, code_block


def update_extract_status(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Updates the status of extraction."""
    response_text = common_util.get_text_from_response(llm_response)
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = callback_context.state.get(code_state_key, "")
    if "No Data Leakage" in response_text:
        leakage_status = "No Data Leakage"
    try:
        leakage_status, code_block = parse_leakage_status(response_text)
        if leakage_status == "No Data Leakage":
            extract_status = True
        else:
            extract_status = code_block in code
    except:
        code_block = ""
        extract_status = False
    extract_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="extract_status",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_block_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_block",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_status",
        prefix=prefix,
        suffix=suffix,
    )
    callback_context.state[extract_status_key] = extract_status
    callback_context.state[leakage_block_key] = code_block
    callback_context.state[leakage_status_key] = leakage_status
    return None


def check_extract_status(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks the status of extraction."""
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    extract_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="extract_status",
        prefix=prefix,
        suffix=suffix,
    )
    skip_data_leakage_check_key = code_util.get_name_with_prefix_and_suffix(
        base_name="skip_data_leakage_check",
        prefix=prefix,
        suffix=suffix,
    )
    extract_status = callback_context.state.get(extract_status_key, False)
    skip_flag = callback_context.state.get(skip_data_leakage_check_key, False)
    if skip_flag or extract_status:
        return llm_response_module.LlmResponse()
    return None


def replace_leakage_code(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Replace the code block that has the data leakage issue."""
    response_text = common_util.get_text_from_response(llm_response)
    refined_code_block = response_text.replace("```python", "").replace("```", "")
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    leakage_block_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_block",
        prefix=prefix,
        suffix=suffix,
    )
    code_block = callback_context.state.get(leakage_block_key, "")
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = callback_context.state.get(code_state_key, "")
    refined_code = code.replace(code_block, refined_code_block)
    callback_context.state[code_state_key] = refined_code
    code_util.evaluate_code(callback_context=callback_context)
    return None


def check_data_leakage(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the code has the data leakage issue."""
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    leakage_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_status",
        prefix=prefix,
        suffix=suffix,
    )
    skip_data_leakage_check_key = code_util.get_name_with_prefix_and_suffix(
        base_name="skip_data_leakage_check",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_status = callback_context.state.get(leakage_status_key, "")
    skip_flag = callback_context.state.get(skip_data_leakage_check_key, False)
    if skip_flag or ("Yes Data Leakage" not in leakage_status):
        return llm_response_module.LlmResponse()
    return None


def get_data_leakage_checker_agent(
    prefix: str,
    suffix: str,
) -> agents.SequentialAgent:
    """Gets the data leakage checker agent."""
    check_leakage_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="check_leakage_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check if the code has the data leakage issue.",
        instruction=get_check_leakage_agent_instruction,
        before_model_callback=functools.partial(
            check_extract_status,
            prefix=prefix,
        ),
        after_model_callback=functools.partial(
            update_extract_status,
            prefix=prefix,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    check_leakage_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="check_leakage_loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check if the code has the data leakage issue until extraction succeeds.",
        sub_agents=[
            check_leakage_agent,
        ],
        max_iterations=config.CONFIG.max_retry,
    )
    refine_leakage_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="refine_leakage_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Refine the code to address the data leakage issue.",
        instruction=get_refine_leakage_agent_instruction,
        before_model_callback=functools.partial(
            check_data_leakage,
            prefix=prefix,
        ),
        after_model_callback=replace_leakage_code,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    data_leakage_checker_agent = agents.SequentialAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="data_leakage_checker_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check and refine the code to address the data leakage issue.",
        sub_agents=[
            check_leakage_loop_agent,
            refine_leakage_agent,
        ],
    )
    return data_leakage_checker_agent
```

-   **Purpose:** Provides utility functions and an agent for checking and refining code to prevent data leakage.
-   **Key Components:**
    -   `get_check_leakage_agent_instruction`: Function to generate the instruction for the `check_leakage_agent`.
        -   **Input:** `context` (`callback_context_module.ReadonlyContext`).
        -   **Output:** `str`. Returns the instruction string.
        -   **Process:** Get agent name, generate suffix, get code state key based on the agent name and suffix. Then, get code based on `code_state_key`. Finally, return a formatted string with `code` inside, based on `data_leakage_prompt.CHECK_LEAKAGE_INSTR`.
        -   **Purpose:** Used to fetch appropriate instruction for the `check_leakage_agent`
    -   `get_refine_leakage_agent_instruction`: Function to generate the instruction for the `refine_leakage_agent`.
        -   **Input:** `context` (`callback_context_module.ReadonlyContext`).
        -   **Output:** `str`. Returns the instruction string.
        -   **Process:** Get agent name, generate suffix, get code state key based on the agent name and suffix. Then, get code based on `code_state_key`. Finally, return a formatted string with `code` inside, based on `data_leakage_prompt.LEAKAGE_REFINE_INSTR`.
        -   **Purpose:** Used to fetch appropriate instruction for the `refine_leakage_agent`
    -   `parse_leakage_status`: Function to extract leakage status and code block from LLM response.
        -   **Input:** `text` (`str`).
        -   **Output:** `tuple[str, str]`. Returns the leakage status and code block.
        -   **Process:** Locate json section in `text`, load it, and extract `leakage_status` and `code_block` fields.
        -   **Purpose:** Used to parse LLM output and get the leakage status.
    -   `update_extract_status`: Function to update states after the `check_leakage_agent` gets a response.
        -   **Input:** `callback_context` (`callback_context_module.CallbackContext`), `llm_response` (`llm_response_module.LlmResponse`), `prefix` (`str`).
        -   **Output:** `Optional[llm_response_module.LlmResponse]`. Returns `None`.
        -   **Process:** Get the text from the response. If "No Data Leakage" is contained in the text, assign `leakage_status` to be "No Data Leakage". If not, use `parse_leakage_status` to get the `leakage_status` and `code_block`. If the `leakage_status` is "No Data Leakage", assign `extract_status` to be True. If not, determine the `extract_status` by checking if the `code_block` is in the original code. After extraction, get the name with the prefix and suffix for `extract_status`, `leakage_block`, `leakage_status`. Then, update the states based on the value.
        -   **Purpose:** Used to determine whether the code block is extracted successfully.
    -   `check_extract_status`: Function to determine whether to skip `check_leakage_agent` based on previous status.
        -   **Input:** `callback_context` (`callback_context_module.CallbackContext`), `llm_request` (`llm_request_module.LlmRequest`), `prefix` (`str`).
        -   **Output:** `Optional[llm_response_module.LlmResponse]`. Return `LlmResponse()` if extraction has been done, otherwise return `None`.
        -   **Process:** Get suffix, generate `extract_status_key` and `skip_data_leakage_check_key`. Get `extract_status` and `skip_flag` based on the keys. If `skip_flag` or `extract_status` is True, then return a dummy `LlmResponse`. Otherwise, return `None`.
        -   **Purpose:** Used as `before_model_callback` to determine whether to call the model or not.
    -   `replace_leakage_code`: Function to replace leakage code after the `refine_leakage_agent` gets a response.
        -   **Input:** `callback_context` (`callback_context_module.CallbackContext`), `llm_response` (`llm_response_module.LlmResponse`), `prefix` (`str`).
        -   **Output:** `Optional[llm_response_module.LlmResponse]`. Returns `None`.
        -   **Process:** Get the refined code block from the llm response. Then get the leakage code block based on state. Finally, get the original code, replace leakage code, update to the state, and evaluate it.
        -   **Purpose:** Used to replace the buggy code.
    -   `check_data_leakage`: Function to determine whether to skip `refine_leakage_agent` based on leakage status.
        -   **Input:** `callback_context` (`callback_context_module.CallbackContext`), `llm_request` (`llm_request_module.LlmRequest`), `prefix` (`str`).
        -   **Output:** `Optional[llm_response_module.LlmResponse]`. Returns `LlmResponse()` if data leakage check can be skipped, otherwise return `None`.
        -   **Process:** Get suffix, generate `leakage_status_key` and `skip_data_leakage_check_key`. Get `leakage_status` and `skip_flag` based on the keys. If `skip_flag` is True or "Yes Data Leakage" is not in `leakage_status`, then return a dummy `LlmResponse`. Otherwise, return `None`.
        -   **Purpose:** Used as `before_model_callback` to determine whether to call the model or not.
    -   `get_data_leakage_checker_agent`: Function to get the data leakage checker agent.
        -   **Input:** `prefix` (`str`), `suffix` (`str`).
        -   **Output:** `agents.SequentialAgent`. Returns the `data_leakage_checker_agent` instance.
        -   **Process:** Creates the `check_leakage_agent`, `check_leakage_loop_agent`, `refine_leakage_agent` and the `data_leakage_checker_agent`. Sets the instructions, before/after callbacks, generate content configs.
        -   **Purpose:** The main function to generate `data_leakage_checker_agent` in order to check data leakage.
        -   `check_leakage_agent`: An instance of `agents.Agent`.
            -   **Purpose:** Check if the code has the data leakage issue.
            -   **`instruction`**: `get_check_leakage_agent_instruction`
            -   **`before_model_callback`**: `check_extract_status`
            -   **`after_model_callback`**: `update_extract_status`
        -   `check_leakage_loop_agent`: An instance of `agents.LoopAgent`.
            -   **Purpose:** Check if the code has the data leakage issue until extraction succeeds.
            -   **`sub_agents`**: `check_leakage_agent`
        -   `refine_leakage_agent`: An instance of `agents.Agent`.
            -   **Purpose:** Refine the code to address the data leakage issue.
            -   **`instruction`**: `get_refine_leakage_agent_instruction`
            -   **`before_model_callback`**: `check_data_leakage`
            -   **`after_model_callback`**: `replace_leakage_code`

### 5.6. `machine_learning_engineering/shared_libraries/code_util.py`

```python
"""Code related utility functions."""

from typing import Any
import subprocess
import os
import time

from google.adk.agents import callback_context as callback_context_module


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_python_code(
    code_text: str,
    run_cwd: str,
    py_filepath: str,
    exec_timeout: int,
) -> dict[str, Any]:
    start_time = time.time()
    output_filepath = os.path.join(run_cwd, py_filepath)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    try:
        result = subprocess.run(
            ["python", py_filepath],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
        )
    except Exception as e:
        result = Result(returncode=1, stdout="", stderr=str(e))
    end_time = time.time()
    execution_time = end_time - start_time
    result_dict = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time,
    }
    return result_dict


def extract_performance_from_text(text: str) -> float | None:
    """Extracts the final validation performance score from the text."""
    lines = text.splitlines()
    performance_value = None
    for line in lines:
        if "Final Validation Performance" in line:
            try:
                parts = line.split(":")
                # score_str = line.split("Final Validation Performance:")[-1].strip()
                score_str = parts[-1].strip()
                performance_value = float(score_str)
            except ValueError:
                pass
    return performance_value


def get_name_with_prefix_and_suffix(
    base_name: str,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Gets the name with the specified prefix and suffix."""
    new_name = base_name
    if prefix:
        new_name = prefix + "_" + new_name
    if suffix:
        new_name = new_name + "_" + suffix
    return new_name


def get_updated_suffix(
    callback_context: callback_context_module.CallbackContext,
) -> str:
    """Gets the suffix string."""
    agent_name = callback_context.agent_name
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{model_id}"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{reference_idx}"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        suffix = f"{task_id}"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        suffix = f"{step}_{task_id}"
    elif agent_name.startswith("plan_implement"):
        task_id = callback_context.agent_name.split("_")[-1]
        step = callback_context.state.get(f"