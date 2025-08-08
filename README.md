# ArgoPulse AI

ArgoPulse AI is an intelligent, agentic framework designed to analyze software projects and generate multi-level technical summaries. It streamlines the process of understanding complex codebases by automating data collection, filtering, and documentation generation.

## The ArgoPulse Workflow

The tool operates in a two-step workflow, allowing for flexibility and efficiency:

1.  **Analyze (`--analyze`):** First, the tool scans a project directory. It intelligently filters out irrelevant "noise" (like dependency folders, build artifacts, and binary files) and produces two key artifacts:
    *   `structure.txt`: A clean, tree-like representation of the project's essential files.
    *   `consolidated_content.json`: A single JSON file containing the code and content of all relevant files.
    This step only needs to be run once per project, or whenever the code changes significantly.

2.  **Summarize (`--summarize`):** Once the analysis files exist, you can generate technical summaries at different levels of detail (low, medium, or high). This step uses the analysis artifacts as context for a powerful Large Language Model (LLM), which writes the documentation. You can re-run this step to generate different summary levels without re-analyzing the project.

## Features

-   **Two-Step Process:** Analyze and summarize independently for maximum efficiency.
-   **Multi-Level Summaries:** Generate high-level, medium-level, and low-level documentation to suit different needs.
-   **LLM-Powered Filtering:** Optionally use an LLM during the analysis phase to programmatically identify and exclude boilerplate and irrelevant files.
-   **Intelligent Content Handling:** Includes code files fully but only includes previews of large data files (like `.csv`) to save tokens.
-   **Fully Configurable:** Control model names, ignore lists, and other settings via a simple `config.json` file.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fatim-Shoaib/ArgoPulse-AI.git
    cd ArgoPulse-AI
    ```

2.  **Create and activate a virtual environment:**
    -   On **Windows**: `python -m venv venv && venv\Scripts\activate`
    -   On **macOS / Linux**: `python -m venv venv && source venv/bin/activate`

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **API Key (`.env` file):**
    -   Copy the example file: `copy .env.example .env` (Windows) or `cp .env.example .env` (macOS/Linux).
    -   Open the new `.env` file and add your Google Gemini API key:
        ```
        GEMINI_API_KEY="your_google_ai_studio_api_key_here"
        ```

2.  **Models and Settings (`config.json` file):**
    -   Copy the example file: `copy config.example.json config.json` (Windows) or `cp config.example.json config.json` (macOS/Linux).
    -   Edit `config.json` to change the default models, add files/folders to the ignore lists, or define new data file types.

## Usage

The script is run from the command line, specifying the project to process and the actions to perform.

### Command-Line Arguments

```
python main.py <project_path> [--analyze] [--summarize low|medium|high] [options]
```

-   `project_path` (Required): The path to the project folder you want to process.
-   `--analyze`: Perform the analysis step. Creates the `structure.txt` and `consolidated_content.json` files.
-   `--summarize <levels...>`: Perform the summarization step. Can accept one or more levels: `low`, `medium`, `high`.
-   `--intelligent-filter`: (Optional, with `--analyze`) Use an LLM to help filter files during analysis.
-   `--count-tokens`: (Optional, with `--analyze`) Count the tokens of the consolidated content.

### Example Workflow

Let's assume you have a project located at `C:/Code/my-app`.

**Step 1: Analyze the Project**

First, run the analysis. We'll use the intelligent filter for the best results.

```bash
python main.py C:/Code/my-app --analyze --intelligent-filter
```
This creates a `my-app-output` folder containing the analysis files.

**Step 2: Generate Summaries**

Now that the analysis is done, you can generate any summary you need.

-   **To get a high-level overview:**
    ```bash
    python main.py C:/Code/my-app --summarize high
    ```

-   **To get a very detailed, low-level summary:**
    ```bash
    python main.py C:/Code/my-app --summarize low
    ```

-   **To get all three summaries at once:**
    ```bash
    python main.py C:/Code/my-app --summarize low medium high
    ```

You can also perform both actions in a single command:
```bash
python main.py C:/Code/my-app --analyze --summarize medium
```

## Output

All output is saved in a new folder named `<project_name>-output`.

-   **Analysis Files:**
    -   `structure.txt`: The filtered project file tree.
    -   `consolidated_content.json`: The consolidated code and content.
    -   `llm_filter_analysis.json` (Optional): The LLM's decisions during intelligent filtering.
-   **Summary Files:**
    -   `high_level_summary.md`
    -   `medium_level_summary.md`
    -   `low_level_summary.md`