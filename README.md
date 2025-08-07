# ArgoPulse AI

ArgoPulse AI is an intelligent, agentic framework designed to analyze and prepare multi-file software projects for summarization by Large Language Models (LLMs). It intelligently filters out irrelevant files and consolidates the essential code into a clean format, making it perfect for efficient and accurate code analysis.

## Overview

When trying to understand a large codebase, much of the content (dependency folders, build artifacts, images, boilerplate configs) is just noise. ArgoPulse AI tackles this problem by:

1.  **Scanning** a project's entire file structure.
2.  **Intelligently Filtering** the structure using a combination of pre-configured rules and LLM-powered analysis to decide what's important.
3.  **Generating** clean outputs: a final file tree and a consolidated content file containing only the relevant code.

This process ensures that the context provided to a final summarization LLM is dense, relevant, and cost-effective (by reducing token count).

## Features

-   **Hierarchical Structure Generation:** Creates a clean, visual tree of the project's file structure.
-   **Code Consolidation:** Combines the contents of all relevant files into a single, structured JSON file.
-   **LLM-Powered Intelligent Filtering:** Uses the Gemini API to analyze the file structure and programmatically identify and exclude irrelevant files and folders (like `node_modules`, `dist/`, etc.).
-   **Configurable and Extensible:** All settings, including model names, ignored files, and binary extensions, are managed in a simple `config.json` file.
-   **Token Counting:** Calculates the total token count of the final consolidated content to estimate API costs.

## Getting Started

Follow these steps to get the project set up and running on your local machine.

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
    -   On **Windows**:
        ```cmd
        python -m venv venv
        venv\Scripts\activate
        ```
    -   On **macOS / Linux**:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Configuration is split into two main files:

1.  **API Key (`.env` file):**
    -   This file stores your secret Google Gemini API key. It is **not** committed to Git.
    -   Create a file named `.env` in the root of the project by copying the provided example:
        -   **Windows:** `copy .env.example .env`
        -   **macOS/Linux:** `cp .env.example .env`
    -   Open the new `.env` file and add your API key:
        ```
        GEMINI_API_KEY="your_google_ai_studio_api_key_here"
        ```

2.  **Models and Settings (`config.json` file):**
    -   This file controls the behavior of the script, such as which models to use and what files/folders to ignore by default.
    -   Copy the example configuration to create your local version:
        -   **Windows:** `copy config.example.json config.json`
        -   **macOS/Linux:** `cp config.example.json config.json`
    -   You can now edit `config.json` to change models or add to the ignore lists.
        -   `generation_model_name`: The model used for intelligent filtering.
        -   `tokenizer_model_name`: The model used for counting tokens.
        -   `ignore_dirs`, `ignore_files`: Default lists of items to always ignore.

## Usage

The script is run from the command line, pointing it to the project you want to analyze.

### Command-Line Arguments

python main.py <project_directory> [options]


-   **`project_directory`** (Required): The path to the project folder you want to analyze.
-   **`--count-tokens`** (Optional): If set, the script will calculate and display the total token count of the final consolidated content.
-   **`--intelligent-filter`** (Optional): If set, the script will use an LLM to analyze the project structure and intelligently add files and folders to the ignore list before generating the final output.

### Example Commands

Let's assume you have a project located at `C:/Users/You/Projects/my-web-app`.

1.  **Basic Analysis:**
    -   Generates `structure.txt` and `consolidated_content.json` using only the rules from `config.json`.
    ```bash
    python main.py C:/Users/You/Projects/my-web-app
    ```

2.  **Analysis with Token Count:**
    -   Does the basic analysis and also reports the total token count.
    ```bash
    python main.py C:/Users/You/Projects/my-web-app --count-tokens
    ```

3.  **Full Intelligent Analysis:**
    -   The most powerful mode. It uses the LLM to filter the project, generates the final outputs, and counts the tokens of the filtered result.
    ```bash
    python main.py C:/Users/You/Projects/my-web-app --intelligent-filter --count-tokens
    ```

## Output

After running, the script will create a new folder in your `ArgoPulse-AI` directory named `<project_name>-output`. This folder will contain:

-   **`structure.txt`**: A clean, tree-like representation of the final, filtered file structure.
-   **`consolidated_content.json`**: A JSON file containing the relative paths and full text content of every file included in the final structure.
-   **`llm_filter_analysis.json`** (if `--intelligent-filter` is used): A JSON file showing exactly which files and folders the LLM decided to ignore. This is useful for debugging the filtering process.