import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# --- Configuration Loader ---

def load_config():
    """Loads configuration from config.json, with fallback to defaults."""
    default_config = {
        "generation_model_name": "gemini-1.5-pro-latest",
        "tokenizer_model_name": "models/gemini-1.5-flash-latest",
        "ignore_dirs": ["__pycache__", ".git", "venv"],
        "ignore_files": [".gitignore", "config.json"],
        "binary_file_extensions": [".pyc"]
    }
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            for key, value in default_config.items():
                config.setdefault(key, value)
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: config.json not found or invalid. Using default settings.")
        return default_config

# --- New LLM-Powered Filtering Function ---

def get_llm_filter_list(structure_string: str, config: dict) -> dict:
    """
    Asks an LLM to identify irrelevant files and folders for summarization.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Cannot perform intelligent filtering: GEMINI_API_KEY not found.")
        return {"ignore_folders": [], "ignore_files": []}

    os.environ['GOOGLE_API_KEY'] = api_key
    model_name = config.get("generation_model_name", "gemini-1.5-pro-latest")
    
    prompt = f"""
You are an expert software developer and code analyst. Your task is to review the following file structure of a project and decide which files and folders are NOT necessary for understanding the core logic of the code.

The goal is to prepare the project for a code summary. Exclude things like:
- Dependency directories (e.g., node_modules, venv)
- Build output folders (e.g., dist, build)
- Generic configuration files that don't reveal project-specific logic (e.g., package-lock.json, .prettierrc)
- Test data, assets, or documentation unless it's critical.

Analyze this file structure:
---
{structure_string}
---

Your response MUST be a valid JSON object and nothing else. Do not add any explanatory text before or after the JSON.
The JSON object must have two keys: "ignore_folders" and "ignore_files". Each key should have a list of strings as its value. Use forward slashes for paths.

Example response format:
{{
  "ignore_folders": ["node_modules/", "dist/"],
  "ignore_files": ["package-lock.json", "assets/logo.svg"]
}}
"""
    try:
        # model = model_name
        # model = genai.GenerativeModel(model_name)
        # client = genai.Client()
        # response = client.models.count_tokens(model=model, contents=prompt)
        # return response.total_tokens
        # client = genai.Client()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("\nWarning: GEMINI_API_KEY not found. Skipping token count.")
            return 0
        os.environ['GOOGLE_API_KEY'] = api_key
        # model = client.generative_models.get(config["generation_model_name"])
        # response = model.generate_content(prompt)
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        # text = response.text.strip()
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        filter_data = json.loads(cleaned_response)

        # Validate the response format
        if isinstance(filter_data, dict) and "ignore_folders" in filter_data and "ignore_files" in filter_data:
            return filter_data
        else:
            print("Warning: LLM response was not in the expected format.")
            return {"ignore_folders": [], "ignore_files": []}

    except Exception as e:
        print(f"Error during LLM filtering: {e}")
        return {"ignore_folders": [], "ignore_files": []}

# --- Core Data Collection Functions (Now with more powerful filtering) ---

def is_binary(file_path: Path, binary_extensions: list) -> bool:
    """Check if a file is likely binary based on its extension."""
    return file_path.suffix.lower() in binary_extensions

def process_project_files(root_dir: Path, config: dict):
    """
    Walks the project directory once to generate both the file structure
    and the consolidated content, respecting all ignore rules.
    """
    tree_lines = [f"{root_dir.name}/"]
    all_files_data = []
    
    ignore_dirs = set(config['ignore_dirs'])
    ignore_files = set(config['ignore_files'])
    binary_extensions = config['binary_file_extensions']

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter directories to ignore
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        current_path = Path(root)
        relative_path = current_path.relative_to(root_dir)
        depth = len(relative_path.parts)
        
        # Build structure string
        indent_dir = '    ' * depth + '└── '
        for d in sorted(dirs):
            tree_lines.append(f"{indent_dir}{d}/")

        indent_file = '    ' * depth + '├── '
        for f_name in sorted(files):
            # Check against all ignore conditions
            if f_name in ignore_files or is_binary(Path(f_name), binary_extensions):
                continue
            
            tree_lines.append(f"{indent_file}{f_name}")

            # Consolidate content for this valid file
            file_path = current_path / f_name
            relative_path_str = str(file_path.relative_to(root_dir)).replace('\\', '/')
            
            content_data = {"path": relative_path_str, "content": None}

            if f_name == '.env':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        env_vars = [line.split('=')[0].strip() for line in f if '=' in line and not line.strip().startswith('#')]
                    content_data['content'] = f"[.env file. Variables found: {', '.join(env_vars)}]"
                except Exception as e:
                    content_data['content'] = f"[Error reading .env keys: {e}]"
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_data['content'] = f.read()
                except Exception:
                    content_data['content'] = "[Error: Could not decode file. It may be binary.]"
            
            all_files_data.append(content_data)

    return "\n".join(tree_lines), all_files_data


def count_project_tokens(consolidated_data: list, config: dict) -> int:
    # This function remains the same as before
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nWarning: GEMINI_API_KEY not found. Skipping token count.")
        return 0
    
    os.environ['GOOGLE_API_KEY'] = api_key
    tokenizer_model = config['tokenizer_model_name']
    print(f"\nCounting tokens using model: {tokenizer_model}")

    full_prompt_string = "Project content:\n\n"
    for file_info in consolidated_data:
        full_prompt_string += f"--- File: {file_info['path']} ---\n{file_info['content']}\n\n"
    
    try:
        client = genai.Client()
        response = client.models.count_tokens(model=tokenizer_model, contents=full_prompt_string)
        return response.total_tokens
    except Exception as e:
        print(f"Could not count tokens. Error: {e}")
        return 0

# --- Main Execution Logic ---

def main():
    load_dotenv()
    config = load_config()
    
    parser = argparse.ArgumentParser(description="ArgoPulse AI: Analyzes a project directory.")
    parser.add_argument("project_directory", type=str, help="The path to the project directory.")
    parser.add_argument("--count-tokens", action="store_true", help="Count total tokens of the project.")
    parser.add_argument("--intelligent-filter", action="store_true", help="Use LLM to intelligently filter irrelevant files.")
    args = parser.parse_args()

    project_path = Path(args.project_directory).resolve()
    if not project_path.is_dir():
        print(f"Error: The directory '{project_path}' does not exist.")
        return

    output_dir = Path(f"{project_path.name}-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved in: {output_dir}/")
    
    # --- The New Intelligent Filtering Logic ---
    if args.intelligent_filter:
        print("\nPhase 1: Performing initial scan for intelligent filtering...")
        # We do a quick, lightweight scan first just to get the structure
        initial_structure, _ = process_project_files(project_path, config)
        
        print("Phase 2: Asking LLM to identify files to ignore...")
        llm_filter = get_llm_filter_list(initial_structure, config)
        
        # Save the LLM's decision for review
        llm_analysis_path = output_dir / 'llm_filter_analysis.json'
        with open(llm_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(llm_filter, f, indent=4)
        print(f"✅ LLM analysis saved to '{llm_analysis_path}'")
        
        # Combine config ignore lists with LLM's suggestions
        config['ignore_dirs'].extend(llm_filter.get('ignore_folders', []))
        config['ignore_files'].extend(llm_filter.get('ignore_files', []))
        print("Phase 3: Re-analyzing project with combined ignore list...")

    # --- Final Data Collection ---
    structure_string, consolidated_data = process_project_files(project_path, config)
    
    structure_file_path = output_dir / 'structure.txt'
    with open(structure_file_path, 'w', encoding='utf-8') as f:
        f.write(structure_string)
    print(f"✅ Final file structure saved to '{structure_file_path}'")

    content_file_path = output_dir / 'consolidated_content.json'
    with open(content_file_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=4)
    print(f"✅ Final file contents saved to '{content_file_path}'")
    
    if args.count_tokens:
        total_tokens = count_project_tokens(consolidated_data, config)
        if total_tokens > 0:
            print(f"✅ Total token count for filtered project: {total_tokens}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()