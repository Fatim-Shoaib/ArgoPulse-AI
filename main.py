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
        "ignore_files": [".gitignore", "config.json"]
    }
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            # Ensure all keys from default are present
            for key, value in default_config.items():
                config.setdefault(key, value)
            return config
    except FileNotFoundError:
        print("Warning: config.json not found. Using default settings.")
        return default_config
    except json.JSONDecodeError:
        print("Warning: Could not decode config.json. Using default settings.")
        return default_config

# --- Core Functions ---

def generate_file_structure(root_dir: Path, config: dict) -> str:
    """Walks through the directory and creates a visual tree structure string."""
    tree_lines = [f"{root_dir.name}/"]
    ignore_dirs = set(config['ignore_dirs'])

    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        current_path = Path(root)
        relative_path = current_path.relative_to(root_dir)
        depth = len(relative_path.parts)
        
        indent = '    ' * depth + '└── '
        for d in sorted(dirs):
            tree_lines.append(f"{indent}{d}/")

        sub_indent = '    ' * depth + '├── '
        for f in sorted(files):
            if f != '.env':
                tree_lines.append(f"{sub_indent}{f}")
            
    return "\n".join(tree_lines)


def consolidate_file_contents(root_dir: Path, config: dict) -> list:
    """Consolidates file contents, respecting ignored files and directories."""
    all_files_data = []
    ignore_dirs = set(config['ignore_dirs'])
    ignore_files = set(config['ignore_files'])

    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            if filename in ignore_files:
                continue

            file_path = Path(root) / filename
            relative_path_str = str(file_path.relative_to(root_dir)).replace('\\', '/')
            
            content_data = {"path": relative_path_str, "content": None}

            if filename == '.env':
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
            
    return all_files_data


def count_project_tokens(consolidated_data: list, config: dict) -> int:
    """Counts tokens using the Gemini API and model from config."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nWarning: GEMINI_API_KEY not found in .env file. Skipping token count.")
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
    args = parser.parse_args()

    project_path = Path(args.project_directory).resolve()
    if not project_path.is_dir():
        print(f"Error: The directory '{project_path}' does not exist.")
        return

    output_dir = Path(f"{project_path.name}-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved in: {output_dir}/")

    print("Generating file structure...")
    structure_string = generate_file_structure(project_path, config)
    structure_file_path = output_dir / 'structure.txt'
    with open(structure_file_path, 'w', encoding='utf-8') as f:
        f.write(structure_string)
    print(f"✅ File structure saved to '{structure_file_path}'")

    print("Consolidating file contents...")
    consolidated_data = consolidate_file_contents(project_path, config)
    content_file_path = output_dir / 'consolidated_content.json'
    with open(content_file_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=4)
    print(f"✅ File contents saved to '{content_file_path}'")
    
    if args.count_tokens:
        total_tokens = count_project_tokens(consolidated_data, config)
        if total_tokens > 0:
            print(f"✅ Total token count for the project: {total_tokens}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()