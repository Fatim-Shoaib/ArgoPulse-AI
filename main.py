import os
import json
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from logging_config import setup_logging

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration Loader ---
def load_config():
    # This function is correct and remains unchanged
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
        logger.warning("config.json not found or invalid. Using default settings.")
        return default_config

# --- LLM-Powered Filtering Function (Using Your Provided API Logic) ---
def get_llm_filter_list(structure_string: str, config: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Cannot perform intelligent filtering: GEMINI_API_KEY not found.")
        return {"ignore_folders": [], "ignore_files": []}
    
    # Set the key for the client to use
    os.environ['GOOGLE_API_KEY'] = api_key
    
    prompt = f"""
You are an expert software developer. Your task is to review the following file structure and identify which files and folders are NOT necessary for understanding the core logic of the code.

The goal is to prepare the project for a code summary. Exclude things like:
- Dependency directories (e.g., node_modules, venv)
- Build output folders (e.g., dist, build)
- Generic configuration files (e.g., package-lock.json)
- Test data, assets, or documentation unless it's critical.

Analyze this file structure:
---
{structure_string}
---

Your response MUST be a valid JSON object with two keys: "ignore_folders" and "ignore_files". Each key should have a list of strings as its value. Use forward slashes for paths relative to the project root. Folders should end with a slash.

Example response format:
{{
  "ignore_folders": ["node_modules/", "dist/"],
  "ignore_files": ["assets/logo.svg", "package-lock.json"]
}}
"""
    logger.debug(f"LLM Prompt for filtering:\n{prompt}")

    try:
        llm_model = config['generation_model_name']
        client = genai.Client()
        response = client.models.generate_content(
            model=llm_model, contents=prompt
        )        
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        logger.debug(f"Raw LLM response:\n{cleaned_response}")

        filter_data = json.loads(cleaned_response)
        logger.info(f"LLM suggested ignoring: {filter_data}")
        
        if isinstance(filter_data, dict) and "ignore_folders" in filter_data and "ignore_files" in filter_data:
            return filter_data
        else:
            logger.warning("LLM response was not in the expected format. Ignoring.")
            return {"ignore_folders": [], "ignore_files": []}

    except Exception as e:
        logger.error(f"An error occurred during LLM filtering. Check log for details. Error: {e}")
        logger.debug("Traceback:", exc_info=True)
        return {"ignore_folders": [], "ignore_files": []}

# --- CORRECTED HIERARCHY AND FILE PROCESSING LOGIC ---

def discover_all_paths(root_dir: Path, config: dict) -> list[Path]:
    """Walks the directory and returns a list of all relative file and directory paths."""
    all_paths = []
    initial_ignore_dirs = set(config.get('ignore_dirs', []))
    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in initial_ignore_dirs]
        for name in files:
            all_paths.append(Path(root) / name)
        # We add directories too, so we can represent empty ones in the tree
        for name in dirs:
            all_paths.append(Path(root) / name)
    return all_paths

def build_file_tree(paths: list[Path], root_dir: Path):
    """Builds a nested dictionary representing the file tree from a list of paths."""
    tree = {'name': root_dir.name, 'type': 'directory', 'children': []}
    
    for path in sorted(paths):
        current_level = tree['children']
        relative_parts = path.relative_to(root_dir).parts
        
        for i, part in enumerate(relative_parts):
            node = next((child for child in current_level if child['name'] == part), None)
            
            if not node:
                is_last_part = i == len(relative_parts) - 1
                node_type = 'file' if path.is_file() and is_last_part else 'directory'
                node = {'name': part, 'type': node_type, 'children': []}
                current_level.append(node)
            
            current_level = node['children']
    return tree

def format_tree(node, prefix=""):
    """Recursively formats the file tree dictionary into a printable string."""
    lines = []
    children = sorted(node.get('children', []), key=lambda x: (x['type'], x['name']))
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        name = child['name'] + ('/' if child['type'] == 'directory' else '')
        lines.append(f"{prefix}{connector}{name}")
        
        if child.get('children'):
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(format_tree(child, new_prefix))
    return lines

def is_path_ignored(relative_path_str: str, full_ignore_list: list) -> bool:
    """Checks if a given relative path string should be ignored based on the master list."""
    for item in full_ignore_list:
        # Check for directory ignore (e.g., "data/")
        if item.endswith('/') and relative_path_str.startswith(item):
            return True
        # Check for exact file ignore (e.g., "app.py")
        elif not item.endswith('/') and relative_path_str == item:
            return True
    return False

# --- Token Counting Function (Using Your Provided API Logic) ---
def count_project_tokens(consolidated_data: list, config: dict) -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found. Skipping token count.")
        return 0
    
    os.environ['GOOGLE_API_KEY'] = api_key
    tokenizer_model = config['tokenizer_model_name']
    logger.info(f"Counting tokens using model: {tokenizer_model}")
    full_prompt_string = "Project content:\n\n"
    for file_info in consolidated_data:
        full_prompt_string += f"--- File: {file_info['path']} ---\n{file_info['content']}\n\n"
    
    try:
        client = genai.Client()
        response = client.models.count_tokens(model=tokenizer_model, contents=full_prompt_string)
        return response.total_tokens
    except Exception as e:
        logger.error(f"Could not count tokens. Check log for details. Error: {e}")
        logger.debug("Traceback:", exc_info=True)
        return 0

# --- Main Execution Logic (Using Refactored Discover -> Filter -> Generate Pattern) ---
def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="ArgoPulse AI: Analyzes a project directory.")
    parser.add_argument("project_directory", type=str)
    parser.add_argument("--count-tokens", action="store_true")
    parser.add_argument("--intelligent-filter", action="store_true")
    args = parser.parse_args()
    
    load_dotenv()

    project_path = Path(args.project_directory).resolve()
    if not project_path.is_dir():
        logger.error(f"The directory '{project_path}' does not exist.")
        return

    output_dir = Path(f"{project_path.name}-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved in: {output_dir}/")

    try:
        # Step 1: Discover all paths initially
        all_paths = discover_all_paths(project_path, config)
        
        # Create a simple string version of the initial structure for the LLM
        initial_tree_for_llm = build_file_tree(all_paths, project_path)
        initial_structure_str = f"{project_path.name}/\n" + "\n".join(format_tree(initial_tree_for_llm))
        
        # Step 2: Build the complete filter list
        full_ignore_list = config['ignore_files'] + [d.strip('/')+'/' for d in config['ignore_dirs']]
        
        if args.intelligent_filter:
            logger.info("Asking LLM to identify files to ignore...")
            llm_filter = get_llm_filter_list(initial_structure_str, config)
            
            with open(output_dir / 'llm_filter_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(llm_filter, f, indent=4)
            logger.info(f"LLM analysis saved to '{output_dir / 'llm_filter_analysis.json'}'")
            
            # Add LLM suggestions to the master ignore list
            full_ignore_list.extend(llm_filter.get('ignore_files', []))
            full_ignore_list.extend(llm_filter.get('ignore_folders', []))

        logger.debug(f"Final combined ignore list: {full_ignore_list}")

        # Filter the master path list to get our final list of paths to process
        binary_ext = set(config['binary_file_extensions'])
        final_paths = []
        for path in all_paths:
            relative_str = str(path.relative_to(project_path)).replace('\\', '/')
            if not is_path_ignored(relative_str, full_ignore_list) and (path.is_dir() or path.suffix not in binary_ext):
                final_paths.append(path)
        
        logger.info(f"Discovered {len(all_paths)} total paths. Using {len(final_paths)} after filtering.")

        # Step 3: Generate outputs from the final, clean list
        final_tree = build_file_tree(final_paths, project_path)
        final_structure_str = f"{project_path.name}/\n" + "\n".join(format_tree(final_tree))
        
        with open(output_dir / 'structure.txt', 'w', encoding='utf-8') as f:
            f.write(final_structure_str)
        logger.info(f"Final file structure saved to '{output_dir / 'structure.txt'}'")

        # Consolidate content ONLY from the files in the final, filtered list
        consolidated_data = []
        for path in sorted(final_paths):
            if path.is_file(): # Only files have content
                relative_path_str = str(path.relative_to(project_path)).replace('\\', '/')
                content_data = {"path": relative_path_str, "content": None}
                if path.name == '.env':
                    content_data['content'] = "[.env content is hidden]"
                else:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content_data['content'] = f.read()
                    except Exception:
                        content_data['content'] = "[Error: Could not decode file.]"
                consolidated_data.append(content_data)

        with open(output_dir / 'consolidated_content.json', 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=4)
        logger.info(f"Final file contents saved to '{output_dir / 'consolidated_content.json'}'")
        
        if args.count_tokens:
            total_tokens = count_project_tokens(consolidated_data, config)
            if total_tokens > 0:
                logger.info(f"Total token count for filtered project: {total_tokens}")

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()