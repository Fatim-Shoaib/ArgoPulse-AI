import os
import json
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from logging_config import setup_logging
from prompt import HIGH_LEVEL_SUMMARY_PROMPT, MEDIUM_LEVEL_SUMMARY_PROMPT, LOW_LEVEL_SUMMARY_PROMPT

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration Loader ---
def load_config():
    default_config = {
        "generation_model_name": "gemini-1.5-pro-latest",
        "tokenizer_model_name": "models/gemini-1.5-flash-latest",
        "ignore_dirs": ["__pycache__", ".git", "venv"],
        "ignore_files": [".gitignore", "config.json"],
        "binary_file_extensions": [".pyc"],
        "data_file_extensions": [".csv", ".json"],
        "data_file_head_lines": 5
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

def get_llm_filter_list(structure_string: str, config: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Cannot perform intelligent filtering: GEMINI_API_KEY not found.")
        return {"ignore_folders": [], "ignore_files": []}
    
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

def generate_summary(prompt_template: str, structure_content: str, consolidated_content: str, config: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Cannot generate summary: GEMINI_API_KEY not found.")
        return ""
    
    os.environ['GOOGLE_API_KEY'] = api_key
    
    final_prompt = f"""{prompt_template}

Here is the content of the project files:

--- consolidated_content.json ---
{consolidated_content}

--- structure.txt ---
{structure_content}
"""
    
    logger.debug("Sending summary generation request to LLM...")
    try:
        llm_model = config['generation_model_name']
        client = genai.Client()
        response = client.models.generate_content(
            model=llm_model, contents=final_prompt
        )
        summary_text = response.text.strip()
        if summary_text.startswith("```markdown"):
            summary_text = summary_text[10:]
        if summary_text.endswith("```"):
            summary_text = summary_text[:-3]
        
        return summary_text.strip()

    except Exception as e:
        logger.error(f"An error occurred during summary generation: {e}")
        logger.debug("Traceback:", exc_info=True)
        return ""

def discover_all_paths(root_dir: Path, config: dict) -> list[Path]:
    all_paths = []
    initial_ignore_dirs = set(config.get('ignore_dirs', []))
    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in initial_ignore_dirs]
        for name in files:
            all_paths.append(Path(root) / name)
        for name in dirs:
            all_paths.append(Path(root) / name)
    return all_paths

def build_file_tree(paths: list[Path], root_dir: Path):
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
    for item in full_ignore_list:
        if item.endswith('/') and relative_path_str.startswith(item):
            return True
        elif not item.endswith('/') and relative_path_str == item:
            return True
    return False

# --- Main Execution Logic ---
def main():
    config = load_config()
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="ArgoPulse AI: Analyze projects and generate summaries.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "project_path", 
        type=str, 
        help="The path to the project directory you want to process."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform analysis: generate structure.txt and consolidated_content.json."
    )
    parser.add_argument(
        "--summarize",
        nargs='+',
        choices=['low', 'medium', 'high'],
        help="Generate summaries for the specified levels (e.g., --summarize low high)."
    )
    parser.add_argument(
        "--intelligent-filter",
        action="store_true",
        help="Use LLM to filter files during analysis. Only used with --analyze."
    )
    parser.add_argument(
        "--count-tokens",
        action="store_true",
        help="Count tokens of consolidated content. Only used with --analyze."
    )
    args = parser.parse_args()

    if not args.analyze and not args.summarize:
        parser.print_help()
        logger.info("\nNo action specified. Please use --analyze or --summarize.")
        return

    project_path = Path(args.project_path).resolve()
    output_dir = Path(f"{project_path.name}-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze:
        logger.info(f"--- Starting Analysis for: {project_path.name} ---")
        if not project_path.is_dir():
            logger.error(f"Project directory does not exist: {project_path}")
            return
        
        all_paths = discover_all_paths(project_path, config)
        initial_tree_for_llm = build_file_tree(all_paths, project_path)
        initial_structure_str = f"{project_path.name}/\n" + "\n".join(format_tree(initial_tree_for_llm))
        full_ignore_list = config['ignore_files'] + [d.strip('/')+'/' for d in config['ignore_dirs']]
        
        if args.intelligent_filter:
            logger.info("Asking LLM to identify files to ignore...")
            llm_filter = get_llm_filter_list(initial_structure_str, config)
            with open(output_dir / 'llm_filter_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(llm_filter, f, indent=4)
            full_ignore_list.extend(llm_filter.get('ignore_files', []))
            full_ignore_list.extend(llm_filter.get('ignore_folders', []))

        binary_ext = set(config['binary_file_extensions'])
        final_paths = [p for p in all_paths if not is_path_ignored(str(p.relative_to(project_path)).replace('\\', '/'), full_ignore_list) and (p.is_dir() or p.suffix not in binary_ext)]
        
        logger.info(f"Discovered {len(all_paths)} total paths. Using {len(final_paths)} after filtering.")
        
        final_tree = build_file_tree(final_paths, project_path)
        final_structure_str = f"{project_path.name}/\n" + "\n".join(format_tree(final_tree))
        with open(output_dir / 'structure.txt', 'w', encoding='utf-8') as f:
            f.write(final_structure_str)
        logger.info(f"File structure saved to '{output_dir / 'structure.txt'}'")

        consolidated_data = []
        data_file_ext = set(config.get('data_file_extensions', []))
        head_lines_count = config.get('data_file_head_lines', 5)
        for path in sorted(p for p in final_paths if p.is_file()):
            content_data = {"path": str(path.relative_to(project_path)).replace('\\', '/'), "content": None}
            if path.name == '.env': content_data['content'] = "[.env content is hidden]"
            elif path.suffix in data_file_ext:
                try:
                    with open(path, 'r', encoding='utf-8') as f: content_data['content'] = f"[Preview of data file (first {head_lines_count} lines)]:\n{''.join([next(f, '') for _ in range(head_lines_count)])}"
                except Exception: content_data['content'] = "[Error reading preview of data file]"
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as f: content_data['content'] = f.read()
                except Exception: content_data['content'] = "[Error: Could not decode file.]"
            consolidated_data.append(content_data)
        
        with open(output_dir / 'consolidated_content.json', 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=4)
        logger.info(f"Consolidated content saved to '{output_dir / 'consolidated_content.json'}'")
        
        if args.count_tokens:
            total_tokens = count_project_tokens(consolidated_data, config)
            if total_tokens > 0: logger.info(f"Total token count for filtered project: {total_tokens}")
        logger.info(f"--- Analysis Complete ---")

    if args.summarize:
        logger.info(f"--- Starting Summarization for: {project_path.name} ---")
        structure_file = output_dir / 'structure.txt'
        content_file = output_dir / 'consolidated_content.json'
        
        if not structure_file.exists() or not content_file.exists():
            logger.error(f"Analysis files not found in '{output_dir}'. Please run with --analyze first.")
            return

        with open(structure_file, 'r', encoding='utf-8') as f:
            structure_content = f.read()
        with open(content_file, 'r', encoding='utf-8') as f:
            consolidated_content = f.read()
            
        prompt_map = {
            'low': LOW_LEVEL_SUMMARY_PROMPT,
            'medium': MEDIUM_LEVEL_SUMMARY_PROMPT,
            'high': HIGH_LEVEL_SUMMARY_PROMPT
        }
        
        for level in args.summarize:
            logger.info(f"Generating {level}-level summary...")
            prompt_template = prompt_map[level]
            summary_text = generate_summary(prompt_template, structure_content, consolidated_content, config)
            
            if summary_text:
                summary_filename = f"{level}_level_summary.md"
                with open(output_dir / summary_filename, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                logger.info(f"✅ {level.capitalize()}-level summary saved to '{output_dir / summary_filename}'")
        logger.info(f"--- Summarization Complete ---")


if __name__ == "__main__":
    main()