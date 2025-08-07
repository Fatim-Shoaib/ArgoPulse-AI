import os
import json
import argparse
from pathlib import Path

# --- Configuration ---
# We define constants for files/directories to ignore. This is good practice.
IGNORE_DIRS = {'venv', '__pycache__', '.git'}
IGNORE_FILES = {'.gitignore', '.env'}
OUTPUT_STRUCTURE_FILE = 'structure.txt'
OUTPUT_CONTENT_FILE = 'consolidated_content.json'

# --- Core Functions ---

def generate_file_structure(root_dir: Path) -> str:
    """Walks through the directory and creates a visual tree structure string."""
    tree_lines = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Exclude specified directories from being walked further
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        current_path = Path(root)
        # Calculate depth for indentation
        relative_path = current_path.relative_to(root_dir)
        depth = len(relative_path.parts)
        
        # Add the current directory to the tree
        indent = '    ' * (depth - 1) + '└── ' if depth > 0 else ''
        if current_path != root_dir:
            tree_lines.append(f"{indent}{current_path.name}/")

        # Add files in the current directory to the tree
        sub_indent = '    ' * depth + '├── '
        for f in sorted(files):
            tree_lines.append(f"{sub_indent}{f}")
            
    return "\n".join(tree_lines)

def consolidate_file_contents(root_dir: Path) -> list:
    """
    Walks through the directory, reads files, and consolidates their
    content into a list of dictionaries.
    """
    all_files_data = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for filename in files:
            # Exclude specified files
            if filename in IGNORE_FILES:
                continue

            file_path = Path(root) / filename
            relative_path_str = str(file_path.relative_to(root_dir))
            
            content_data = {
                "path": relative_path_str.replace('\\', '/'), # Standardize path separators
                "content": None
            }

            # Special handling for .env file
            if filename == '.env':
                try:
                    # We only read the keys, not the values, for security.
                    with open(file_path, 'r', encoding='utf-8') as f:
                        env_vars = [line.split('=')[0] for line in f if '=' in line]
                    content_data['content'] = f"[.env file. Variables: {', '.join(env_vars)}]"
                except Exception as e:
                    content_data['content'] = f"[Error reading .env keys: {e}]"
            else:
                # General file handling
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_data['content'] = f.read()
                except UnicodeDecodeError:
                    content_data['content'] = "[Error: Could not decode file as UTF-8. It may be a binary file.]"
                except Exception as e:
                    content_data['content'] = f"[Error reading file: {e}]"
            
            all_files_data.append(content_data)
            
    return all_files_data

# --- Main Execution Logic ---

def main():
    """Main function to parse arguments and run the summarization prep."""
    parser = argparse.ArgumentParser(
        description="ArgoPulse AI: Analyzes a project directory and prepares it for summarization."
    )
    parser.add_argument(
        "project_directory", 
        type=str, 
        help="The path to the project directory you want to analyze."
    )
    args = parser.parse_args()

    # Use pathlib for robust path handling
    project_path = Path(args.project_directory)

    if not project_path.is_dir():
        print(f"Error: The directory '{project_path}' does not exist.")
        return

    print(f"Analyzing project at: {project_path}")

    # 1. Generate and save the file structure
    print("Generating file structure...")
    structure_string = generate_file_structure(project_path)
    with open(OUTPUT_STRUCTURE_FILE, 'w', encoding='utf-8') as f:
        f.write(structure_string)
    print(f"✅ File structure saved to '{OUTPUT_STRUCTURE_FILE}'")

    # 2. Consolidate and save all file contents
    print("Consolidating file contents...")
    consolidated_data = consolidate_file_contents(project_path)
    with open(OUTPUT_CONTENT_FILE, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=4)
    print(f"✅ File contents saved to '{OUTPUT_CONTENT_FILE}'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()