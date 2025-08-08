HIGH_LEVEL_SUMMARY_PROMPT = '''
You are being given two (or three) files:

1. A consolidated_content.json file, which contains all the code and relevant content.
2. A structure.txt file, which explains the structure of the project (likely directory and file layout).
3. A research paper (optional - might not be given)

# Task
Your task is to generate a high-level summary documentation of the entire pipeline.
This documentation should include:
- A description of what the pipeline does (including the inputs, processing and outputs of the complete pipeline).
- The major components of the pipeline, step by step.
- How data flows across the pipeline.
- Where LLMs are used (if any) and for what.

# Instructions
- Your output should be in raw markdown
- Do NOT render the markdown — just output raw markdown syntax as plain text, using backticks
- Do NOT split the output into multiple responses. Give the entire documentation in a single response, even if it is very long
- Use <detail> tags to make the document more user-friendly
- Do NOT stylize, format, or visually collapse sections in your output — output everything as raw, literal text
- Please provide the raw markdown for your output, enclosed in a markdown code block.
'''

MEDIUM_LEVEL_SUMMARY_PROMPT = '''
You are being given two (or three) files:

1. A consolidated_content.json file, which contains all the code and relevant content.
2. A structure.txt file, which explains the structure of the project (likely directory and file layout).
3. A research paper (optional - might not be given)

# Task
Your task is to generate a detailed summary documentation of the entire pipeline.
This documentation should include:
- A description of what the pipeline does (including the inputs, processing and outputs of the complete pipeline).
- The major components of the pipeline, step by step.
- How data flows across the pipeline.
- Where LLMs are used (if any) and for what.
- Details of all the relevant files, and how they interact with each other and their purpose in the overall pipeline.
- Purposes of functions/components/classes in the files and how they interact with the file and their overall purpose in the pipeline.

# Instructions
- Your output should be in raw markdown
- Do NOT render the markdown — just output raw markdown syntax as plain text, using backticks
- Do NOT split the output into multiple responses. Give the entire documentation in a single response, even if it is very long
- Use <detail> tags to make the document more user-friendly
- Do NOT stylize, format, or visually collapse sections in your output — output everything as raw, literal text
- Please provide the raw markdown for your output, enclosed in a markdown code block.
'''

LOW_LEVEL_SUMMARY_PROMPT = '''
You are being given two (or three) files:

1. A consolidated_content.json file, which contains all the code and relevant content.
2. A structure.txt file, which explains the structure of the project (likely directory and file layout).
3. A research paper (optional - might not be given)

# Task
Your task is to generate a low-level, extremely detailed summary documentation of the entire pipeline.
This documentation should include:
- A description of what the pipeline does (including the inputs, processing and outputs of the complete pipeline).
- The major components of the pipeline, step by step.
- How data flows across the pipeline.
- Where LLMs are used (if any) and for what.
- Details of ALL the relevant files, and how they interact with each other and their purpose in the overall pipeline.
- Details of ALL (every single) functions in all the relevant files. This includes (but is not limited): input of function, output of function, process in the function, purpose of function in the overall pipeline.
- Details of ALL other components in a similar way (for example, if the files contain classes, give details of all classes, if it contains any other modularization, like objects from predefined classes, etc, then do for them as well)
- Details of ALL (every single) functions/components/classes in the relelvant files and how they interact with the file and their overall purpose in the pipeline. This should also include their inputs, procces and outputs.
- Just make sure that ALL parts of every single relevant file are covered properly.

# Instructions
- Your output should be in raw markdown
- Do NOT render the markdown — just output raw markdown syntax as plain text, using backticks
- Do NOT split the output into multiple responses. Give the entire documentation in a single response, even if it is very long
- Use <detail> tags to make the document more user-friendly
- Do NOT stylize, format, or visually collapse sections in your output — output everything as raw, literal text
- Please provide the raw markdown for your output, enclosed in a markdown code block.
'''

# LOW_LEVEL_SUMMARY_PROMPT='''
# You are being given two (or three) files:

# 1. A consolidated_content.json file, which contains all the code and relevant content.
# 2. A structure.txt file, which explains the structure of the project (likely directory and file layout).
# 3. A research paper (optional - might not be given)

# # Task
# Your task is to generate comprehensive documentation of the entire pipeline. The documentation must include the following section:

# FUNCTION-LEVEL SUMMARIES (Low-Level Overview)

# You must give a detailed breakdown of every single function found in the consolidated_content.json file. For each function, you must include all of the following:
# - Function name
# - Location (file/module) where it is defined
# - Inputs (parameters, expected types or formats)
# - Outputs (return values, side effects)
# - Process (what the function does internally, step-by-step if needed)
# - Purpose in the overall pipeline (why this function exists and what role it plays)
# - How it connects to the rest of the pipeline

# Also include and summarize any prompt templates found in the code:

# - Show where the prompts are used (e.g., with LLMs),
# - What kind of task they are prompting for,
# - What kind of outputs are expected from the LLM,
# - How they fit into the system logic.

# If there are any other modularizations/components (such as classes, agents, objects, etc.), then give the complete details of that as well, similar to the instructions I gave you for functions.

# # Instructions
# - Your output should be in raw markdown
# - Do NOT render the markdown — just output raw markdown syntax as plain text, using backticks
# - Do NOT split the output into multiple responses. Give the entire documentation in a single response, even if it is very long
# - Use <detail> tags to make the document more user-friendly
# - Do NOT stylize, format, or visually collapse sections in your output — output everything as raw, literal text
# - Please provide the raw markdown for your output, enclosed in a markdown code block.
# '''

# LOW_LEVEL_SUMMARY_PROMPT='''
# You will be given two (or three) files:

# 1. A `.json` file that contains the contents of only the **relevant** files in a project directory.
# 2. A `.txt` file that describes the **hierarchical folder structure** of only those relevant files.
# 3. A research paper about the project. (optional, might not be given to you)

# ---

# Your task is to **reverse-engineer** the entire system based on these files and provide a **complete technical pipeline explanation**.
# Your output should be a **structured, raw markdown document**, enclosed within a markdown code block (use triple backticks and specify `markdown`, like this: ```` ```markdown ````).

# ---

# ### Goals:

# You must create documentation that includes:

# #### High-level pipeline overview
# - Describe what the system does overall.
# - Summarize the core phases of the pipeline (e.g., input processing, file selection, aggregation, summarization, etc.)

# #### Mid- AND **LOW-LEVEL** DETAILED component breakdown

# I need a DETAILED breakdown of EVERY SINGLE function and agent and object and class and any other modular component in all of the files I have given.
# Each breakdown NEEDS to include:
# 1. **Purpose:** What is it for?
# 2. **Input:** What input does it receive?
# 3. **Process:** What logic or algorithm does it apply?
# 4. **Output:** What does it return or produce?
# 5. **Role in Pipeline:** How does this fit into the overall system?

# The components that I need these descriptions for include (but are not limited to):
# - Python functions
# - Classes
# - LLM agent definitions
# - Prompt templates
# - Objects of imported libraries
# - Specialized config files
# - Or any helper logic

# #### Instructions:
# - Use nested **`<details>`** tags to structure explanations.
# - High-level modules → mid-level components → low-level logic.
# - Use clear markdown sectioning: `#`, `##`, `###` etc.
# - Do NOT render the markdown — just output raw markdown syntax as plain text, using backticks
# - Do NOT split the output into multiple responses. Give the entire documentation in a single response, even if it is very long.
# - It is okay if your response is very long. You just need to make sure that you give a complete response.
# - Please provide the raw markdown for your output, enclosed in a markdown code block.
# '''