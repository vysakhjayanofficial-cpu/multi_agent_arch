# A GOOD PROMPT -- ROLE TASK CONTEXT FORMAT
'''
https://docs.langchain.com/oss/python/langchain/tools
'''


from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langchain.agents import AgentState
from typing import Literal,Union,Annotated,List
from llm import llm
from langchain.messages import AIMessage, HumanMessage, SystemMessage

class CodeState(AgentState):
    """Custom state to track of code which was analyzed"""
    codes: Annotated[List[str],lambda x: x]


@tool("code_reformatter",description="Reformat the code as per requirement specified and create a new file with reformatted code.")
def reformat_code(loc:str,requirement:str,runtime:ToolRuntime) -> str:
    """Reformat the code as per requirement specified and create a new file with reformatted code."""
    with open('debug.txt','w') as debug_file:
        debug_file.write(str(runtime.state["messages"]))
    try:
        with open(loc,'r') as file:
            content = file.read()
            messages = [SystemMessage(content="You are code reformatter. You reformat code as per user requirement with code quality standards."),
                        HumanMessage(content="Code : \n\n"+ content + "\n\n Requirement : "+requirement)]
            with open(f'{loc.split(".")[0]}_reformatted.{loc.split(".")[1]}','w') as new_file:
                new_file.write(llm.invoke(messages).content)
            return f"Reformatted code saved in {loc.split('.')[0]}_reformatted.{loc.split('.')[1]}"
    except Exception as e:
        return str(e)
@tool
def folder_explorer(path: str) -> str:
    """Explore the contents of a folder and return a list of files and subfolders."""
    import os

    try:
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return str(e)
@tool 
def file_reader(file_path: str) -> str:
    """Read the contents of a file and return it's summary as a string."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        messages = [SystemMessage(content="You are a helpful assistant that analyzes file contents and provide a brief description on the content."),
                HumanMessage(content="Content \n\n"+ content + "\n\n Please provide a brief summary of the content.")]
        return llm.invoke(messages).content
    except Exception as e:
        return str(e)
    
system_prompt = '''
You are a file explorer and analyzer agent. 
Your job is to navigate through folders and files and give analytical insights based on their contents.
Your capabilities:
1. You can explore folders to list files and subfolders.
2. You can read files to extract their contents.

Answer format:

Observed Files : <list of point of interest files or folders>
Analytical Insights : <your analysis based on the observed files in 2 -3 sentences>
Goal Acheieved: <Yes/No>
'''

data_ingestor_agent = create_agent(llm,
                                   tools=[folder_explorer, file_reader], 
                                   system_prompt=system_prompt)




program_enhancing_agent = create_agent(llm,tools=[reformat_code],system_prompt="You are a code enhancement agent. Your job is to reformat code files as per user requirements to improve code quality.")

@tool
def invoke_data_ingestor_agent(query:str) -> str:
    """Invoke the data ingestor agent to explore and analyze files.
    
    Use this when you need to find and understand files in the current directory.

    Inputs: Natural Language Request for file exploration and/or analysis.(e.g., "Find the SEC 10k filing program and explain what it does.")
    """
    result  = data_ingestor_agent.invoke({"messages":[HumanMessage(content=query)]})

    return result["messages"][-1].text

@tool
def invoke_program_enhancing_agent(file_path:str,requirement:str) -> str:
    """Invoke the program enhancing agent to reformat code files as per user requirements.
    
    Use this when you need to improve code quality of a specific file.

    Inputs: 
    1. file_path: Path to the code file to be reformatted.
    2. requirement: Natural Language description of the reformatting requirement.(e.g., "Improve code readability and add comments.")
    """
    result  = program_enhancing_agent.invoke({"messages":[HumanMessage(content=f"File Path : {file_path}\n\n Requirement : {requirement}")]})

    return result["messages"][-1].text


SUPERVISOR_SYSTEM_PROMPT = """
You are a supervisor agent having tools to analyze files and reformat programs.
You can traverse folders to find files of interest and analyze their contents.
You can also reformat code files as per user requirements to improve code quality.
When a user request involves mulitple actions, use multiple tools in a sequence to achieve the final goal.
Keep the response concise and to the point.
Follow the format below strictly:
Steps Performed: <describe the steps you took using the tools in a few numbered points>
Observations: <describe your observations from each step in a few numbered points>
Final Response: <provide the final response to the user query based on your observations>
"""

supervisor_agent = create_agent(llm,
                                tools=[invoke_data_ingestor_agent, invoke_program_enhancing_agent],
                                system_prompt=SUPERVISOR_SYSTEM_PROMPT
                                )
query = (
    "I wrote a program to get data from SEC 10k filings in chains folder in current directory with complete connection to SEC api. "
    "I forgot where I saved it. Can you help me find the file and reformat the code to add comments"
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()