from typing import Dict, Any

class ToolExecutor:
    """
    A tool executor responsible for managing and executing tools.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        Register a new tool in the toolbox.
        """
        if name in self.tools:
            print(f"Warning: Tool '{name}' already exists and will be overwritten.")
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool '{name}' registered.")

    def getTool(self, name: str) -> callable:
        """
        Get a tool's execution function by name.
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        Get a formatted description string of all available tools.
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

# --- Tool Initialization and Usage Example ---
if __name__ == '__main__':
    import WebSearch

    # 1. Initialize tool executor
    toolExecutor = ToolExecutor()

    # 2. Register our practical search tool
    search_description = "A web search engine. Use this tool when you need to answer questions about current events, facts, and information not found in your knowledge base."
    toolExecutor.registerTool("Search", search_description, WebSearch.ddgs_search)

    # 3. Print available tools
    print("\n--- Available Tools ---")
    print(toolExecutor.getAvailableTools())

    # 4. Agent's Action call, this time we ask a real-time question
    print("\n--- Execute Action: Search['What is NVIDIA's latest GPU model'] ---")
    tool_name = "Search"
    tool_input = "What is NVIDIA's latest GPU model"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- Observation ---")
        print(observation)
    else:
        print(f"Error: Tool named '{tool_name}' not found.")

