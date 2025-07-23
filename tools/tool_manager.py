import os
import shutil

class ToolManager:
    def __init__(self, config):
        self.config = config
        self.tools_dir = config['paths']['tools']
        
    def list_tools(self):
        """List all available tools"""
        tools = []
        tool_files = os.listdir(self.tools_dir)
        for tool_file in tool_files:
            tool_path = os.path.join(self.tools_dir, tool_file)
            if os.path.isfile(tool_path):
                tools.append({
                    "name": tool_file,
                    "type": self.determine_tool_type(tool_file)
                })
        return tools
    
    def determine_tool_type(self, filename):
        """Determine the type of tool based on filename extension"""
        _, ext = os.path.splitext(filename)
        if ext in ['.py', '.sh', '.bat', '.cmd']:
            return "Script"
        elif ext in ['.exe', '.bin', '.dll']:
            return "Binary"
        else:
            return "Unknown"
    
    def install_tool(self, tool_url, tool_name):
        """Download and install a tool from a given URL"""
        download_path = os.path.join(self.tools_dir, tool_name)
        
        # Placeholder download logic (mocked for demonstration)
        print(f"Downloading tool from {tool_url}...")
        with open(download_path, 'w') as f:
            f.write("# Placeholder for tool content")
        print(f"Tool {tool_name} installed successfully at {download_path}.")
    
    def remove_tool(self, tool_name):
        """Remove a tool by its name"""
        tool_path = os.path.join(self.tools_dir, tool_name)
        if os.path.exists(tool_path):
            os.remove(tool_path)
            print(f"Tool {tool_name} removed successfully.")
        else:
            print(f"Tool {tool_name} not found.")
    
    def clear_all_tools(self):
        """Clear all tools from the directory"""
        shutil.rmtree(self.tools_dir)
        os.makedirs(self.tools_dir, exist_ok=True)
        print("All tools cleared.")
