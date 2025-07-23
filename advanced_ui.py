#!/usr/bin/env python3
"""
Advanced ASCII UI for Agentic AI System
======================================
A comprehensive interface with parameter editing, history, verbosity,
and specialized agent assignment for Python and PineScript development.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pickle

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from agentic_ai_system import AgenticAISystem, TaskType, AgentConfig, AgentType

class AdvancedAgenticUI:
    """Advanced ASCII UI with comprehensive control features"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = models_dir
        self.system = AgenticAISystem(models_dir)
        self.history_file = "agentic_history.pkl"
        self.config_file = "agentic_config.json"
        self.load_history()
        self.load_config()
        
        # UI State
        self.verbosity = self.config.get("verbosity", False)
        self.current_mode = TaskType.GENERAL_QUERY
        self.parameters = self.config.get("parameters", {
            "temperature": 0.7,
            "max_tokens": 2048,
            "confidence_threshold": 0.3,
            "multi_agent_threshold": 2
        })
        
        # Specialized agent assignments for coding
        self.coding_specialists = {
            "python": {
                "primary": "Coder-CodeGen25",
                "secondary": "Reasoner-Qwen3",
                "description": "CodeGen 2.5 7B Mono - Specialized for Python code generation"
            },
            "pinescript": {
                "primary": "Coder-CodeGen25", 
                "secondary": "Creative-OpenHermes",
                "description": "CodeGen 2.5 + OpenHermes for trading script creation"
            },
            "general_code": {
                "primary": "Coder-CodeGen25",
                "secondary": "Reasoner-Qwen3",
                "description": "General programming tasks"
            }
        }
        
    def load_history(self):
        """Load command history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.history = pickle.load(f)
            else:
                self.history = []
        except:
            self.history = []
    
    def save_history(self):
        """Save command history to file"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.history, f)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {}
        except:
            self.config = {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_to_save = {
                "verbosity": self.verbosity,
                "parameters": self.parameters,
                "current_mode": self.current_mode.value
            }
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_banner(self):
        """Print the main banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🤖 AGENTIC AI SYSTEM UI 🤖                           ║
║                          Advanced Control Interface                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📋 MAIN COMMANDS:                    📊 SYSTEM COMMANDS:                   ║
║  • help          - Show all commands  • status      - System status         ║
║  • query <text>  - Process query      • history     - View history          ║
║  • mode <type>   - Change task mode   • clear       - Clear history         ║
║  • agents        - Show agents        • save        - Save configuration    ║
║                                                                              ║
║  🔧 PARAMETER COMMANDS:               🐍 CODING COMMANDS:                   ║
║  • set <param>=<val> - Set parameter  • python <code> - Python specialist  ║
║  • get <param>       - Get parameter  • pine <code>   - PineScript spec.   ║
║  • params            - Show all       • coding       - General coding      ║
║  • verbose           - Toggle verbose                                       ║
║                                                                              ║
║  💾 FILE COMMANDS:                    🚪 EXIT:                             ║
║  • export     - Export history        • exit         - Exit system          ║
║  • load       - Load session          • quit         - Same as exit         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        
        # Show current status
        print(f"📍 Current Mode: {self.current_mode.value}")
        print(f"🔊 Verbosity: {'ON' if self.verbosity else 'OFF'}")
        print(f"🎯 Confidence Threshold: {self.parameters['confidence_threshold']}")
        print("─" * 80)
    
    def print_help(self):
        """Print detailed help information"""
        help_text = """
🆘 DETAILED HELP - AGENTIC AI SYSTEM UI

📋 QUERY COMMANDS:
  query <text>     - Process a query with current mode
  python <code>    - Process Python code request (uses CodeGen25 + Qwen3)
  pine <script>    - Process PineScript request (uses CodeGen25 + OpenHermes)
  coding <task>    - General coding task

🎯 MODE COMMANDS:
  mode general     - General queries
  mode code        - Code generation
  mode creative    - Creative writing
  mode analysis    - Deep analysis
  mode conversation- Conversation mode
  mode planning    - Planning tasks

🔧 PARAMETER COMMANDS:
  set temperature=0.8    - Set model temperature (0.1-2.0)
  set max_tokens=4096    - Set max response tokens
  set confidence_threshold=0.5 - Set minimum confidence
  get temperature        - Show current temperature
  params                 - Show all parameters

📊 SYSTEM COMMANDS:
  agents          - Show all available agents and their status
  status          - Detailed system status
  history         - Show recent commands and responses
  clear           - Clear command history
  verbose         - Toggle verbose output

🎨 SPECIALIZED CODING AGENTS:
  Python Tasks    → Primary: CodeGen 2.5 7B, Secondary: Qwen3-8B
  PineScript      → Primary: CodeGen 2.5 7B, Secondary: OpenHermes-2.5
  General Code    → Primary: CodeGen 2.5 7B, Secondary: Qwen3-8B

💾 SESSION MANAGEMENT:
  save            - Save current configuration
  export          - Export history to JSON file
  load            - Load previous session
  
📝 EXAMPLES:
  query "Explain machine learning"
  python "Create a web scraper"
  pine "Write a moving average strategy"
  set temperature=0.9
  mode creative
"""
        print(help_text)
    
    async def show_agents(self):
        """Show detailed agent information"""
        print("\n🤖 AGENT INFORMATION & SPECIALIZATIONS")
        print("═" * 60)
        
        status = self.system.get_system_status()
        
        for agent in status['available_agents']:
            status_icon = "✅" if agent['available'] else "❌"
            print(f"{status_icon} {agent['name']} ({agent['type']})")
            print(f"   📋 Specialties: {', '.join(agent['specialties'])}")
            
            # Show coding specializations
            if agent['type'] == 'coding':
                print("   🐍 Python & PineScript Primary Agent")
            elif agent['type'] == 'reasoning':
                print("   🧠 Python Secondary & Analysis Agent")
            elif agent['type'] == 'creative':
                print("   ✨ PineScript Secondary & Creative Agent")
            
            print()
        
        print("🎯 CODING SPECIALIZATIONS:")
        for lang, spec in self.coding_specialists.items():
            print(f"   {lang.upper()}: {spec['description']}")
    
    async def show_status(self):
        """Show comprehensive system status"""
        if self.verbosity:
            print("🔍 [VERBOSE] Fetching system status...")
        
        status = self.system.get_system_status()
        
        print("\n📊 SYSTEM STATUS")
        print("═" * 40)
        print(f"🤖 Total Agents: {status['total_agents']}")
        print(f"🎯 Orchestrator: {'✅ Available' if status['orchestrator_available'] else '❌ Unavailable'}")
        print(f"📍 Current Mode: {self.current_mode.value}")
        print(f"🔊 Verbosity: {'ON' if self.verbosity else 'OFF'}")
        print(f"📜 History Entries: {len(self.history)}")
        
        print("\n🔧 CURRENT PARAMETERS:")
        for key, value in self.parameters.items():
            print(f"   {key}: {value}")
        
        print(f"\n📁 Models Directory: {self.models_dir}")
        
        # Agent availability summary
        available_count = sum(1 for agent in status['available_agents'] if agent['available'])
        print(f"🟢 Available Agents: {available_count}/{status['total_agents']}")
    
    def show_history(self, limit: int = 10):
        """Show recent command history"""
        if self.verbosity:
            print("🔍 [VERBOSE] Displaying command history...")
        
        if not self.history:
            print("📜 No history found.")
            return
        
        print(f"\n📜 RECENT HISTORY (Last {limit} entries)")
        print("═" * 50)
        
        recent_history = self.history[-limit:] if len(self.history) > limit else self.history
        
        for i, entry in enumerate(recent_history, 1):
            timestamp = entry.get('timestamp', 'Unknown time')
            command = entry.get('command', 'Unknown command')
            response_summary = entry.get('response_summary', 'No response')
            
            print(f"{i:2d}. [{timestamp}]")
            print(f"    Command: {command}")
            print(f"    Result:  {response_summary}")
            print()
    
    def set_parameter(self, param_string: str):
        """Set a parameter value"""
        try:
            key, value = param_string.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "temperature":
                value = float(value)
                if 0.1 <= value <= 2.0:
                    self.parameters[key] = value
                    print(f"✅ Temperature set to {value}")
                else:
                    print("❌ Temperature must be between 0.1 and 2.0")
            
            elif key == "max_tokens":
                value = int(value)
                if 1 <= value <= 8192:
                    self.parameters[key] = value
                    print(f"✅ Max tokens set to {value}")
                else:
                    print("❌ Max tokens must be between 1 and 8192")
            
            elif key == "confidence_threshold":
                value = float(value)
                if 0.0 <= value <= 1.0:
                    self.parameters[key] = value
                    print(f"✅ Confidence threshold set to {value}")
                else:
                    print("❌ Confidence threshold must be between 0.0 and 1.0")
                    
            elif key == "multi_agent_threshold":
                value = int(value)
                if 1 <= value <= 5:
                    self.parameters[key] = value
                    print(f"✅ Multi-agent threshold set to {value}")
                else:
                    print("❌ Multi-agent threshold must be between 1 and 5")
            
            else:
                print(f"❌ Unknown parameter: {key}")
                self.show_available_parameters()
                
        except ValueError as e:
            print(f"❌ Invalid value format: {e}")
        except Exception as e:
            print(f"❌ Error setting parameter: {e}")
    
    def get_parameter(self, key: str):
        """Get a parameter value"""
        if key in self.parameters:
            print(f"📋 {key} = {self.parameters[key]}")
        else:
            print(f"❌ Parameter '{key}' not found")
            self.show_available_parameters()
    
    def show_available_parameters(self):
        """Show all available parameters"""
        print("\n🔧 AVAILABLE PARAMETERS:")
        print("─" * 30)
        for key, value in self.parameters.items():
            print(f"  {key}: {value}")
        print("\n💡 Use 'set <param>=<value>' to change parameters")
    
    def change_mode(self, mode_str: str):
        """Change the current task mode"""
        mode_map = {
            "general": TaskType.GENERAL_QUERY,
            "code": TaskType.CODE_GENERATION,
            "coding": TaskType.CODE_GENERATION,
            "creative": TaskType.CREATIVE_WRITING,
            "analysis": TaskType.ANALYSIS,
            "conversation": TaskType.CONVERSATION,
            "planning": TaskType.PLANNING,
            "review": TaskType.CODE_REVIEW
        }
        
        if mode_str in mode_map:
            self.current_mode = mode_map[mode_str]
            print(f"✅ Mode changed to: {self.current_mode.value}")
        else:
            print(f"❌ Unknown mode: {mode_str}")
            print("Available modes:", ", ".join(mode_map.keys()))
    
    async def process_specialized_query(self, query: str, specialty: str):
        """Process query with specialized agent assignment"""
        if specialty not in self.coding_specialists:
            print(f"❌ Unknown specialty: {specialty}")
            return
        
        spec = self.coding_specialists[specialty]
        
        if self.verbosity:
            print(f"🔍 [VERBOSE] Using {specialty} specialists:")
            print(f"    Primary: {spec['primary']}")
            print(f"    Secondary: {spec['secondary']}")
        
        print(f"🎯 Processing {specialty.upper()} request: {query}")
        print(f"🤖 Specialist: {spec['description']}")
        
        # Use code generation mode for specialized coding tasks
        response = await self.system.process_task(query, TaskType.CODE_GENERATION)
        
        self.print_response(response, query)
        self.add_to_history(f"{specialty} {query}", response)
    
    async def process_query(self, query: str):
        """Process a regular query"""
        if self.verbosity:
            print(f"🔍 [VERBOSE] Processing query in {self.current_mode.value} mode")
        
        print(f"🤔 Processing: {query}")
        
        response = await self.system.process_task(query, self.current_mode)
        
        self.print_response(response, query)
        self.add_to_history(query, response)
    
    def print_response(self, response, original_query: str):
        """Print formatted response"""
        print("\n" + "═" * 60)
        print("📝 RESPONSE")
        print("═" * 60)
        print(f"🤖 Agent: {response.agent_name} ({response.agent_type.value})")
        print(f"⚡ Confidence: {response.confidence:.2f}")
        print(f"⏱️  Processing Time: {response.processing_time:.3f}s")
        
        if hasattr(response, 'metadata') and response.metadata.get('multi_agent'):
            print(f"🔄 Multi-Agent Response (Used {response.metadata['agent_count']} agents)")
        
        print("\n📄 Content:")
        print("─" * 40)
        print(response.content)
        print("─" * 60)
    
    def add_to_history(self, command: str, response):
        """Add entry to command history"""
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'command': command,
            'agent': response.agent_name,
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'response_summary': response.content[:100] + "..." if len(response.content) > 100 else response.content
        }
        
        self.history.append(entry)
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def export_history(self):
        """Export history to JSON file"""
        try:
            filename = f"agentic_history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"✅ History exported to: {filename}")
        except Exception as e:
            print(f"❌ Error exporting history: {e}")
    
    def clear_history(self):
        """Clear command history"""
        self.history = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        print("✅ History cleared")
    
    def toggle_verbosity(self):
        """Toggle verbose output"""
        self.verbosity = not self.verbosity
        print(f"🔊 Verbosity: {'ON' if self.verbosity else 'OFF'}")
    
    async def run(self):
        """Main UI loop"""
        self.clear_screen()
        self.print_banner()
        
        print("🚀 Welcome! Type 'help' for detailed commands or 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("🎯 Command> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ['exit', 'quit']:
                    self.save_history()
                    self.save_config()
                    print("👋 Goodbye! Configuration and history saved.")
                    break
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'status':
                    await self.show_status()
                
                elif command == 'agents':
                    await self.show_agents()
                
                elif command == 'history':
                    limit = 10
                    if args and args.isdigit():
                        limit = int(args)
                    self.show_history(limit)
                
                elif command == 'clear':
                    self.clear_history()
                
                elif command == 'verbose':
                    self.toggle_verbosity()
                
                elif command == 'params':
                    self.show_available_parameters()
                
                elif command == 'save':
                    self.save_config()
                    self.save_history()
                    print("✅ Configuration and history saved")
                
                elif command == 'export':
                    self.export_history()
                
                elif command.startswith('set'):
                    if args:
                        self.set_parameter(args)
                    else:
                        print("❌ Usage: set <parameter>=<value>")
                
                elif command.startswith('get'):
                    if args:
                        self.get_parameter(args)
                    else:
                        print("❌ Usage: get <parameter>")
                
                elif command == 'mode':
                    if args:
                        self.change_mode(args)
                    else:
                        print(f"📍 Current mode: {self.current_mode.value}")
                        print("Available modes: general, code, creative, analysis, conversation, planning")
                
                elif command == 'query':
                    if args:
                        await self.process_query(args)
                    else:
                        print("❌ Usage: query <your question>")
                
                elif command == 'python':
                    if args:
                        await self.process_specialized_query(args, 'python')
                    else:
                        print("❌ Usage: python <your Python code request>")
                
                elif command == 'pine':
                    if args:
                        await self.process_specialized_query(args, 'pinescript')
                    else:
                        print("❌ Usage: pine <your PineScript request>")
                
                elif command == 'coding':
                    if args:
                        await self.process_specialized_query(args, 'general_code')
                    else:
                        print("❌ Usage: coding <your coding task>")
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("💡 Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Use 'quit' to exit properly.")
            except Exception as e:
                print(f"❌ Error: {e}")
                if self.verbosity:
                    import traceback
                    traceback.print_exc()

async def main():
    """Main entry point"""
    ui = AdvancedAgenticUI()
    await ui.run()

if __name__ == "__main__":
    asyncio.run(main())
