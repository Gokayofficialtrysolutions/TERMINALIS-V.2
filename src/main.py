#!/usr/bin/env python3
"""
TERMINALIS-V.2 Agentic AI System
Main Application Entry Point
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_manager import AgentManager
from tools.tool_manager import ToolManager
from src.model_orchestrator import ModelOrchestrator

class TerminalisAI:
    def __init__(self, config_path=None):
        self.version = "2.0.0"
        self.start_time = datetime.now()
        
        # Setup paths
        self.base_path = Path(__file__).parent.parent
        self.config_path = config_path or self.base_path / "config.yaml"
        self.logs_path = self.base_path / "logs"
        self.models_path = self.base_path / "models"
        self.data_path = self.base_path / "data"
        
        # Create directories if they don't exist
        for path in [self.logs_path, self.models_path, self.data_path]:
            path.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize managers
        self.agent_manager = AgentManager(self.config)
        self.tool_manager = ToolManager(self.config)
        self.model_orchestrator = ModelOrchestrator()
        
        # Discover cached models and orchestrate agents
        self.discover_and_orchestrate_models()
        
        self.logger.info(f"TERMINALIS-V.2 v{self.version} initialized successfully")
    
    def discover_and_orchestrate_models(self):
        """Discover cached models and orchestrate agents"""
        try:
            # Discover models from Hugging Face cache
            self.discovered_models = self.model_orchestrator.discover_cached_models()
            
            # Orchestrate agents based on available models
            self.agent_assignments = self.model_orchestrator.orchestrate_agents()
            
            self.logger.info(f"Discovered {len(self.discovered_models)} cached models")
            self.logger.info(f"Orchestrated {len(self.agent_assignments)} agent types")
            
        except Exception as e:
            self.logger.error(f"Error in model discovery: {e}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.logs_path / f"terminalis_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("TERMINALIS")
    
    def load_config(self):
        """Load system configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self.get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "system": {
                "name": "TERMINALIS-V.2",
                "version": self.version,
                "max_agents": 10,
                "max_tools": 50
            },
            "ai": {
                "default_model": "bert-base-uncased",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "paths": {
                "models": str(self.models_path),
                "data": str(self.data_path),
                "logs": str(self.logs_path)
            }
        }
    
    def display_banner(self):
        """Display system banner"""
        banner = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗     ██╗███████╗ ║
║  ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██║██╔════╝ ║
║     ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ██║███████╗ ║
║     ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██║╚════██║ ║
║     ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║███████║ ║
║     ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝ ║
║                                                                               ║
║                            🤖 AGENTIC AI SYSTEM V.2 🤖                        ║
║                         Advanced Terminal Intelligence                         ║
║                                                                               ║
║  Version: {self.version:<20} Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S'):<20} ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def display_menu(self):
        """Display main menu"""
        menu = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN MENU                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. 🤖 Agent Management                                                     │
│  2. 🔧 Tool Management                                                      │
│  3. 📊 System Status                                                       │
│  4. 📁 File Operations                                                     │
│  5. 🧠 AI Model Management                                                 │
│  6. ⚙️  Configuration                                                       │
│  7. 📋 View Logs                                                           │
│  8. ❓ Help                                                                │
│  9. 🚪 Exit                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
        """
        print(menu)
    
    def run_interactive_mode(self):
        """Run in interactive mode"""
        self.display_banner()
        
        while True:
            try:
                self.display_menu()
                choice = input("\n🎯 Enter your choice (1-9): ").strip()
                
                if choice == '1':
                    self.agent_management_menu()
                elif choice == '2':
                    self.tool_management_menu()
                elif choice == '3':
                    self.show_system_status()
                elif choice == '4':
                    self.file_operations_menu()
                elif choice == '5':
                    self.ai_model_menu()
                elif choice == '6':
                    self.configuration_menu()
                elif choice == '7':
                    self.view_logs()
                elif choice == '8':
                    self.show_help()
                elif choice == '9':
                    print("\n👋 Thank you for using TERMINALIS-V.2!")
                    break
                else:
                    print("\n❌ Invalid choice. Please select 1-9.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ An error occurred: {e}")
    
    def agent_management_menu(self):
        """Agent management submenu"""
        print("\n🤖 AGENT MANAGEMENT")
        print("=" * 50)
        agents = self.agent_manager.list_agents()
        if agents:
            for i, agent in enumerate(agents, 1):
                print(f"{i}. {agent['name']} - Status: {agent['status']}")
        else:
            print("No agents currently active.")
    
    def tool_management_menu(self):
        """Tool management submenu"""
        print("\n🔧 TOOL MANAGEMENT")
        print("=" * 50)
        tools = self.tool_manager.list_tools()
        if tools:
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool['name']} - Type: {tool['type']}")
        else:
            print("No tools currently loaded.")
    
    def show_system_status(self):
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 50)
        print(f"Version: {self.version}")
        print(f"Uptime: {datetime.now() - self.start_time}")
        print(f"Active Agents: {len(self.agent_manager.list_agents())}")
        print(f"Loaded Tools: {len(self.tool_manager.list_tools())}")
        print(f"Base Path: {self.base_path}")
    
    def file_operations_menu(self):
        """File operations menu"""
        print("\n📁 FILE OPERATIONS")
        print("=" * 50)
        print("1. List files in data directory")
        print("2. List models")
        print("3. View configuration")
        
        choice = input("Select option (1-3): ").strip()
        if choice == '1':
            files = list(self.data_path.glob("*"))
            for file in files:
                print(f"  📄 {file.name}")
        elif choice == '2':
            models = list(self.models_path.glob("*"))
            for model in models:
                print(f"  🧠 {model.name}")
        elif choice == '3':
            print(json.dumps(self.config, indent=2))
    
    def ai_model_menu(self):
        """AI model management menu"""
        print("\n🧠 AI MODEL MANAGEMENT")
        print("=" * 50)
        print("Available models:")
        models = list(self.models_path.glob("*"))
        for model in models:
            print(f"  🤖 {model.name}")
    
    def configuration_menu(self):
        """Configuration menu"""
        print("\n⚙️ CONFIGURATION")
        print("=" * 50)
        print("Current configuration:")
        print(json.dumps(self.config, indent=2))
    
    def view_logs(self):
        """View recent logs"""
        print("\n📋 RECENT LOGS")
        print("=" * 50)
        log_files = list(self.logs_path.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:  # Show last 10 lines
                    print(line.strip())
        else:
            print("No log files found.")
    
    def show_help(self):
        """Show help information"""
        help_text = """
❓ HELP - TERMINALIS-V.2 Commands

🤖 AGENT MANAGEMENT:
   - Create and manage AI agents
   - Monitor agent status and performance
   - Configure agent behaviors

🔧 TOOL MANAGEMENT:
   - Load and manage system tools
   - Configure tool parameters
   - Monitor tool usage

📊 SYSTEM STATUS:
   - View system health and performance
   - Monitor resource usage
   - Check component status

For more information, visit:
📖 https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2
        """
        print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TERMINALIS-V.2 Agentic AI System")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--version", "-v", action="version", version="TERMINALIS-V.2 2.0.0")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run in daemon mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the system
        system = TerminalisAI(config_path=args.config)
        
        if args.daemon:
            print("🚀 Starting TERMINALIS-V.2 in daemon mode...")
            # Add daemon mode logic here
        else:
            # Run in interactive mode
            system.run_interactive_mode()
            
    except Exception as e:
        print(f"❌ Failed to start TERMINALIS-V.2: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
