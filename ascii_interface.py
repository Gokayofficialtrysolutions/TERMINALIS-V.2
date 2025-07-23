#!/usr/bin/env python3
"""
Graphical Agentic AI Interface (ASCII UI)
========================================
An interactive ASCII-based interface for operating and fine-tuning the Agentic AI System.
Provides editing capabilities for agent parameters and hyperparameters.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from agentic_ai_system import AgenticAISystem, TaskType

class ASCIIInterface:
    """ASCII-based user interface for interactive control and fine-tuning"""
    
    def __init__(self, models_dir: str = "models"):
        self.system = AgenticAISystem(models_dir)
        self.history = []
        self.verbosity = False
        
    async def run(self):
        """Start the interactive ASCII UI"""
        self.clear_screen()
        print(self.banner())
        
        while True:
            user_input = input("\n> Command: ").strip().lower()
            
            if user_input == 'exit':
                break
            elif user_input == 'status':
                await self.show_status()
            elif user_input == 'history':
                self.show_history()
            elif user_input.startswith('set '):
                self.set_parameter(user_input)
            elif user_input.startswith('query '):
                await self.process_query(user_input[6:])
            elif user_input == 'verbose':
                self.toggle_verbosity()
            else:
                print("Unknown command. Type 'help' for assistance.")

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def banner(self) -> str:
        return dedent("""
            ********************************************
            **    Graphical Agentic AI System UI     **
            ********************************************
            Commands Available:
              - status       : Show system status
              - history      : View command history
              - set <param>  : Set parameter value (e.g. set verbosity=true)
              - query <text> : Process a query
              - verbose      : Toggle verbosity
              - exit         : Exit the UI
            ********************************************
        """)

    async def show_status(self):
        status = self.system.get_system_status()
        if self.verbosity:
            print("\n[VERBOSE] Fetching system status...")
        if status:
            print("\n=== AGENTIC AI SYSTEM STATUS ===")
            print(f"Total Agents: {status['total_agents']}")
            print(f"Orchestrator Available: {status['orchestrator_available']}")
            for agent in status['available_agents']:
                print(f"- {agent['name']} [{agent['type']}] - Enabled: {agent['available']}")
        else:
            print("System status unavailable.")

    def show_history(self):
        if self.verbosity:
            print("\n[VERBOSE] Displaying history...")
        if not self.history:
            print("No history found.")
        else:
            print("\n=== COMMAND HISTORY ===")
            for entry in self.history:
                print(entry)

    def set_parameter(self, command: str):
        try:
            _, param = command.split(' ', 1)
            key, value = param.split('=')
            key = key.strip()
            value = value.strip()
            if key == "verbosity":
                self.verbosity = value.lower() == 'true'
                print(f"Verbosity set to {self.verbosity}")
            else:
                print(f"Parameter '{key}' not recognized.")
        except Exception as e:
            print(f"Error setting parameter: {e}")

    async def process_query(self, query: str):
        print(f"\nProcessing query: '{query}'")
        task_type = TaskType.GENERAL_QUERY
        response = await self.system.process_task(query, task_type)
        print("\n=== QUERY RESULT ===")
        print(f"Agent: {response.agent_name}")
        print(f"Type: {response.agent_type}")
        print(f"Confidence: {response.confidence}")
        print(f"Response: {response.content}")
        self.history.append((datetime.now(), query))

    def toggle_verbosity(self):
        self.verbosity = not self.verbosity
        print(f"Verbosity {'enabled' if self.verbosity else 'disabled'}")

async def main():
    interface = ASCIIInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main())

