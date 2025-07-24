#!/usr/bin/env python3
"""
PineScript XXXXLARGE Development Environment
Main Entry Point

Author: Gokaytrysolutions Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.engine import PineScriptEngine
from ai.assistant import AIAssistant
from tools.ide import PineScriptIDE
from utils.logger import setup_logger
from utils.config import ConfigManager
from ui.cli import CommandLineInterface
from ui.web import WebInterface

class PineScriptEnvironment:
    """Main PineScript Development Environment Controller"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.config = ConfigManager()
        self.engine = None
        self.ai_assistant = None
        self.ide = None
        self.web_interface = None
        self.cli = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("🚀 Initializing PineScript XXXXLARGE Environment...")
            
            # Initialize core engine
            self.engine = PineScriptEngine(self.config)
            await self.engine.initialize()
            
            # Initialize AI assistant
            self.ai_assistant = AIAssistant(self.config, self.engine)
            await self.ai_assistant.initialize()
            
            # Initialize IDE
            self.ide = PineScriptIDE(self.config, self.engine, self.ai_assistant)
            
            # Initialize interfaces
            self.cli = CommandLineInterface(self.config, self.engine, self.ai_assistant, self.ide)
            self.web_interface = WebInterface(self.config, self.engine, self.ai_assistant, self.ide)
            
            self.logger.info("✅ Environment initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize environment: {e}")
            raise
    
    async def start_cli_mode(self):
        """Start command-line interface mode"""
        self.logger.info("🖥️  Starting CLI mode...")
        await self.cli.start()
    
    async def start_web_mode(self, host: str = "localhost", port: int = 8080):
        """Start web interface mode"""
        self.logger.info(f"🌐 Starting web interface at http://{host}:{port}")
        await self.web_interface.start(host, port)
    
    async def start_ide_mode(self):
        """Start integrated development environment"""
        self.logger.info("🎨 Starting IDE mode...")
        await self.ide.start()
    
    def display_banner(self):
        """Display startup banner"""
        banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    PineScript XXXXLARGE Development Environment             ║
║                              by Gokaytrysolutions                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║ 🎯 Complete PineScript Development Suite                                   ║
║ 🤖 AI-Powered Code Generation & Optimization                              ║
║ 📊 Advanced Backtesting & Analytics                                       ║
║ 🔧 Professional Trading Strategy Development                               ║
╚════════════════════════════════════════════════════════════════════════════╝

🌟 Features Available:
   ✅ PineScript v5 Full Support
   ✅ AI Code Generation & Completion
   ✅ Real-time Strategy Optimization
   ✅ Advanced Backtesting Engine
   ✅ Multi-timeframe Analysis
   ✅ Risk Management Tools
   ✅ Portfolio Management
   ✅ Market Pattern Recognition
   ✅ Automated Bug Detection & Fixing
   ✅ Strategy Performance Analytics

🚀 Ready to revolutionize your trading strategies!
"""
        print(banner)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PineScript XXXXLARGE Development Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['cli', 'web', 'ide'], 
        default='cli',
        help='Interface mode (default: cli)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host for web mode (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for web mode (default: 8080)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PineScript XXXXLARGE Environment v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and initialize environment
        env = PineScriptEnvironment()
        
        # Display banner
        env.display_banner()
        
        # Initialize environment
        await env.initialize()
        
        # Start appropriate mode
        if args.mode == 'cli':
            await env.start_cli_mode()
        elif args.mode == 'web':
            await env.start_web_mode(args.host, args.port)
        elif args.mode == 'ide':
            await env.start_ide_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using PineScript XXXXLARGE Environment!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up event loop policy for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main application
    asyncio.run(main())
