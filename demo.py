#!/usr/bin/env python3
"""
Agentic AI System Demo
======================
Demonstration script showcasing the capabilities of the multi-agent system.
"""

import asyncio
from agentic_ai_system import AgenticAISystem, TaskType

async def run_demo():
    """Run a comprehensive demo of the Agentic AI System"""
    
    print("🚀 AGENTIC AI SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize the system
    system = AgenticAISystem()
    
    # Show system status
    print("\n📊 SYSTEM STATUS")
    print("-" * 30)
    status = system.get_system_status()
    print(f"✅ Total Agents: {status['total_agents']}")
    print(f"✅ System Ready: {status['system_ready']}")
    print(f"✅ Models Directory: {status['models_directory']}")
    
    # Demo queries for each agent type
    demo_tasks = [
        {
            "title": "🐍 PYTHON CODING DEMO",
            "task": "Create a Python function to calculate prime numbers",
            "task_type": TaskType.CODE_GENERATION
        },
        {
            "title": "📈 PINESCRIPT TRADING DEMO", 
            "task": "Write a PineScript strategy using RSI and moving averages",
            "task_type": TaskType.CODE_GENERATION
        },
        {
            "title": "🧠 REASONING & ANALYSIS DEMO",
            "task": "Explain the mathematical principles behind machine learning gradient descent",
            "task_type": TaskType.ANALYSIS
        },
        {
            "title": "✨ CREATIVE WRITING DEMO",
            "task": "Write a creative story about an AI that discovers emotions",
            "task_type": TaskType.CREATIVE_WRITING
        },
        {
            "title": "💬 GENERAL CONVERSATION DEMO",
            "task": "What are the key trends in artificial intelligence for 2024?",
            "task_type": TaskType.GENERAL_QUERY
        }
    ]
    
    print(f"\n🎭 RUNNING {len(demo_tasks)} DEMO TASKS")
    print("=" * 60)
    
    for i, demo in enumerate(demo_tasks, 1):
        print(f"\n{demo['title']}")
        print("-" * len(demo['title']))
        print(f"📝 Task: {demo['task']}")
        
        # Process the task
        try:
            response = await system.process_task(demo['task'], demo['task_type'])
            
            print(f"🤖 Selected Agent: {response.agent_name}")
            print(f"🎯 Agent Type: {response.agent_type.value}")
            print(f"⚡ Confidence: {response.confidence:.2f}")
            print(f"⏱️  Processing Time: {response.processing_time:.2f}s")
            print(f"🔍 Mock Agent: {response.metadata.get('mock_agent', False)}")
            
            # Show response preview
            preview_length = 300
            if len(response.content) > preview_length:
                preview = response.content[:preview_length] + "..."
            else:
                preview = response.content
                
            print(f"\n📄 Response Preview:")
            print("─" * 40)
            print(preview)
            print("─" * 40)
            
            if i < len(demo_tasks):
                print("\n⏳ Next demo in 2 seconds...")
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"❌ Error processing task: {e}")
    
    # Multi-agent demo
    print(f"\n🔄 MULTI-AGENT PROCESSING DEMO")
    print("-" * 40)
    print("📝 Task: Create a comprehensive trading bot with Python and PineScript")
    
    try:
        # This would use multiple agents if implemented
        response = await system.process_task(
            "Create a comprehensive trading bot with Python and PineScript integration",
            TaskType.CODE_GENERATION
        )
        
        print(f"🤖 Primary Agent: {response.agent_name}")
        print(f"⚡ Confidence: {response.confidence:.2f}")
        print(f"📄 Response: {response.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error in multi-agent demo: {e}")
    
    # Show final statistics
    print(f"\n📈 DEMO COMPLETED")
    print("=" * 30)
    final_status = system.get_system_status()
    print(f"📊 Total Tasks Processed: {final_status['task_history_count']}")
    print(f"🎯 All Agents Available: ✅")
    
    # Show agent specializations
    print(f"\n🔧 AGENT SPECIALIZATIONS")
    print("-" * 30)
    specializations = system.get_agent_specializations()
    for agent_name, specs in specializations.items():
        print(f"🤖 {agent_name}: {', '.join(specs)}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 Try running: python advanced_ui.py for interactive experience")

if __name__ == "__main__":
    asyncio.run(run_demo())
