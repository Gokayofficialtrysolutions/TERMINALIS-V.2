# TERMINALIS-V.2 🤖
## The Most Powerful Agentic AI Coding Orchestrator

<div align="center">

![TERMINALIS-V.2 Logo](https://img.shields.io/badge/TERMINALIS-V.2-red?style=for-the-badge&logo=artificial-intelligence)
![Power Level](https://img.shields.io/badge/Power_Level-MAXIMUM-ff0000?style=for-the-badge)
![AI Agents](https://img.shields.io/badge/AI_Agents-20+-blue?style=for-the-badge)

**🚀 Ultra-Powerful Multi-Agent AI Coding System**
**⚡ Maximum Performance • 🧠 Advanced Orchestration • 🔥 No Limits**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Safetensors](https://img.shields.io/badge/Safetensors-0.3+-green.svg)](https://huggingface.co/docs/safetensors)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Gokayofficialtrysolutions/TERMINALIS-V.2?style=social)](https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2/stargazers)

</div>

## 🚀 One-Line Installation

```powershell
iwr -useb https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/install.ps1 | iex
```

*Run this command in PowerShell as Administrator and watch the magic happen!*

A sophisticated multi-agent AI system with advanced model management, real-time progress tracking, and safetensors support.

## 📋 Features

### 🎯 Multi-Agent Architecture
- **4 Specialized Agents**: Coding, Reasoning, Creative, and General purpose agents
- **Smart Agent Selection**: Automatically selects the best agent based on task type and content
- **Multi-Agent Processing**: Option to use multiple agents for complex tasks

### 🐍 Coding Specializations
- **Python Development**: CodeGen 2.5 7B specialized for Python code generation
- **PineScript Trading**: Specialized agent for TradingView PineScript strategies
- **General Programming**: Support for JavaScript, HTML, and other languages

### 🎨 Advanced UI Features
- **Parameter Control**: Adjust temperature, tokens, confidence thresholds
- **Command History**: Persistent history with export capabilities
- **Verbosity Control**: Toggle detailed processing information
- **Session Management**: Save and load configurations

## 🚀 Quick Start

### 1. Run the Advanced UI (Recommended)
```bash
python advanced_ui.py
```

### 2. Run the Simple ASCII Interface
```bash
python ascii_interface.py
```

### 3. Test the Core System
```bash
python agentic_ai_system.py
```

## 💡 Usage Examples

### Basic Commands
```
🎯 Command> query "Explain machine learning"
🎯 Command> python "Create a fibonacci function"
🎯 Command> pine "Write a moving average strategy"
🎯 Command> status
🎯 Command> agents
```

### Parameter Management
```
🎯 Command> set temperature=0.8
🎯 Command> set max_tokens=4096
🎯 Command> params
🎯 Command> get temperature
```

### Mode Changes
```
🎯 Command> mode creative
🎯 Command> mode analysis
🎯 Command> mode code
```

## 🔧 Available Agents

### 1. **Coder-CodeGen25** (Coding Agent)
- **Type**: Primary coding agent
- **Specialties**: Python, JavaScript, PineScript, General coding
- **Best For**: Code generation, debugging, programming tasks

### 2. **Reasoner-Qwen3** (Reasoning Agent)
- **Type**: Analysis and logic agent
- **Specialties**: Analysis, logic, problem-solving, mathematics
- **Best For**: Complex reasoning, data analysis, mathematical problems

### 3. **Creative-OpenHermes** (Creative Agent)
- **Type**: Creative and storytelling agent
- **Specialties**: Creative writing, storytelling, brainstorming, marketing
- **Best For**: Content creation, creative tasks, brainstorming

### 4. **General-Assistant** (General Agent)
- **Type**: General purpose assistant
- **Specialties**: Conversation, general queries, information, support
- **Best For**: General questions, conversation, information lookup

## 📊 Command Reference

### 📋 Query Commands
| Command | Description | Example |
|---------|-------------|---------|
| `query <text>` | Process query with current mode | `query "What is AI?"` |
| `python <code>` | Python code request | `python "Create a web scraper"` |
| `pine <script>` | PineScript request | `pine "RSI strategy"` |
| `coding <task>` | General coding task | `coding "HTML contact form"` |

### 🎯 Mode Commands
| Command | Description |
|---------|-------------|
| `mode general` | General queries |
| `mode code` | Code generation |
| `mode creative` | Creative writing |
| `mode analysis` | Deep analysis |
| `mode conversation` | Conversation mode |
| `mode planning` | Planning tasks |

### 🔧 Parameter Commands
| Command | Description | Range |
|---------|-------------|-------|
| `set temperature=X` | Model creativity | 0.1 - 2.0 |
| `set max_tokens=X` | Response length | 1 - 8192 |
| `set confidence_threshold=X` | Min confidence | 0.0 - 1.0 |
| `get <param>` | Show parameter value | - |
| `params` | Show all parameters | - |

### 📊 System Commands
| Command | Description |
|---------|-------------|
| `status` | Detailed system status |
| `agents` | Show all agents and specializations |
| `history [N]` | Show recent commands (default 10) |
| `clear` | Clear command history |
| `verbose` | Toggle verbose output |

### 💾 Session Commands
| Command | Description |
|---------|-------------|
| `save` | Save current configuration |
| `export` | Export history to JSON |
| `load` | Load previous session |

## 🎨 Specialized Coding Features

### Python Development
- Automatic selection of CodeGen 2.5 7B for Python tasks
- Secondary reasoning agent (Qwen3) for complex logic
- Optimized for web scraping, data analysis, automation

### PineScript Trading
- Specialized for TradingView strategies and indicators
- Creative agent integration for innovative trading ideas
- Support for Pine Script v5 syntax

### General Programming
- Multi-language support (JavaScript, HTML, CSS, etc.)
- Code review and optimization suggestions
- Best practices and documentation generation

## 🔥 Advanced Features

### Multi-Agent Processing
Enable multi-agent processing for complex tasks:
```python
response = await system.process_task(task, task_type, multi_agent=True)
```

### Custom Agent Configuration
Agents can be configured with custom parameters:
- Temperature control for creativity
- Token limits for response length
- Confidence thresholds for quality control

### Persistent History
- Automatic saving of command history
- Export capabilities for session analysis
- Configuration persistence across sessions

## 📁 Project Structure

```
Agentic AI System/
├── agentic_ai_system.py    # Core multi-agent system
├── advanced_ui.py          # Advanced UI with full features
├── ascii_interface.py      # Simple ASCII interface
├── README.md              # This documentation
├── models/                # Model directory (auto-created)
├── agentic_history.pkl    # Command history (auto-created)
└── agentic_config.json    # Configuration (auto-created)
```

## 🛠️ System Requirements

- Python 3.7+
- asyncio support
- Windows/Linux/macOS compatible
- No external dependencies required (uses mock agents)

## 🔮 Future Enhancements

- **Real Model Integration**: Connect to actual AI models (Ollama, OpenAI, etc.)
- **Web Interface**: Browser-based UI for easier access
- **Plugin System**: Extensible agent architecture
- **Cloud Integration**: Deploy agents in cloud environments
- **Performance Metrics**: Detailed agent performance tracking

## 🤝 Contributing

This is a mock system designed for demonstration. To extend:

1. Replace `MockAgent` with real model implementations
2. Add new agent types in `AgentType` enum
3. Extend task types in `TaskType` enum
4. Implement new UI features in the interface files

## 📝 License

This project is provided as-is for educational and demonstration purposes.

---

**Ready to explore the future of AI agents?** 🚀

Start with: `python advanced_ui.py` and type `help` for a complete command reference!
