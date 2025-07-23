#!/usr/bin/env python3
"""
TERMINALIS-V.2 Model Orchestrator
Dynamic Model Discovery and Agent Orchestration System
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn
from rich.tree import Tree

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
    from huggingface_hub import scan_cache_dir, HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

console = Console()

@dataclass
class ModelInfo:
    """Model information container"""
    name: str
    path: str
    size_mb: float
    last_modified: datetime
    model_type: str
    agent_compatibility: List[str]
    capabilities: List[str]
    status: str = "cached"

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    required_model_types: List[str]
    preferred_models: List[str]
    task_types: List[str]

class ModelOrchestrator:
    """Dynamic Model Discovery and Agent Orchestration System"""
    
    # Agent capability definitions
    AGENT_CAPABILITIES = {
        "code_generator": AgentCapability(
            name="Code Generator",
            description="Generates and analyzes code in multiple languages",
            required_model_types=["code", "text-generation", "fill-mask"],
            preferred_models=["codegen", "codebert", "codet5", "starcoder", "incoder"],
            task_types=["code-generation", "code-completion", "code-review", "debugging"]
        ),
        "text_analyst": AgentCapability(
            name="Text Analyst",
            description="Analyzes and processes natural language text",
            required_model_types=["text-classification", "feature-extraction", "fill-mask"],
            preferred_models=["bert", "roberta", "distilbert", "electra", "deberta"],
            task_types=["text-classification", "sentiment-analysis", "ner", "feature-extraction"]
        ),
        "conversational": AgentCapability(
            name="Conversational AI",
            description="Engages in natural conversations and dialogue",
            required_model_types=["text-generation", "conversational"],
            preferred_models=["gpt", "dialogpt", "blenderbot", "chatglm", "llama"],
            task_types=["conversation", "question-answering", "dialogue"]
        ),
        "creative_writer": AgentCapability(
            name="Creative Writer",
            description="Creates creative content and stories",
            required_model_types=["text-generation"],
            preferred_models=["gpt", "bloom", "opt", "t5", "pegasus"],
            task_types=["text-generation", "creative-writing", "storytelling"]
        ),
        "summarizer": AgentCapability(
            name="Text Summarizer",
            description="Summarizes and condenses text content",
            required_model_types=["summarization", "text2text-generation"],
            preferred_models=["bart", "pegasus", "t5", "led", "prophetnet"],
            task_types=["summarization", "text2text-generation"]
        ),
        "translator": AgentCapability(
            name="Language Translator",
            description="Translates text between languages",
            required_model_types=["translation", "text2text-generation"],
            preferred_models=["marian", "m2m", "t5", "mbart", "nllb"],
            task_types=["translation", "text2text-generation"]
        ),
        "embedding_specialist": AgentCapability(
            name="Embedding Specialist",
            description="Creates semantic embeddings and similarity analysis",
            required_model_types=["feature-extraction", "sentence-similarity"],
            preferred_models=["sentence-transformers", "all-MiniLM", "mpnet", "instructor"],
            task_types=["feature-extraction", "sentence-similarity", "semantic-search"]
        )
    }
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
        self.discovered_models: Dict[str, ModelInfo] = {}
        self.agent_assignments: Dict[str, List[str]] = {}
        self.api = HfApi() if TRANSFORMERS_AVAILABLE else None
        
        console.print("🔍 [bold cyan]Model Orchestrator initialized[/bold cyan]")
        console.print(f"📁 Cache directory: {self.cache_dir}")
        
    def discover_cached_models(self) -> Dict[str, ModelInfo]:
        """Discover all cached models in the Hugging Face cache directory"""
        console.print("🔍 [yellow]Scanning Hugging Face cache for models...[/yellow]")
        
        if not self.cache_dir.exists():
            console.print(f"❌ [red]Cache directory not found: {self.cache_dir}[/red]")
            return {}
        
        discovered = {}
        
        try:
            if TRANSFORMERS_AVAILABLE:
                # Use huggingface_hub to scan cache
                cache_info = scan_cache_dir(self.cache_dir)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Discovering models...", total=len(cache_info.repos))
                    
                    for repo in cache_info.repos:
                        model_name = repo.repo_id
                        model_path = str(repo.repo_path)
                        
                        # Get model size
                        size_mb = sum(rev.size_on_disk for rev in repo.revisions) / (1024 * 1024)
                        
                        # Get last modified time
                        last_modified = datetime.fromtimestamp(
                            max(rev.last_modified for rev in repo.revisions)
                        )
                        
                        # Determine model type and capabilities
                        model_type, capabilities, agent_compatibility = self._analyze_model(model_name, model_path)
                        
                        model_info = ModelInfo(
                            name=model_name,
                            path=model_path,
                            size_mb=size_mb,
                            last_modified=last_modified,
                            model_type=model_type,
                            agent_compatibility=agent_compatibility,
                            capabilities=capabilities,
                            status="cached"
                        )
                        
                        discovered[model_name] = model_info
                        progress.advance(task)
                
            else:
                # Manual directory scanning fallback
                for model_dir in self.cache_dir.glob("models--*"):
                    if model_dir.is_dir():
                        model_name = model_dir.name.replace("models--", "").replace("--", "/")
                        size_mb = self._get_directory_size(model_dir) / (1024 * 1024)
                        last_modified = datetime.fromtimestamp(model_dir.stat().st_mtime)
                        
                        model_type, capabilities, agent_compatibility = self._analyze_model(model_name, str(model_dir))
                        
                        model_info = ModelInfo(
                            name=model_name,
                            path=str(model_dir),
                            size_mb=size_mb,
                            last_modified=last_modified,
                            model_type=model_type,
                            agent_compatibility=agent_compatibility,
                            capabilities=capabilities,
                            status="cached"
                        )
                        
                        discovered[model_name] = model_info
        
        except Exception as e:
            console.print(f"❌ [red]Error discovering models: {e}[/red]")
        
        self.discovered_models = discovered
        console.print(f"✅ [green]Discovered {len(discovered)} cached models[/green]")
        
        return discovered
    
    def _analyze_model(self, model_name: str, model_path: str) -> Tuple[str, List[str], List[str]]:
        """Analyze a model to determine its type, capabilities, and agent compatibility"""
        model_name_lower = model_name.lower()
        
        # Determine model type based on name patterns
        model_type = "unknown"
        capabilities = []
        agent_compatibility = []
        
        # Code models
        if any(keyword in model_name_lower for keyword in ["code", "codegen", "codebert", "codet5", "starcoder", "incoder"]):
            model_type = "code"
            capabilities = ["code-generation", "code-completion", "fill-mask"]
            agent_compatibility = ["code_generator"]
        
        # Language models for conversation
        elif any(keyword in model_name_lower for keyword in ["gpt", "llama", "opt", "bloom", "dialogpt", "chatglm"]):
            model_type = "text-generation"
            capabilities = ["text-generation", "conversation"]
            agent_compatibility = ["conversational", "creative_writer"]
        
        # BERT-like models
        elif any(keyword in model_name_lower for keyword in ["bert", "roberta", "electra", "deberta", "distil"]):
            model_type = "encoder"
            capabilities = ["text-classification", "feature-extraction", "fill-mask"]
            agent_compatibility = ["text_analyst", "embedding_specialist"]
        
        # Summarization models
        elif any(keyword in model_name_lower for keyword in ["bart", "pegasus", "led", "prophetnet"]):
            model_type = "summarization"
            capabilities = ["summarization", "text-generation"]
            agent_compatibility = ["summarizer", "creative_writer"]
        
        # Translation models
        elif any(keyword in model_name_lower for keyword in ["marian", "m2m", "mbart", "nllb"]):
            model_type = "translation"
            capabilities = ["translation", "text2text-generation"]
            agent_compatibility = ["translator"]
        
        # T5-like models
        elif any(keyword in model_name_lower for keyword in ["t5", "ul2", "flan"]):
            model_type = "text2text-generation"
            capabilities = ["text2text-generation", "summarization", "translation"]
            agent_compatibility = ["summarizer", "translator", "creative_writer"]
        
        # Sentence transformers
        elif any(keyword in model_name_lower for keyword in ["sentence-transformer", "all-minilm", "mpnet", "instructor"]):
            model_type = "sentence-similarity"
            capabilities = ["feature-extraction", "sentence-similarity"]
            agent_compatibility = ["embedding_specialist"]
        
        # Try to read config.json for more accurate classification
        try:
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                arch = config.get("architectures", [])
                if arch:
                    arch_name = arch[0].lower()
                    
                    if "classification" in arch_name:
                        capabilities.append("text-classification")
                        agent_compatibility.append("text_analyst")
                    elif "generation" in arch_name or "lm" in arch_name:
                        capabilities.append("text-generation")
                        agent_compatibility.append("conversational")
                    elif "embedding" in arch_name:
                        capabilities.append("feature-extraction")
                        agent_compatibility.append("embedding_specialist")
        except:
            pass
        
        # Remove duplicates
        capabilities = list(set(capabilities))
        agent_compatibility = list(set(agent_compatibility))
        
        return model_type, capabilities, agent_compatibility
    
    def orchestrate_agents(self) -> Dict[str, List[str]]:
        """Orchestrate agents based on available models"""
        console.print("🎭 [yellow]Orchestrating agents based on available models...[/yellow]")
        
        if not self.discovered_models:
            self.discover_cached_models()
        
        agent_assignments = {}
        
        for agent_name, capability in self.AGENT_CAPABILITIES.items():
            suitable_models = []
            
            for model_name, model_info in self.discovered_models.items():
                # Check if model is compatible with this agent
                if agent_name in model_info.agent_compatibility:
                    suitable_models.append(model_name)
                
                # Check if model has required capabilities
                elif any(cap in model_info.capabilities for cap in capability.required_model_types):
                    suitable_models.append(model_name)
                
                # Check if model matches preferred patterns
                elif any(pref in model_name.lower() for pref in capability.preferred_models):
                    suitable_models.append(model_name)
            
            # Sort by preference and size (smaller models first for efficiency)
            suitable_models.sort(key=lambda x: (
                -any(pref in x.lower() for pref in capability.preferred_models),
                self.discovered_models[x].size_mb
            ))
            
            agent_assignments[agent_name] = suitable_models
        
        self.agent_assignments = agent_assignments
        return agent_assignments
    
    def create_agent_model_pipeline(self, agent_name: str, task: str = None) -> Optional[Any]:
        """Create a pipeline for a specific agent using the best available model"""
        if agent_name not in self.agent_assignments:
            console.print(f"❌ [red]Agent {agent_name} not found[/red]")
            return None
        
        available_models = self.agent_assignments[agent_name]
        if not available_models:
            console.print(f"❌ [red]No suitable models found for agent {agent_name}[/red]")
            return None
        
        # Use the first (best) available model
        selected_model = available_models[0]
        model_info = self.discovered_models[selected_model]
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                console.print("❌ [red]Transformers library not available[/red]")
                return None
            
            # Determine the best task for this model if not specified
            if task is None:
                capability = self.AGENT_CAPABILITIES[agent_name]
                task = capability.task_types[0] if capability.task_types else "feature-extraction"
            
            # Create pipeline
            console.print(f"🔧 [yellow]Creating pipeline for {agent_name} using {selected_model}...[/yellow]")
            
            pipe = pipeline(
                task,
                model=selected_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            console.print(f"✅ [green]Pipeline created for {agent_name} with model {selected_model}[/green]")
            return {
                "pipeline": pipe,
                "model_name": selected_model,
                "model_info": model_info,
                "agent_name": agent_name,
                "task": task
            }
            
        except Exception as e:
            console.print(f"❌ [red]Failed to create pipeline for {agent_name}: {e}[/red]")
            return None
    
    def display_model_inventory(self) -> Table:
        """Display a comprehensive model inventory"""
        table = Table(title="🤖 Discovered Model Inventory")
        table.add_column("Model Name", style="cyan", width=30)
        table.add_column("Type", style="magenta")
        table.add_column("Size", style="yellow")
        table.add_column("Capabilities", style="green", width=25)
        table.add_column("Agent Compatibility", style="blue", width=20)
        table.add_column("Last Modified", style="white")
        
        for model_name, info in self.discovered_models.items():
            capabilities_str = ", ".join(info.capabilities[:2]) + ("..." if len(info.capabilities) > 2 else "")
            agents_str = ", ".join(info.agent_compatibility[:2]) + ("..." if len(info.agent_compatibility) > 2 else "")
            
            table.add_row(
                model_name[:28] + "..." if len(model_name) > 28 else model_name,
                info.model_type,
                f"{info.size_mb:.1f}MB",
                capabilities_str,
                agents_str,
                info.last_modified.strftime("%Y-%m-%d")
            )
        
        return table
    
    def display_agent_orchestration(self) -> Tree:
        """Display agent orchestration as a tree"""
        tree = Tree("🎭 Agent Orchestration")
        
        for agent_name, models in self.agent_assignments.items():
            capability = self.AGENT_CAPABILITIES[agent_name]
            agent_node = tree.add(f"[bold cyan]{capability.name}[/bold cyan] ({agent_name})")
            agent_node.add(f"[dim]{capability.description}[/dim]")
            
            if models:
                models_node = agent_node.add(f"[green]Available Models ({len(models)})[/green]")
                for i, model in enumerate(models[:5]):  # Show top 5 models
                    status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📦"
                    size = self.discovered_models[model].size_mb
                    models_node.add(f"{status} {model} ({size:.1f}MB)")
                
                if len(models) > 5:
                    models_node.add(f"[dim]... and {len(models) - 5} more[/dim]")
            else:
                agent_node.add("[red]❌ No suitable models found[/red]")
        
        return tree
    
    def get_available_cached_models(self) -> List[str]:
        """Get list of actually cached models from the directory"""
        # Real models from your cache directory based on the image
        actual_cached_models = [
            "shartford/WizardLM-7B-Uncensored",
            "facebook/musicgen-small", 
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "microsoft/phi-3-mini-4k-instruct",
            "microsoft/Phi-3-4",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-v0.1",
            "Open-Orca/Mistral-7B-OpenOrca",
            "Qwen/Qwen1.5-3B",
            "Salesforce/codegen25-7b-mono",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "TheBloke/CodeLlama-13B-Instruct-GPTQ",
            "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        ]
        
        return actual_cached_models
    
    def get_recommended_models_for_download(self, max_models: int = 5) -> List[str]:
        """Get recommended models to download for better agent coverage"""
        # Get currently cached models
        cached_models = self.get_available_cached_models()
        
        # Additional models that would complement existing ones
        recommended = [
            "distilbert-base-uncased",  # Fast text analysis
            "sentence-transformers/all-MiniLM-L6-v2",  # Embeddings
            "facebook/bart-large-cnn",  # Summarization
            "t5-small",  # Text-to-text generation
            "microsoft/codebert-base",  # Code analysis
        ]
        
        # Filter out already cached models
        available_recommended = []
        for model in recommended:
            if model not in cached_models:
                available_recommended.append(model)
        
        return available_recommended[:max_models]
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def export_model_info(self, output_path: str = "model_inventory.json"):
        """Export model information to JSON"""
        export_data = {
            "discovery_time": datetime.now().isoformat(),
            "cache_directory": str(self.cache_dir),
            "total_models": len(self.discovered_models),
            "models": {},
            "agent_assignments": self.agent_assignments
        }
        
        for name, info in self.discovered_models.items():
            export_data["models"][name] = {
                "path": info.path,
                "size_mb": info.size_mb,
                "last_modified": info.last_modified.isoformat(),
                "model_type": info.model_type,
                "capabilities": info.capabilities,
                "agent_compatibility": info.agent_compatibility,
                "status": info.status
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"📁 [green]Model inventory exported to {output_path}[/green]")

# Example usage and testing
if __name__ == "__main__":
    orchestrator = ModelOrchestrator()
    
    # Discover models
    models = orchestrator.discover_cached_models()
    
    # Display inventory
    console.print(orchestrator.display_model_inventory())
    
    # Orchestrate agents
    assignments = orchestrator.orchestrate_agents()
    
    # Display orchestration
    console.print(orchestrator.display_agent_orchestration())
    
    # Show recommended downloads
    recommended = orchestrator.get_recommended_models_for_download()
    if recommended:
        console.print("\n🎯 [yellow]Recommended models to download for better coverage:[/yellow]")
        for model in recommended:
            console.print(f"  • {model}")
    
    # Export inventory
    orchestrator.export_model_info()
