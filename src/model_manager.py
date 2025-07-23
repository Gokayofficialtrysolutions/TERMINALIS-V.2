#!/usr/bin/env python3
"""
TERMINALIS-V.2 Model Manager
Advanced AI Model Management with Safetensors Support
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from safetensors import safe_open
from safetensors.torch import save_file, load_file

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig, 
        AutoModelForCausalLM, AutoModelForSequenceClassification,
        pipeline, PreTrainedModel, PreTrainedTokenizer
    )
    from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

console = Console()

class ModelManager:
    """Advanced Model Manager with Safetensors support"""
    
    SUPPORTED_MODELS = {
        # Language Models
        "bert-base-uncased": {
            "type": "encoder",
            "size": "440MB",
            "description": "BERT Base model for general NLP tasks",
            "use_cases": ["text-classification", "feature-extraction", "fill-mask"]
        },
        "gpt2-medium": {
            "type": "decoder",
            "size": "1.4GB",
            "description": "GPT-2 Medium for text generation",
            "use_cases": ["text-generation", "conversation"]
        },
        "distilbert-base-uncased": {
            "type": "encoder",
            "size": "255MB",
            "description": "Distilled BERT for faster inference",
            "use_cases": ["text-classification", "feature-extraction"]
        },
        "microsoft/DialoGPT-medium": {
            "type": "decoder",
            "size": "774MB",
            "description": "Conversational AI model",
            "use_cases": ["conversation", "chatbot"]
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "type": "encoder",
            "size": "80MB",
            "description": "Lightweight sentence embeddings",
            "use_cases": ["sentence-similarity", "semantic-search"]
        },
        "microsoft/codebert-base": {
            "type": "encoder",
            "size": "500MB",
            "description": "BERT for code understanding",
            "use_cases": ["code-search", "code-classification"]
        },
        "facebook/bart-large-cnn": {
            "type": "encoder-decoder",
            "size": "1.6GB",
            "description": "BART for summarization",
            "use_cases": ["summarization", "text-generation"]
        },
        "t5-small": {
            "type": "encoder-decoder",
            "size": "242MB",
            "description": "T5 Small for text-to-text generation",
            "use_cases": ["text2text-generation", "translation"]
        }
    }
    
    def __init__(self, models_path: str = "models", config: Dict = None):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        self.config = config or {}
        self.loaded_models: Dict[str, Dict] = {}
        self.download_progress = {}
        
        # Create subdirectories
        (self.models_path / "cache").mkdir(exist_ok=True)
        (self.models_path / "safetensors").mkdir(exist_ok=True)
        (self.models_path / "metadata").mkdir(exist_ok=True)
        
        console.print("🤖 [bold cyan]Model Manager initialized[/bold cyan]")
        
    def list_available_models(self) -> Table:
        """Display available models in a rich table"""
        table = Table(title="🤖 Available AI Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Size", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        
        for model_name, info in self.SUPPORTED_MODELS.items():
            status = "✅ Downloaded" if self.is_model_downloaded(model_name) else "⬇️ Available"
            table.add_row(
                model_name, 
                info["type"], 
                info["size"], 
                info["description"][:50] + "..." if len(info["description"]) > 50 else info["description"],
                status
            )
        
        return table
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        model_path = self.models_path / model_name.replace("/", "_")
        return model_path.exists() and (model_path / "model.safetensors").exists()
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Download a single model with progress tracking"""
        if not TRANSFORMERS_AVAILABLE:
            console.print("❌ [red]Transformers library not available[/red]")
            return False
            
        if model_name not in self.SUPPORTED_MODELS:
            console.print(f"❌ [red]Model {model_name} not supported[/red]")
            return False
        
        model_info = self.SUPPORTED_MODELS[model_name]
        safe_model_name = model_name.replace("/", "_")
        model_path = self.models_path / safe_model_name
        
        if self.is_model_downloaded(model_name) and not force_download:
            console.print(f"✅ [green]Model {model_name} already downloaded[/green]")
            return True
        
        model_path.mkdir(exist_ok=True)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                # Download tokenizer
                tokenizer_task = progress.add_task(f"Downloading tokenizer for {model_name}...", total=100)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.models_path / "cache"))
                tokenizer.save_pretrained(model_path)
                progress.update(tokenizer_task, completed=100)
                
                # Download config
                config_task = progress.add_task(f"Downloading config for {model_name}...", total=100)
                config = AutoConfig.from_pretrained(model_name, cache_dir=str(self.models_path / "cache"))
                config.save_pretrained(model_path)
                progress.update(config_task, completed=100)
                
                # Download model
                model_task = progress.add_task(f"Downloading model {model_name}...", total=100)
                
                # Determine model class based on type
                if model_info["type"] == "decoder":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        cache_dir=str(self.models_path / "cache"),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                else:
                    model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=str(self.models_path / "cache"),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                
                progress.update(model_task, completed=50)
                
                # Save using safetensors
                safetensors_task = progress.add_task(f"Converting to Safetensors...", total=100)
                self._save_model_safetensors(model, model_path / "model.safetensors")
                progress.update(safetensors_task, completed=100)
                progress.update(model_task, completed=100)
                
                # Save metadata
                metadata = {
                    "model_name": model_name,
                    "model_info": model_info,
                    "download_time": time.time(),
                    "safetensors_format": True,
                    "model_size_mb": self._get_model_size(model_path),
                    "torch_dtype": str(model.dtype) if hasattr(model, 'dtype') else "unknown"
                }
                
                with open(model_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
            console.print(f"✅ [green]Successfully downloaded {model_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"❌ [red]Failed to download {model_name}: {str(e)}[/red]")
            return False
    
    def download_multiple_models(self, model_names: List[str], max_workers: int = 2) -> Dict[str, bool]:
        """Download multiple models concurrently"""
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            overall_task = progress.add_task("Overall Progress", total=len(model_names))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self.download_model, model_name): model_name 
                    for model_name in model_names
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        results[model_name] = future.result()
                        progress.advance(overall_task)
                    except Exception as e:
                        console.print(f"❌ [red]Error downloading {model_name}: {e}[/red]")
                        results[model_name] = False
                        progress.advance(overall_task)
        
        return results
    
    def load_model(self, model_name: str, device: str = "auto") -> Optional[Dict]:
        """Load a model from safetensors format"""
        if not self.is_model_downloaded(model_name):
            console.print(f"❌ [red]Model {model_name} not found. Download it first.[/red]")
            return None
        
        safe_model_name = model_name.replace("/", "_")
        model_path = self.models_path / safe_model_name
        
        try:
            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load config
            config = AutoConfig.from_pretrained(model_path)
            
            # Load model from safetensors
            model_weights = load_file(model_path / "model.safetensors")
            
            # Initialize model architecture
            model_info = self.SUPPORTED_MODELS[model_name]
            if model_info["type"] == "decoder":
                model = AutoModelForCausalLM.from_config(config)
            else:
                model = AutoModel.from_config(config)
            
            # Load weights
            model.load_state_dict(model_weights)
            model.to(device)
            
            # Load metadata
            metadata_path = model_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "metadata": metadata,
                "device": device,
                "loaded_time": time.time()
            }
            
            self.loaded_models[model_name] = model_data
            console.print(f"✅ [green]Model {model_name} loaded successfully on {device}[/green]")
            
            return model_data
            
        except Exception as e:
            console.print(f"❌ [red]Failed to load {model_name}: {str(e)}[/red]")
            return None
    
    def create_pipeline(self, model_name: str, task: str = None) -> Optional[Any]:
        """Create a pipeline for a loaded model"""
        if model_name not in self.loaded_models:
            console.print(f"❌ [red]Model {model_name} not loaded[/red]")
            return None
        
        model_data = self.loaded_models[model_name]
        model_info = self.SUPPORTED_MODELS[model_name]
        
        # Auto-detect task if not provided
        if task is None:
            task = model_info["use_cases"][0] if model_info["use_cases"] else "feature-extraction"
        
        try:
            pipe = pipeline(
                task,
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                device=0 if model_data["device"] == "cuda" else -1
            )
            
            console.print(f"✅ [green]Pipeline created for {model_name} with task: {task}[/green]")
            return pipe
            
        except Exception as e:
            console.print(f"❌ [red]Failed to create pipeline: {str(e)}[/red]")
            return None
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model"""
        info = {
            "available": model_name in self.SUPPORTED_MODELS,
            "downloaded": self.is_model_downloaded(model_name),
            "loaded": model_name in self.loaded_models,
            "details": self.SUPPORTED_MODELS.get(model_name, {})
        }
        
        if info["downloaded"]:
            safe_model_name = model_name.replace("/", "_")
            model_path = self.models_path / safe_model_name
            metadata_path = model_path / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info["metadata"] = json.load(f)
        
        return info
    
    def cleanup_cache(self):
        """Clean up downloaded cache files"""
        cache_path = self.models_path / "cache"
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            cache_path.mkdir(exist_ok=True)
            console.print("🧹 [yellow]Cache cleaned up[/yellow]")
    
    def _save_model_safetensors(self, model: PreTrainedModel, save_path: Path):
        """Save model in safetensors format"""
        state_dict = model.state_dict()
        
        # Convert to CPU if on GPU
        cpu_state_dict = {}
        for key, tensor in state_dict.items():
            cpu_state_dict[key] = tensor.cpu()
        
        save_file(cpu_state_dict, save_path)
    
    def _get_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB"""
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def show_system_info(self):
        """Display system information"""
        info_panel = Panel.fit(
            f"""
🖥️  **System Information**
• Python: {torch.__version__ if torch else 'Not available'}
• PyTorch: {torch.__version__ if torch else 'Not available'}
• CUDA Available: {'✅ Yes' if torch.cuda.is_available() else '❌ No'}
• Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
• Models Path: {self.models_path}
• Downloaded Models: {len([m for m in self.SUPPORTED_MODELS.keys() if self.is_model_downloaded(m)])}
• Loaded Models: {len(self.loaded_models)}
            """,
            title="🤖 TERMINALIS-V.2 Model Manager",
            border_style="cyan"
        )
        console.print(info_panel)
