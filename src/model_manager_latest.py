#!/usr/bin/env python3
"""
Latest Open-Source Model Manager for TERMINALIS-V.2
===================================================
Handles the latest and most powerful open-source AI models:
- WizardLM-34B (Latest coding powerhouse)
- DialoGPT-large (Advanced conversation)
- Phi-4 (Microsoft's latest reasoning model)
- Qwen3-8B (Alibaba's advanced instruct model)
- OpenHermes-2.5-Mistral-7B (Enhanced creative model)
- CodeLlama-13B-Instruct-GGUF (Meta's latest code model)
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from huggingface_hub import hf_hub_download, list_repo_files
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    repo_id: str
    filename: str
    size_gb: float
    model_type: str
    format: str
    quantization: Optional[str] = None
    context_length: int = 4096
    recommended_ram_gb: int = 8

class LatestModelManager:
    """Manager for the latest open-source AI models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.downloaded_models: Dict[str, str] = {}
        self._initialize_model_catalog()
    
    def _initialize_model_catalog(self):
        """Initialize catalog of latest open-source models"""
        self.model_catalog = {
            "WizardLM-34B": ModelInfo(
                name="WizardLM-34B",
                repo_id="WizardLM/WizardLM-34B-V1.0",
                filename="WizardLM-34B-V1.0.Q4_K_M.gguf",
                size_gb=20.5,
                model_type="coding",
                format="GGUF",
                quantization="Q4_K_M",
                context_length=32768,
                recommended_ram_gb=32
            ),
            "DialoGPT-large": ModelInfo(
                name="DialoGPT-large",
                repo_id="microsoft/DialoGPT-large",
                filename="pytorch_model.bin",
                size_gb=1.2,
                model_type="conversational",
                format="PyTorch",
                context_length=1024,
                recommended_ram_gb=4
            ),
            "Phi-4": ModelInfo(
                name="Phi-4",
                repo_id="microsoft/Phi-4",
                filename="Phi-4.Q4_K_M.gguf",
                size_gb=2.8,
                model_type="reasoning",
                format="GGUF",
                quantization="Q4_K_M",
                context_length=16384,
                recommended_ram_gb=8
            ),
            "Qwen3-8B": ModelInfo(
                name="Qwen3-8B-Instruct",
                repo_id="Qwen/Qwen2.5-8B-Instruct",
                filename="Qwen2.5-8B-Instruct.Q4_K_M.gguf",
                size_gb=4.8,
                model_type="reasoning",
                format="GGUF",
                quantization="Q4_K_M",
                context_length=32768,
                recommended_ram_gb=12
            ),
            "OpenHermes-2.5-Mistral-7B": ModelInfo(
                name="OpenHermes-2.5-Mistral-7B",
                repo_id="teknium/OpenHermes-2.5-Mistral-7B",
                filename="OpenHermes-2.5-Mistral-7B.Q4_K_M.gguf",
                size_gb=4.1,
                model_type="creative",
                format="GGUF",
                quantization="Q4_K_M",
                context_length=8192,
                recommended_ram_gb=8
            ),
            "CodeLlama-13B-Instruct": ModelInfo(
                name="CodeLlama-13B-Instruct",
                repo_id="codellama/CodeLlama-13b-Instruct-hf",
                filename="CodeLlama-13B-Instruct.Q4_K_M.gguf",
                size_gb=7.3,
                model_type="coding",
                format="GGUF",
                quantization="Q4_K_M",
                context_length=16384,
                recommended_ram_gb=16
            )
        }
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """List all available models"""
        return self.model_catalog
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.model_catalog.get(model_name)
    
    async def download_model(self, model_name: str, progress_callback=None) -> str:
        """Download a specific model"""
        if model_name not in self.model_catalog:
            raise ValueError(f"Model {model_name} not found in catalog")
        
        model_info = self.model_catalog[model_name]
        model_path = self.models_dir / model_info.filename
        
        # Check if already downloaded
        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            self.downloaded_models[model_name] = str(model_path)
            return str(model_path)
        
        logger.info(f"Downloading {model_name} from {model_info.repo_id}...")
        
        try:
            # Download from Hugging Face Hub
            downloaded_path = hf_hub_download(
                repo_id=model_info.repo_id,
                filename=model_info.filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            
            self.downloaded_models[model_name] = downloaded_path
            logger.info(f"Successfully downloaded {model_name} to {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            raise
    
    async def download_essential_models(self, progress_callback=None) -> Dict[str, bool]:
        """Download essential models for the system"""
        essential_models = [
            "WizardLM-34B",
            "Qwen3-8B", 
            "OpenHermes-2.5-Mistral-7B",
            "CodeLlama-13B-Instruct"
        ]
        
        results = {}
        for model in essential_models:
            try:
                await self.download_model(model, progress_callback)
                results[model] = True
                logger.info(f"✅ {model} downloaded successfully")
            except Exception as e:
                results[model] = False
                logger.error(f"❌ Failed to download {model}: {e}")
        
        return results
    
    def get_system_requirements(self) -> Dict[str, Any]:
        """Get system requirements for running the models"""
        total_storage_gb = sum(model.size_gb for model in self.model_catalog.values())
        max_ram_gb = max(model.recommended_ram_gb for model in self.model_catalog.values())
        
        return {
            "storage_required_gb": total_storage_gb,
            "ram_recommended_gb": max_ram_gb,
            "gpu_recommended": True,
            "cpu_cores_minimum": 4,
            "python_version": "3.8+",
            "supported_formats": ["GGUF", "PyTorch", "Safetensors"]
        }
    
    def check_compatibility(self) -> Dict[str, Any]:
        """Check system compatibility"""
        import psutil
        import platform
        
        system_info = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 1)
        }
        
        requirements = self.get_system_requirements()
        
        compatibility = {
            "system_info": system_info,
            "requirements": requirements,
            "compatible": (
                system_info["ram_gb"] >= requirements["ram_recommended_gb"] and
                system_info["disk_free_gb"] >= requirements["storage_required_gb"] and
                system_info["cpu_count"] >= requirements["cpu_cores_minimum"]
            ),
            "warnings": []
        }
        
        if system_info["ram_gb"] < requirements["ram_recommended_gb"]:
            compatibility["warnings"].append("Insufficient RAM for optimal performance")
        
        if system_info["disk_free_gb"] < requirements["storage_required_gb"]:
            compatibility["warnings"].append("Insufficient disk space for all models")
        
        return compatibility
    
    def create_model_config(self) -> Dict[str, Any]:
        """Create configuration for the downloaded models"""
        config = {
            "models": {},
            "system": {
                "models_directory": str(self.models_dir),
                "total_models": len(self.downloaded_models),
                "formats_supported": ["GGUF", "PyTorch", "Safetensors"]
            }
        }
        
        for model_name, model_path in self.downloaded_models.items():
            model_info = self.model_catalog[model_name]
            config["models"][model_name] = {
                "path": model_path,
                "type": model_info.model_type,
                "format": model_info.format,
                "context_length": model_info.context_length,
                "quantization": model_info.quantization,
                "size_gb": model_info.size_gb,
                "recommended_ram_gb": model_info.recommended_ram_gb
            }
        
        return config
    
    def save_config(self, config_path: str = "model_config.json"):
        """Save model configuration to file"""
        config = self.create_model_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model configuration saved to {config_path}")
    
    def show_model_status(self):
        """Display status of all models"""
        print("\n🤖 TERMINALIS-V.2 Model Status")
        print("=" * 50)
        
        for name, info in self.model_catalog.items():
            status = "✅ Downloaded" if name in self.downloaded_models else "⬇️ Available"
            print(f"{status} {name}")
            print(f"   Type: {info.model_type.title()}")
            print(f"   Size: {info.size_gb} GB")
            print(f"   Format: {info.format}")
            print(f"   Context: {info.context_length} tokens")
            print(f"   RAM Required: {info.recommended_ram_gb} GB")
            print()

async def main():
    """Demo the model manager"""
    manager = LatestModelManager()
    
    print("🚀 TERMINALIS-V.2 Latest Model Manager")
    print("=" * 50)
    
    # Show available models
    manager.show_model_status()
    
    # Check system compatibility
    compatibility = manager.check_compatibility()
    print("\n💻 System Compatibility Check")
    print("=" * 30)
    print(f"RAM: {compatibility['system_info']['ram_gb']} GB")
    print(f"Disk Space: {compatibility['system_info']['disk_free_gb']} GB")
    print(f"CPU Cores: {compatibility['system_info']['cpu_count']}")
    print(f"Compatible: {'✅ Yes' if compatibility['compatible'] else '❌ No'}")
    
    if compatibility['warnings']:
        print("\n⚠️ Warnings:")
        for warning in compatibility['warnings']:
            print(f"  - {warning}")
    
    print("\n📋 System Requirements")
    requirements = manager.get_system_requirements()
    print(f"Storage needed: {requirements['storage_required_gb']} GB")
    print(f"RAM recommended: {requirements['ram_recommended_gb']} GB")
    print(f"Supported formats: {', '.join(requirements['supported_formats'])}")

if __name__ == "__main__":
    asyncio.run(main())
