# TERMINALIS-V.2 Agentic AI System Installer
# One-line install: iwr -useb https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/install.ps1 | iex

Write-Host "🤖 TERMINALIS-V.2 Agentic AI System Installer" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script requires Administrator privileges. Restarting as Administrator..." -ForegroundColor Red
    Start-Process PowerShell -Verb RunAs "-NoProfile -ExecutionPolicy Bypass -Command `"cd '$pwd'; & '$PSCommandPath';`""
    exit
}

# Set execution policy
Set-ExecutionPolicy Bypass -Scope Process -Force

# Create installation directory
$InstallPath = "$env:USERPROFILE\TERMINALIS-V2"
if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
}

Set-Location $InstallPath

Write-Host "📦 Installing TERMINALIS-V.2 System to: $InstallPath" -ForegroundColor Green

# Download and extract the system
Write-Host "⬇️  Downloading system files..." -ForegroundColor Yellow
$ProgressPreference = 'SilentlyContinue'

# Download main system files
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/src/main.py" -OutFile "main.py"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/requirements.txt" -OutFile "requirements.txt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/config.yaml" -OutFile "config.yaml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/setup.py" -OutFile "setup.py"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/agents/agent_manager.py" -OutFile "agents/agent_manager.py"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/tools/tool_manager.py" -OutFile "tools/tool_manager.py"

# Create directory structure
$Directories = @("models", "agents", "tools", "data", "logs", "scripts")
foreach ($dir in $Directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Check for Python
Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Installing Python..." -ForegroundColor Red
    # Install Python via winget
    winget install Python.Python.3.11
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
}

# Install Python dependencies
Write-Host "📚 Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install additional AI/ML packages
Write-Host "🧠 Installing AI/ML packages..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers accelerate datasets
python -m pip install langchain langchain-community langchain-openai
python -m pip install chromadb faiss-cpu
python -m pip install gradio streamlit
python -m pip install opencv-python pillow
python -m pip install scikit-learn pandas numpy matplotlib seaborn

# Download and setup latest open-source AI models
Write-Host "🤖 Setting up latest open-source AI models..." -ForegroundColor Yellow
python -c "
import sys
import asyncio
sys.path.append('src')
from model_manager_latest import LatestModelManager

# Initialize latest model manager
model_mgr = LatestModelManager()

# Show available models
print('🤖 Latest Open-Source Models Available:')
model_mgr.show_model_status()

# Check system compatibility
compatibility = model_mgr.check_compatibility()
print('\n💻 System Compatibility:')
print(f'RAM: {compatibility[\"system_info\"][\"ram_gb\"]} GB')
print(f'Storage: {compatibility[\"system_info\"][\"disk_free_gb\"]} GB')
print(f'Compatible: {\"✅ Yes\" if compatibility[\"compatible\"] else \"❌ Limited\"}')

# Download essential models asynchronously
async def setup_models():
    print('\n📥 Downloading essential models...')
    results = await model_mgr.download_essential_models()
    
    print('\n📊 Download Results:')
    for model, success in results.items():
        status = '✅ Success' if success else '❌ Failed'
        print(f'{status}: {model}')
    
    # Save configuration
    model_mgr.save_config()
    print('\n🎉 Latest model setup complete!')
    return results

# Run the async setup
if __name__ == '__main__':
    try:
        results = asyncio.run(setup_models())
    except Exception as e:
        print(f'⚠️ Model setup encountered issues: {e}')
        print('System will continue with basic functionality')
"

# Create desktop shortcut
Write-Host "🖥️  Creating desktop shortcut..." -ForegroundColor Yellow
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut("$env:USERPROFILE\Desktop\TERMINALIS-V.2.lnk")
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = "`"$InstallPath\main.py`""
$Shortcut.WorkingDirectory = $InstallPath
$Shortcut.IconLocation = "python.exe"
$Shortcut.Description = "TERMINALIS-V.2 Agentic AI System"
$Shortcut.Save()

# Create start menu entry
$StartMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
$StartMenuShortcut = $WScriptShell.CreateShortcut("$StartMenuPath\TERMINALIS-V.2.lnk")
$StartMenuShortcut.TargetPath = "python"
$StartMenuShortcut.Arguments = "`"$InstallPath\main.py`""
$StartMenuShortcut.WorkingDirectory = $InstallPath
$StartMenuShortcut.IconLocation = "python.exe"
$StartMenuShortcut.Description = "TERMINALIS-V.2 Agentic AI System"
$StartMenuShortcut.Save()

# Set up environment variables
Write-Host "🔧 Setting up environment variables..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("TERMINALIS_V2_PATH", $InstallPath, "User")

# Final setup
Write-Host "⚙️  Running final setup..." -ForegroundColor Yellow
python setup.py install

Write-Host ""
Write-Host "🎉 Installation Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "📁 Installation Path: $InstallPath" -ForegroundColor White
Write-Host "🖥️  Desktop shortcut created" -ForegroundColor White
Write-Host "📋 Start menu entry added" -ForegroundColor White
Write-Host ""
Write-Host "🚀 To start the system:" -ForegroundColor Cyan
Write-Host "   1. Use the desktop shortcut" -ForegroundColor White
Write-Host "   2. Or run: python `"$InstallPath\main.py`"" -ForegroundColor White
Write-Host "   3. Or search 'TERMINALIS-V.2' in Start Menu" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation: https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
