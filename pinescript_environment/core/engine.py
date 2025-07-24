"""
PineScript Engine - Core Development Engine
Comprehensive PineScript v5 support with all features

Author: Gokaytrysolutions Team
Version: 1.0.0
"""

import re
import ast
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import json

class PineScriptVersion(Enum):
    """PineScript version enumeration"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5 = "v5"

class ScriptType(Enum):
    """PineScript script type enumeration"""
    INDICATOR = "indicator"
    STRATEGY = "strategy"
    LIBRARY = "library"

@dataclass
class ParseResult:
    """Result of PineScript parsing"""
    success: bool
    errors: List[str]
    warnings: List[str]
    ast: Optional[Dict]
    metadata: Dict[str, Any]
    version: PineScriptVersion
    script_type: ScriptType

@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class PineScriptEngine:
    """
    Core PineScript Development Engine
    
    Features:
    - Complete PineScript v5 syntax support
    - Intelligent parsing and validation
    - Code generation and optimization
    - Real-time error detection
    - Performance analysis
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PineScript v5 Keywords and Functions
        self.keywords = {
            'version_declaration': ['@version=5'],
            'script_declaration': ['indicator', 'strategy', 'library'],
            'variable_declaration': ['var', 'varip'],
            'control_flow': ['if', 'else', 'for', 'while', 'switch'],
            'data_types': ['int', 'float', 'bool', 'string', 'color', 'line', 'label', 'table', 'box'],
            'builtin_variables': ['open', 'high', 'low', 'close', 'volume', 'time', 'bar_index'],
            'builtin_functions': [
                'ta.sma', 'ta.ema', 'ta.rsi', 'ta.macd', 'ta.stoch', 'ta.bb',
                'request.security', 'str.tostring', 'array.new', 'map.new',
                'math.abs', 'math.max', 'math.min', 'math.round', 'math.floor',
                'plot', 'plotshape', 'plotchar', 'bgcolor', 'fill',
                'strategy.entry', 'strategy.exit', 'strategy.close'
            ]
        }
        
        # PineScript v5 Syntax Patterns
        self.syntax_patterns = {
            'version': r'@version\s*=\s*[1-5]',
            'declaration': r'(indicator|strategy|library)\s*\(',
            'variable': r'(var|varip)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            'function_def': r'([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*=>)',
            'plot': r'plot\s*\(',
            'if_statement': r'if\s+',
            'for_loop': r'for\s+',
            'array': r'array\.(new|push|pop|get|set|size)',
            'map': r'map\.(new|put|get|remove|size)',
            'request': r'request\.(security|dividends|splits|earnings)',
            'ta_functions': r'ta\.[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
            'math_functions': r'math\.[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
            'strategy_functions': r'strategy\.[a-zA-Z_][a-zA-Z0-9_]*\s*\('
        }
        
        # Built-in functions with signatures
        self.function_signatures = {
            'ta.sma': 'ta.sma(source, length)',
            'ta.ema': 'ta.ema(source, length)',
            'ta.rsi': 'ta.rsi(source, length)',
            'ta.macd': 'ta.macd(source, fast, slow, signal)',
            'ta.stoch': 'ta.stoch(source, high, low, length)',
            'ta.bb': 'ta.bb(source, length, mult)',
            'request.security': 'request.security(symbol, timeframe, expression)',
            'plot': 'plot(series, title, color, linewidth, style)',
            'strategy.entry': 'strategy.entry(id, direction, qty, limit, stop)',
            'strategy.exit': 'strategy.exit(id, from_entry, loss, profit, trail_price)',
            'array.new<type>': 'array.new<type>(size, initial_value)',
            'map.new<key_type, value_type>': 'map.new<key_type, value_type>()',
        }
        
        # Error patterns and fixes
        self.common_errors = {
            'missing_version': {
                'pattern': r'^(?!.*@version)',
                'message': 'Missing version declaration',
                'fix': '@version=5'
            },
            'invalid_syntax': {
                'pattern': r'[^\w\s\(\)\[\]\{\}\+\-\*\/\%\=\!\<\>\&\|\?\:\;\,\.]',
                'message': 'Invalid character in code',
                'fix': 'Remove invalid characters'
            },
            'missing_parentheses': {
                'pattern': r'\w+\s*\[.*\]\s*(?!\()',
                'message': 'Missing parentheses for function call',
                'fix': 'Add parentheses after function name'
            }
        }
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the PineScript engine"""
        try:
            self.logger.info("Initializing PineScript Engine...")
            
            # Load additional configurations
            await self._load_syntax_rules()
            await self._load_builtin_functions()
            await self._initialize_validators()
            
            self.initialized = True
            self.logger.info("✅ PineScript Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize PineScript Engine: {e}")
            raise
    
    async def _load_syntax_rules(self):
        """Load comprehensive syntax rules for all PineScript versions"""
        # This would load from configuration files or database
        pass
    
    async def _load_builtin_functions(self):
        """Load all built-in functions and their signatures"""
        # This would load from PineScript documentation or API
        pass
    
    async def _initialize_validators(self):
        """Initialize code validators"""
        # Set up validation rules and patterns
        pass
    
    def parse_code(self, code: str) -> ParseResult:
        """
        Parse PineScript code and return detailed analysis
        
        Args:
            code: PineScript source code
            
        Returns:
            ParseResult with parsing information
        """
        try:
            errors = []
            warnings = []
            metadata = {}
            
            # Detect version
            version_match = re.search(self.syntax_patterns['version'], code)
            if version_match:
                version_str = version_match.group(0).split('=')[1].strip()
                version = PineScriptVersion(f"v{version_str}")
            else:
                version = PineScriptVersion.V5  # Default to v5
                warnings.append("No version specified, defaulting to v5")
            
            # Detect script type
            script_type_match = re.search(self.syntax_patterns['declaration'], code)
            if script_type_match:
                script_type_str = script_type_match.group(1)
                script_type = ScriptType(script_type_str)
            else:
                script_type = ScriptType.INDICATOR  # Default
                errors.append("No script declaration found")
            
            # Parse structure
            ast_tree = self._build_ast(code)
            
            # Extract metadata
            metadata = {
                'line_count': len(code.split('\n')),
                'function_count': len(re.findall(self.syntax_patterns['function_def'], code)),
                'plot_count': len(re.findall(self.syntax_patterns['plot'], code)),
                'complexity_score': self._calculate_complexity(code)
            }
            
            return ParseResult(
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                ast=ast_tree,
                metadata=metadata,
                version=version,
                script_type=script_type
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"Parsing failed: {str(e)}"],
                warnings=[],
                ast=None,
                metadata={},
                version=PineScriptVersion.V5,
                script_type=ScriptType.INDICATOR
            )
    
    def _build_ast(self, code: str) -> Dict:
        """Build Abstract Syntax Tree from PineScript code"""
        ast_tree = {
            'type': 'script',
            'children': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Parse different statement types
            if re.match(self.syntax_patterns['version'], line):
                ast_tree['children'].append({
                    'type': 'version_declaration',
                    'line': i + 1,
                    'content': line
                })
            elif re.match(self.syntax_patterns['declaration'], line):
                ast_tree['children'].append({
                    'type': 'script_declaration',
                    'line': i + 1,
                    'content': line
                })
            elif re.match(self.syntax_patterns['variable'], line):
                ast_tree['children'].append({
                    'type': 'variable_declaration',
                    'line': i + 1,
                    'content': line
                })
            elif re.match(self.syntax_patterns['plot'], line):
                ast_tree['children'].append({
                    'type': 'plot_statement',
                    'line': i + 1,
                    'content': line
                })
            else:
                ast_tree['children'].append({
                    'type': 'statement',
                    'line': i + 1,
                    'content': line
                })
        
        return ast_tree
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity score"""
        complexity = 0
        
        # Count control flow statements
        complexity += len(re.findall(r'\bif\b', code)) * 2
        complexity += len(re.findall(r'\bfor\b', code)) * 3
        complexity += len(re.findall(r'\bwhile\b', code)) * 3
        complexity += len(re.findall(r'\bswitch\b', code)) * 2
        
        # Count function definitions
        complexity += len(re.findall(self.syntax_patterns['function_def'], code)) * 1
        
        # Count nested structures
        complexity += code.count('{') * 1
        
        return complexity
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate PineScript code for syntax and logical errors
        
        Args:
            code: PineScript source code
            
        Returns:
            ValidationResult with validation information
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Check for common errors
            for error_type, error_info in self.common_errors.items():
                if re.search(error_info['pattern'], code):
                    errors.append(error_info['message'])
                    suggestions.append(error_info['fix'])
            
            # Check syntax patterns
            if not re.search(self.syntax_patterns['version'], code):
                warnings.append("Consider adding version declaration for better compatibility")
            
            if not re.search(self.syntax_patterns['declaration'], code):
                errors.append("Missing script declaration (indicator, strategy, or library)")
            
            # Check for undefined variables
            variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
            used_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
            
            for var in used_vars:
                if var not in variables and var not in self.keywords['builtin_variables']:
                    if var not in [kw for kw_list in self.keywords.values() for kw in kw_list]:
                        warnings.append(f"Variable '{var}' may be undefined")
            
            # Performance suggestions
            if code.count('request.security') > 5:
                suggestions.append("Consider reducing request.security calls for better performance")
            
            if len(re.findall(self.syntax_patterns['plot'], code)) > 10:
                suggestions.append("Too many plot statements may slow down the script")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=[]
            )
    
    def optimize_code(self, code: str) -> Tuple[str, List[str]]:
        """
        Optimize PineScript code for better performance
        
        Args:
            code: Original PineScript code
            
        Returns:
            Tuple of (optimized_code, optimization_notes)
        """
        optimizations = []
        optimized_code = code
        
        try:
            # Remove redundant calculations
            lines = optimized_code.split('\n')
            optimized_lines = []
            seen_calculations = set()
            
            for line in lines:
                # Check for duplicate calculations
                if '=' in line and not line.strip().startswith('//'):
                    calc_part = line.split('=')[1].strip()
                    if calc_part in seen_calculations:
                        optimizations.append(f"Removed duplicate calculation: {calc_part}")
                        continue
                    seen_calculations.add(calc_part)
                
                optimized_lines.append(line)
            
            optimized_code = '\n'.join(optimized_lines)
            
            # Optimize common patterns
            optimizations_map = {
                r'ta\.sma\(ta\.sma\(([^,]+),\s*(\d+)\),\s*(\d+)\)': r'ta.sma(\1, \2 + \3)',
                r'close\s*>\s*open\s*and\s*open\s*>\s*close\[1\]': r'close > open and open > close[1]',
            }
            
            for pattern, replacement in optimizations_map.items():
                if re.search(pattern, optimized_code):
                    optimized_code = re.sub(pattern, replacement, optimized_code)
                    optimizations.append(f"Optimized pattern: {pattern}")
            
            return optimized_code, optimizations
            
        except Exception as e:
            self.logger.error(f"Code optimization failed: {e}")
            return code, [f"Optimization failed: {str(e)}"]
    
    def generate_code_template(self, script_type: ScriptType, **kwargs) -> str:
        """
        Generate PineScript code template
        
        Args:
            script_type: Type of script to generate
            **kwargs: Additional parameters for template generation
            
        Returns:
            Generated PineScript template
        """
        templates = {
            ScriptType.INDICATOR: self._generate_indicator_template,
            ScriptType.STRATEGY: self._generate_strategy_template,
            ScriptType.LIBRARY: self._generate_library_template
        }
        
        generator = templates.get(script_type, self._generate_indicator_template)
        return generator(**kwargs)
    
    def _generate_indicator_template(self, **kwargs) -> str:
        """Generate indicator template"""
        title = kwargs.get('title', 'Custom Indicator')
        short_title = kwargs.get('short_title', 'CI')
        
        template = f'''// {title}
// © Gokaytrysolutions

@version=5
indicator(title="{title}", shorttitle="{short_title}", overlay=true)

// Input parameters
length = input.int(14, title="Length", minval=1)
source = input.source(close, title="Source")

// Calculations
value = ta.sma(source, length)

// Plot
plot(value, title="Value", color=color.blue, linewidth=2)

// Optional: Add alerts
alertcondition(ta.crossover(source, value), title="Crossover Alert", message="Source crossed over indicator")
alertcondition(ta.crossunder(source, value), title="Crossunder Alert", message="Source crossed under indicator")
'''
        return template
    
    def _generate_strategy_template(self, **kwargs) -> str:
        """Generate strategy template"""
        title = kwargs.get('title', 'Custom Strategy')
        
        template = f'''// {title}
// © Gokaytrysolutions

@version=5
strategy(title="{title}", overlay=true, default_qty_type=strategy.percent_of_equity)

// Input parameters
fast_length = input.int(12, title="Fast Length", minval=1)
slow_length = input.int(26, title="Slow Length", minval=1)
signal_length = input.int(9, title="Signal Length", minval=1)
risk_percent = input.float(1.0, title="Risk %", minval=0.1, maxval=5.0)

// Calculations
[macd_line, signal_line, histogram] = ta.macd(close, fast_length, slow_length, signal_length)

// Entry conditions
long_condition = ta.crossover(macd_line, signal_line) and histogram > 0
short_condition = ta.crossunder(macd_line, signal_line) and histogram < 0

// Risk management
atr_value = ta.atr(14)
risk_amount = strategy.equity * (risk_percent / 100)

// Strategy execution
if long_condition
    strategy.entry("Long", strategy.long)
    
if short_condition
    strategy.entry("Short", strategy.short)

// Exit conditions
strategy.exit("Long Exit", from_entry="Long", stop=close - (2 * atr_value), limit=close + (3 * atr_value))
strategy.exit("Short Exit", from_entry="Short", stop=close + (2 * atr_value), limit=close - (3 * atr_value))

// Plot indicators
plot(macd_line, title="MACD", color=color.blue)
plot(signal_line, title="Signal", color=color.red)
'''
        return template
    
    def _generate_library_template(self, **kwargs) -> str:
        """Generate library template"""
        title = kwargs.get('title', 'Custom Library')
        
        template = f'''// {title}
// © Gokaytrysolutions

@version=5
library("{title.lower().replace(' ', '_')}")

// Export functions
export sma_cross(simple float source, simple int fast_length, simple int slow_length) =>
    fast_ma = ta.sma(source, fast_length)
    slow_ma = ta.sma(source, slow_length)
    [fast_ma, slow_ma, ta.crossover(fast_ma, slow_ma), ta.crossunder(fast_ma, slow_ma)]

export rsi_overbought_oversold(simple float source, simple int length, simple float overbought = 70, simple float oversold = 30) =>
    rsi_value = ta.rsi(source, length)
    [rsi_value, rsi_value > overbought, rsi_value < oversold]

export atr_stop_loss(simple float atr_length = 14, simple float multiplier = 2.0) =>
    atr_value = ta.atr(atr_length)
    long_stop = close - (atr_value * multiplier)
    short_stop = close + (atr_value * multiplier)
    [long_stop, short_stop, atr_value]
'''
        return template
    
    def get_function_help(self, function_name: str) -> Optional[str]:
        """Get help information for a PineScript function"""
        return self.function_signatures.get(function_name)
    
    def get_autocomplete_suggestions(self, partial_code: str, cursor_position: int) -> List[str]:
        """Get autocomplete suggestions for the current cursor position"""
        suggestions = []
        
        # Get the current word being typed
        lines = partial_code[:cursor_position].split('\n')
        current_line = lines[-1] if lines else ""
        
        # Find the word at cursor
        words = current_line.split()
        current_word = words[-1] if words else ""
        
        # Add keyword suggestions
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.startswith(current_word.lower()):
                    suggestions.append(keyword)
        
        # Add function suggestions
        for func in self.function_signatures.keys():
            if func.startswith(current_word.lower()):
                suggestions.append(func)
        
        return sorted(list(set(suggestions)))
    
    def format_code(self, code: str) -> str:
        """Format PineScript code with proper indentation and spacing"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines and comments
            if not stripped_line or stripped_line.startswith('//'):
                formatted_lines.append(line)
                continue
            
            # Decrease indent for closing braces
            if stripped_line.startswith('}') or stripped_line.startswith(']') or stripped_line.startswith(')'):
                indent_level = max(0, indent_level - 1)
            
            # Add proper indentation
            formatted_line = '    ' * indent_level + stripped_line
            formatted_lines.append(formatted_line)
            
            # Increase indent for opening braces
            if stripped_line.endswith('{') or stripped_line.endswith('[') or stripped_line.endswith('('):
                indent_level += 1
            elif any(keyword in stripped_line for keyword in ['if ', 'for ', 'while ']):
                if not stripped_line.endswith('{'):
                    indent_level += 1
        
        return '\n'.join(formatted_lines)
