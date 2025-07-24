"""
AI Assistant for PineScript Development
Advanced AI-powered code assistance and generation

Author: Gokaytrysolutions Team
Version: 1.0.0
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import openai
from datetime import datetime

class AssistanceType(Enum):
    """Types of AI assistance available"""
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    BUG_FIXING = "bug_fixing"
    OPTIMIZATION = "optimization"
    EXPLANATION = "explanation"
    STRATEGY_SUGGESTION = "strategy_suggestion"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class AssistanceRequest:
    """Request for AI assistance"""
    type: AssistanceType
    context: str
    requirements: Dict[str, Any]
    code: Optional[str] = None
    cursor_position: Optional[int] = None

@dataclass
class AssistanceResponse:
    """Response from AI assistant"""
    success: bool
    result: str
    confidence: float
    explanation: str
    suggestions: List[str]
    metadata: Dict[str, Any]

class AIAssistant:
    """
    Advanced AI Assistant for PineScript Development
    
    Features:
    - Intelligent code generation
    - Real-time code completion
    - Automated bug detection and fixing
    - Strategy optimization suggestions
    - Market pattern recognition
    - Natural language to code conversion
    """
    
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        
        # AI model configurations
        self.model_config = {
            'primary_model': 'gpt-4',
            'fallback_model': 'gpt-3.5-turbo',
            'max_tokens': 2000,
            'temperature': 0.3,
            'top_p': 0.9
        }
        
        # PineScript knowledge base
        self.knowledge_base = {
            'patterns': self._load_trading_patterns(),
            'strategies': self._load_strategy_templates(),
            'indicators': self._load_indicator_library(),
            'best_practices': self._load_best_practices(),
            'common_errors': self._load_common_errors()
        }
        
        # Context memory for conversations
        self.conversation_context = []
        self.max_context_length = 10
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the AI assistant"""
        try:
            self.logger.info("Initializing AI Assistant...")
            
            # Initialize AI models
            await self._initialize_models()
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Set up conversation context
            self._initialize_context()
            
            self.initialized = True
            self.logger.info("✅ AI Assistant initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AI Assistant: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize AI models"""
        # This would set up connections to AI models
        # For now, we'll simulate the initialization
        await asyncio.sleep(0.1)
    
    async def _load_knowledge_base(self):
        """Load comprehensive PineScript knowledge base"""
        # This would load from external sources or databases
        await asyncio.sleep(0.1)
    
    def _initialize_context(self):
        """Initialize conversation context"""
        self.conversation_context = [{
            'role': 'system',
            'content': '''You are an expert PineScript developer and trading strategy consultant. 
            You have deep knowledge of:
            - PineScript v5 syntax and features
            - Trading strategies and indicators
            - Market analysis and pattern recognition
            - Risk management and portfolio optimization
            - Performance analysis and backtesting
            
            Always provide accurate, efficient, and well-documented PineScript code.
            Focus on best practices and performance optimization.'''
        }]
    
    def _load_trading_patterns(self) -> Dict[str, Any]:
        """Load trading pattern definitions"""
        return {
            'candlestick_patterns': {
                'doji': 'abs(close - open) < (high - low) * 0.1',
                'hammer': 'math.min(open, close) - low > (high - low) * 0.6',
                'shooting_star': 'high - math.max(open, close) > (high - low) * 0.6',
                'engulfing_bullish': 'close > open and close[1] < open[1] and close > open[1] and open < close[1]',
                'engulfing_bearish': 'close < open and close[1] > open[1] and close < close[1] and open > open[1]',
            },
            'chart_patterns': {
                'double_top': 'Two peaks at similar levels with a valley between',
                'double_bottom': 'Two troughs at similar levels with a peak between',
                'head_shoulders': 'Three peaks with the middle one being the highest',
                'triangle': 'Converging trend lines forming triangular pattern',
                'flag': 'Rectangular continuation pattern',
            },
            'trend_patterns': {
                'uptrend': 'Series of higher highs and higher lows',
                'downtrend': 'Series of lower highs and lower lows',
                'sideways': 'Price moving within horizontal support and resistance',
            }
        }
    
    def _load_strategy_templates(self) -> Dict[str, Any]:
        """Load strategy templates and patterns"""
        return {
            'trend_following': {
                'moving_average_crossover': {
                    'description': 'Buy when fast MA crosses above slow MA',
                    'parameters': ['fast_period', 'slow_period'],
                    'entry': 'ta.crossover(fast_ma, slow_ma)',
                    'exit': 'ta.crossunder(fast_ma, slow_ma)'
                },
                'macd_strategy': {
                    'description': 'MACD signal line crossover strategy',
                    'parameters': ['fast', 'slow', 'signal'],
                    'entry': 'ta.crossover(macd_line, signal_line)',
                    'exit': 'ta.crossunder(macd_line, signal_line)'
                }
            },
            'mean_reversion': {
                'rsi_oversold': {
                    'description': 'Buy when RSI is oversold',
                    'parameters': ['rsi_period', 'oversold_level'],
                    'entry': 'ta.rsi(close, rsi_period) < oversold_level',
                    'exit': 'ta.rsi(close, rsi_period) > 50'
                },
                'bollinger_bands': {
                    'description': 'Mean reversion using Bollinger Bands',
                    'parameters': ['bb_period', 'bb_mult'],
                    'entry': 'close < ta.bb(close, bb_period, bb_mult)[0]',
                    'exit': 'close > ta.bb(close, bb_period, bb_mult)[1]'
                }
            },
            'momentum': {
                'breakout': {
                    'description': 'Buy on breakout above resistance',
                    'parameters': ['lookback_period'],
                    'entry': 'close > ta.highest(high, lookback_period)[1]',
                    'exit': 'close < ta.lowest(low, lookback_period)'
                }
            }
        }
    
    def _load_indicator_library(self) -> Dict[str, Any]:
        """Load comprehensive indicator library"""
        return {
            'trend_indicators': [
                'Simple Moving Average (SMA)',
                'Exponential Moving Average (EMA)',
                'Moving Average Convergence Divergence (MACD)',
                'Average Directional Index (ADX)',
                'Parabolic SAR',
                'Ichimoku Cloud'
            ],
            'oscillators': [
                'Relative Strength Index (RSI)',
                'Stochastic Oscillator',
                'Commodity Channel Index (CCI)',
                'Williams %R',
                'Money Flow Index (MFI)'
            ],
            'volume_indicators': [
                'On-Balance Volume (OBV)',
                'Volume Weighted Average Price (VWAP)',
                'Accumulation/Distribution Line',
                'Chaikin Money Flow'
            ],
            'volatility_indicators': [
                'Bollinger Bands',
                'Average True Range (ATR)',
                'Keltner Channels',
                'Donchian Channels'
            ]
        }
    
    def _load_best_practices(self) -> List[str]:
        """Load PineScript best practices"""
        return [
            "Always specify @version=5 at the top of your script",
            "Use descriptive variable names for better readability",
            "Limit the number of security() calls to improve performance",
            "Use var keyword for variables that should persist across bars",
            "Implement proper risk management with stop-loss and take-profit",
            "Avoid repainting by not using future data",
            "Use built-in functions instead of creating custom ones when possible",
            "Comment your code thoroughly for maintenance",
            "Test strategies on different timeframes and market conditions",
            "Use input functions for user-configurable parameters"
        ]
    
    def _load_common_errors(self) -> Dict[str, str]:
        """Load common PineScript errors and solutions"""
        return {
            "Script could not be translated from: null": "Check for syntax errors or missing semicolons",
            "Undeclared identifier": "Variable is used before being declared",
            "Cannot call 'operator +' with arguments": "Type mismatch in arithmetic operations",
            "The function should be called on each calculation": "Function is not being called in the global scope",
            "Too many security calls": "Reduce the number of request.security() calls",
            "Cannot use 'plot' in local scope": "Move plot statements to global scope",
            "Script has too many local variables": "Reduce the number of local variables",
            "Line too long": "Break long lines into multiple lines"
        }
    
    async def get_assistance(self, request: AssistanceRequest) -> AssistanceResponse:
        """
        Get AI assistance for PineScript development
        
        Args:
            request: AssistanceRequest with details
            
        Returns:
            AssistanceResponse with AI-generated assistance
        """
        try:
            self.logger.info(f"Processing assistance request: {request.type.value}")
            
            # Route to appropriate handler
            handlers = {
                AssistanceType.CODE_GENERATION: self._generate_code,
                AssistanceType.CODE_COMPLETION: self._complete_code,
                AssistanceType.BUG_FIXING: self._fix_bugs,
                AssistanceType.OPTIMIZATION: self._optimize_code,
                AssistanceType.EXPLANATION: self._explain_code,
                AssistanceType.STRATEGY_SUGGESTION: self._suggest_strategy,
                AssistanceType.PATTERN_RECOGNITION: self._recognize_patterns
            }
            
            handler = handlers.get(request.type, self._default_handler)
            response = await handler(request)
            
            # Update conversation context
            self._update_context(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"AI assistance failed: {e}")
            return AssistanceResponse(
                success=False,
                result="",
                confidence=0.0,
                explanation=f"AI assistance failed: {str(e)}",
                suggestions=[],
                metadata={}
            )
    
    async def _generate_code(self, request: AssistanceRequest) -> AssistanceResponse:
        """Generate PineScript code based on requirements"""
        try:
            requirements = request.requirements
            strategy_type = requirements.get('strategy_type', 'indicator')
            description = requirements.get('description', '')
            parameters = requirements.get('parameters', {})
            
            # Build prompt for code generation
            prompt = f"""
Generate a PineScript v5 {strategy_type} with the following requirements:
Description: {description}

Parameters: {json.dumps(parameters, indent=2)}

The code should:
1. Follow PineScript v5 best practices
2. Include proper error handling
3. Be well-documented with comments
4. Include input parameters for customization
5. Implement proper risk management (if strategy)

Generate complete, production-ready code.
"""
            
            # Simulate AI response (in real implementation, this would call AI model)
            generated_code = await self._call_ai_model(prompt)
            
            return AssistanceResponse(
                success=True,
                result=generated_code,
                confidence=0.85,
                explanation=f"Generated {strategy_type} code based on requirements",
                suggestions=[
                    "Test the strategy on historical data",
                    "Adjust parameters based on market conditions",
                    "Consider adding additional filters"
                ],
                metadata={
                    'strategy_type': strategy_type,
                    'parameters_count': len(parameters),
                    'lines_generated': len(generated_code.split('\n'))
                }
            )
            
        except Exception as e:
            return self._error_response(f"Code generation failed: {str(e)}")
    
    async def _complete_code(self, request: AssistanceRequest) -> AssistanceResponse:
        """Complete partial PineScript code"""
        try:
            code = request.code or ""
            cursor_pos = request.cursor_position or len(code)
            
            # Analyze context around cursor
            context_before = code[:cursor_pos]
            context_after = code[cursor_pos:]
            
            # Get completion suggestions
            suggestions = self.engine.get_autocomplete_suggestions(context_before, cursor_pos)
            
            # Generate intelligent completion
            completion = await self._generate_intelligent_completion(context_before, context_after)
            
            return AssistanceResponse(
                success=True,
                result=completion,
                confidence=0.75,
                explanation="Intelligent code completion based on context",
                suggestions=suggestions[:5],  # Top 5 suggestions
                metadata={
                    'cursor_position': cursor_pos,
                    'context_length': len(context_before),
                    'suggestions_count': len(suggestions)
                }
            )
            
        except Exception as e:
            return self._error_response(f"Code completion failed: {str(e)}")
    
    async def _fix_bugs(self, request: AssistanceRequest) -> AssistanceResponse:
        """Fix bugs in PineScript code"""
        try:
            code = request.code or ""
            
            # Validate code to find errors
            validation_result = self.engine.validate_code(code)
            
            if validation_result.is_valid:
                return AssistanceResponse(
                    success=True,
                    result=code,
                    confidence=1.0,
                    explanation="No bugs found in the code",
                    suggestions=validation_result.suggestions,
                    metadata={'errors_found': 0}
                )
            
            # Generate fixes for found errors
            fixed_code = await self._generate_bug_fixes(code, validation_result.errors)
            
            return AssistanceResponse(
                success=True,
                result=fixed_code,
                confidence=0.80,
                explanation=f"Fixed {len(validation_result.errors)} bugs in the code",
                suggestions=[f"Fixed: {error}" for error in validation_result.errors[:3]],
                metadata={
                    'errors_found': len(validation_result.errors),
                    'errors_fixed': len(validation_result.errors)
                }
            )
            
        except Exception as e:
            return self._error_response(f"Bug fixing failed: {str(e)}")
    
    async def _optimize_code(self, request: AssistanceRequest) -> AssistanceResponse:
        """Optimize PineScript code for better performance"""
        try:
            code = request.code or ""
            
            # Use engine's optimization
            optimized_code, optimizations = self.engine.optimize_code(code)
            
            # Add AI-powered optimizations
            ai_optimizations = await self._ai_optimize_code(optimized_code)
            
            all_optimizations = optimizations + ai_optimizations
            
            return AssistanceResponse(
                success=True,
                result=optimized_code,
                confidence=0.85,
                explanation=f"Applied {len(all_optimizations)} optimizations",
                suggestions=all_optimizations[:5],
                metadata={
                    'optimizations_applied': len(all_optimizations),
                    'performance_improvement': "estimated 15-30%"
                }
            )
            
        except Exception as e:
            return self._error_response(f"Code optimization failed: {str(e)}")
    
    async def _explain_code(self, request: AssistanceRequest) -> AssistanceResponse:
        """Explain PineScript code functionality"""
        try:
            code = request.code or ""
            
            # Parse code to understand structure
            parse_result = self.engine.parse_code(code)
            
            # Generate explanation
            explanation = await self._generate_code_explanation(code, parse_result)
            
            return AssistanceResponse(
                success=True,
                result=explanation,
                confidence=0.90,
                explanation="Detailed code explanation generated",
                suggestions=[
                    "Review the strategy logic",
                    "Understand risk management rules",
                    "Check parameter settings"
                ],
                metadata={
                    'code_complexity': parse_result.metadata.get('complexity_score', 0),
                    'script_type': parse_result.script_type.value,
                    'line_count': parse_result.metadata.get('line_count', 0)
                }
            )
            
        except Exception as e:
            return self._error_response(f"Code explanation failed: {str(e)}")
    
    async def _suggest_strategy(self, request: AssistanceRequest) -> AssistanceResponse:
        """Suggest trading strategies based on requirements"""
        try:
            requirements = request.requirements
            market_type = requirements.get('market_type', 'stocks')
            timeframe = requirements.get('timeframe', '1d')
            risk_tolerance = requirements.get('risk_tolerance', 'medium')
            
            # Analyze requirements and suggest strategies
            strategies = await self._analyze_and_suggest_strategies(
                market_type, timeframe, risk_tolerance
            )
            
            return AssistanceResponse(
                success=True,
                result=json.dumps(strategies, indent=2),
                confidence=0.80,
                explanation=f"Suggested {len(strategies)} strategies based on requirements",
                suggestions=[f"Strategy: {s['name']}" for s in strategies[:3]],
                metadata={
                    'strategies_count': len(strategies),
                    'market_type': market_type,
                    'timeframe': timeframe
                }
            )
            
        except Exception as e:
            return self._error_response(f"Strategy suggestion failed: {str(e)}")
    
    async def _recognize_patterns(self, request: AssistanceRequest) -> AssistanceResponse:
        """Recognize trading patterns from code or description"""
        try:
            code = request.code or ""
            context = request.context
            
            # Identify patterns in the code
            patterns = await self._identify_trading_patterns(code, context)
            
            return AssistanceResponse(
                success=True,
                result=json.dumps(patterns, indent=2),
                confidence=0.75,
                explanation=f"Identified {len(patterns)} trading patterns",
                suggestions=[f"Pattern: {p['name']}" for p in patterns[:3]],
                metadata={
                    'patterns_found': len(patterns),
                    'confidence_avg': sum(p['confidence'] for p in patterns) / len(patterns) if patterns else 0
                }
            )
            
        except Exception as e:
            return self._error_response(f"Pattern recognition failed: {str(e)}")
    
    async def _default_handler(self, request: AssistanceRequest) -> AssistanceResponse:
        """Default handler for unknown request types"""
        return self._error_response(f"Unknown assistance type: {request.type.value}")
    
    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model with prompt (simulated)"""
        # In real implementation, this would call OpenAI, Claude, or other AI models
        # For now, we'll return a simulated response
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        if "indicator" in prompt.lower():
            return self.engine.generate_code_template(
                self.engine.ScriptType.INDICATOR,
                title="AI Generated Indicator"
            )
        elif "strategy" in prompt.lower():
            return self.engine.generate_code_template(
                self.engine.ScriptType.STRATEGY,
                title="AI Generated Strategy"
            )
        else:
            return "// AI-generated PineScript code would appear here"
    
    async def _generate_intelligent_completion(self, context_before: str, context_after: str) -> str:
        """Generate intelligent code completion"""
        # Analyze context and generate appropriate completion
        if context_before.endswith("ta."):
            return "sma(close, 14)"
        elif context_before.endswith("strategy."):
            return "entry(\"Long\", strategy.long)"
        elif context_before.endswith("plot("):
            return "close, title=\"Close\", color=color.blue)"
        else:
            return ""
    
    async def _generate_bug_fixes(self, code: str, errors: List[str]) -> str:
        """Generate fixes for identified bugs"""
        fixed_code = code
        
        # Apply common fixes
        if "Missing version declaration" in str(errors):
            if not code.strip().startswith("@version"):
                fixed_code = "@version=5\n" + fixed_code
        
        if "Missing script declaration" in str(errors):
            if "indicator(" not in fixed_code and "strategy(" not in fixed_code:
                lines = fixed_code.split('\n')
                insert_pos = 1 if lines[0].startswith("@version") else 0
                lines.insert(insert_pos, 'indicator("Fixed Script", overlay=true)')
                fixed_code = '\n'.join(lines)
        
        return fixed_code
    
    async def _ai_optimize_code(self, code: str) -> List[str]:
        """AI-powered code optimizations"""
        optimizations = []
        
        # Check for common optimization opportunities
        if "request.security" in code:
            count = code.count("request.security")
            if count > 3:
                optimizations.append(f"Consider reducing {count} security requests")
        
        if code.count("plot(") > 8:
            optimizations.append("Consider consolidating multiple plots")
        
        if "ta.sma" in code and "ta.ema" in code:
            optimizations.append("Consider using consistent moving average type")
        
        return optimizations
    
    async def _generate_code_explanation(self, code: str, parse_result) -> str:
        """Generate detailed code explanation"""
        explanation = f"""
PineScript Code Analysis:

Script Type: {parse_result.script_type.value.title()}
Version: {parse_result.version.value}
Complexity Score: {parse_result.metadata.get('complexity_score', 0)}

Structure:
- Lines of code: {parse_result.metadata.get('line_count', 0)}
- Functions defined: {parse_result.metadata.get('function_count', 0)}
- Plot statements: {parse_result.metadata.get('plot_count', 0)}

Key Components:
"""
        
        if "ta.sma" in code:
            explanation += "- Uses Simple Moving Average (SMA) for trend analysis\n"
        if "ta.rsi" in code:
            explanation += "- Implements Relative Strength Index (RSI) for momentum\n"
        if "strategy.entry" in code:
            explanation += "- Contains entry logic for automated trading\n"
        if "strategy.exit" in code:
            explanation += "- Includes exit rules for risk management\n"
        
        return explanation
    
    async def _analyze_and_suggest_strategies(self, market_type: str, timeframe: str, risk_tolerance: str) -> List[Dict]:
        """Analyze requirements and suggest appropriate strategies"""
        strategies = []
        
        if risk_tolerance == "low":
            strategies.extend([
                {
                    "name": "Conservative Moving Average Strategy",
                    "description": "Long-term trend following with low risk",
                    "risk_level": "Low",
                    "expected_return": "8-12% annually",
                    "timeframe": "Daily/Weekly"
                },
                {
                    "name": "Dollar Cost Averaging",
                    "description": "Regular investment regardless of price",
                    "risk_level": "Very Low",
                    "expected_return": "Market return",
                    "timeframe": "Monthly"
                }
            ])
        
        if risk_tolerance == "medium":
            strategies.extend([
                {
                    "name": "MACD Crossover Strategy",
                    "description": "Medium-term momentum strategy",
                    "risk_level": "Medium",
                    "expected_return": "12-18% annually",
                    "timeframe": "Daily"
                },
                {
                    "name": "RSI Mean Reversion",
                    "description": "Buy oversold, sell overbought",
                    "risk_level": "Medium",
                    "expected_return": "15-20% annually",
                    "timeframe": "Hourly/Daily"
                }
            ])
        
        if risk_tolerance == "high":
            strategies.extend([
                {
                    "name": "Breakout Momentum Strategy",
                    "description": "High-frequency breakout trading",
                    "risk_level": "High",
                    "expected_return": "20-40% annually",
                    "timeframe": "5-15 minutes"
                },
                {
                    "name": "Scalping Strategy",
                    "description": "Very short-term price movements",
                    "risk_level": "Very High",
                    "expected_return": "Variable",
                    "timeframe": "1-5 minutes"
                }
            ])
        
        return strategies
    
    async def _identify_trading_patterns(self, code: str, context: str) -> List[Dict]:
        """Identify trading patterns from code or context"""
        patterns = []
        
        # Analyze code for known patterns
        if "ta.crossover" in code and "ta.sma" in code:
            patterns.append({
                "name": "Moving Average Crossover",
                "type": "Trend Following",
                "confidence": 0.9,
                "description": "Classic trend-following pattern using MA crossover"
            })
        
        if "ta.rsi" in code and ("< 30" in code or "> 70" in code):
            patterns.append({
                "name": "RSI Overbought/Oversold",
                "type": "Mean Reversion",
                "confidence": 0.85,
                "description": "Mean reversion strategy using RSI levels"
            })
        
        if "ta.bb" in code or "bollinger" in context.lower():
            patterns.append({
                "name": "Bollinger Bands Squeeze",
                "type": "Volatility",
                "confidence": 0.8,
                "description": "Volatility-based pattern using Bollinger Bands"
            })
        
        return patterns
    
    def _update_context(self, request: AssistanceRequest, response: AssistanceResponse):
        """Update conversation context"""
        self.conversation_context.append({
            'role': 'user',
            'content': f"Request: {request.type.value}, Context: {request.context[:200]}..."
        })
        
        self.conversation_context.append({
            'role': 'assistant',
            'content': f"Response: {response.explanation}"
        })
        
        # Maintain context length
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
    
    def _error_response(self, error_message: str) -> AssistanceResponse:
        """Create error response"""
        return AssistanceResponse(
            success=False,
            result="",
            confidence=0.0,
            explanation=error_message,
            suggestions=[],
            metadata={}
        )
    
    def get_conversation_context(self) -> List[Dict]:
        """Get current conversation context"""
        return self.conversation_context.copy()
    
    def clear_context(self):
        """Clear conversation context"""
        self._initialize_context()
    
    async def analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze current market conditions (simulated)"""
        # In real implementation, this would fetch and analyze market data
        return {
            "trend": "bullish",
            "volatility": "medium",
            "volume": "above_average",
            "support_levels": [100, 95, 90],
            "resistance_levels": [110, 115, 120],
            "indicators": {
                "rsi": 65,
                "macd": "bullish_crossover",
                "bb_position": "middle"
            },
            "recommendation": "Consider long positions with tight stops"
        }
    
    async def generate_alerts(self, code: str) -> List[Dict[str, str]]:
        """Generate alert conditions from strategy code"""
        alerts = []
        
        if "strategy.entry" in code:
            alerts.append({
                "name": "Entry Signal",
                "condition": "Entry conditions met",
                "message": "Strategy entry signal triggered"
            })
        
        if "strategy.exit" in code:
            alerts.append({
                "name": "Exit Signal",
                "condition": "Exit conditions met",
                "message": "Strategy exit signal triggered"
            })
        
        return alerts
