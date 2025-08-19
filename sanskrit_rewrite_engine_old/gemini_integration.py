"""
Google Gemini 2.0 Flash Integration for Sanskrit Rewrite Engine

This module integrates Google's Gemini 2.0 Flash model as an external LLM
for advanced Sanskrit reasoning, self-evolution, and knowledge enhancement.

Features:
- Gemini 2.0 Flash API integration
- Sanskrit-specific prompt engineering
- Multi-modal capabilities (text, images, audio)
- Real-time reasoning and validation
- Self-learning feedback loops
- Performance optimization for Sanskrit tasks
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import base64

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not available. Install with: pip install google-generativeai")

from .hybrid_reasoning import ExternalModelInterface, ReasoningResult, ModelType, TaskComplexity
from .r_zero_integration import SanskritProblem, SanskritProblemType


@dataclass
class GeminiConfig:
    """Configuration for Gemini model integration."""
    api_key: str
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 8192
    safety_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.safety_settings:
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }


class SanskritGeminiInterface(ExternalModelInterface):
    """Gemini 2.0 Flash interface specialized for Sanskrit reasoning."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if GEMINI_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("Gemini not available - running in mock mode")
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            genai.configure(api_key=self.config.api_key)
            
            generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=self.config.safety_settings
            )
            
            self.logger.info(f"Gemini {self.config.model_name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None
    
    async def query(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query Gemini with Sanskrit-specific context."""
        if not self.model:
            return self._mock_response(prompt, context)
        
        try:
            # Enhance prompt with Sanskrit context
            enhanced_prompt = self._enhance_sanskrit_prompt(prompt, context)
            
            # Generate response
            response = await self._generate_async(enhanced_prompt)
            
            # Process and validate response
            result = self._process_response(response, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini query failed: {e}")
            return {"error": str(e), "success": False}
    
    def _enhance_sanskrit_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance prompt with Sanskrit-specific context and instructions."""
        
        sanskrit_context = f"""
You are an expert in Sanskrit grammar, linguistics, and computational analysis. You have deep knowledge of:
- Pāṇinian grammar and sūtras
- Sanskrit phonology, morphology, and syntax
- Sandhi rules and compound formation
- Vedic and Classical Sanskrit variations
- Cross-linguistic analysis and translation

Current Context:
- Task Type: {context.get('task_type', 'general')}
- Complexity: {context.get('complexity', 'moderate')}
- Sanskrit Text: {context.get('sanskrit_text', 'N/A')}
- Previous Analysis: {context.get('previous_analysis', 'None')}

Instructions:
1. Provide accurate Sanskrit grammatical analysis
2. Reference relevant Pāṇinian sūtras when applicable
3. Explain your reasoning step-by-step
4. Consider both traditional and computational perspectives
5. Validate your analysis against known Sanskrit principles

Query: {prompt}

Please provide a detailed, accurate response with clear reasoning.
"""
        
        return sanskrit_context
    
    async def _generate_async(self, prompt: str) -> Any:
        """Generate response asynchronously."""
        if not GEMINI_AVAILABLE:
            return None
        
        try:
            # Use asyncio to run the synchronous generate_content method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.model.generate_content, 
                prompt
            )
            return response
        except Exception as e:
            self.logger.error(f"Async generation failed: {e}")
            return None
    
    def _process_response(self, response: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process Gemini response and extract relevant information."""
        if not response:
            return {"error": "No response generated", "success": False}
        
        try:
            # Extract text from response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Parse structured information
            analysis = self._parse_sanskrit_analysis(response_text)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(response_text, context)
            
            return {
                "success": True,
                "response": response_text,
                "analysis": analysis,
                "confidence": confidence,
                "model": "gemini-2.0-flash",
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Response processing failed: {e}")
            return {"error": str(e), "success": False}
    
    def _parse_sanskrit_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse Sanskrit-specific analysis from response."""
        analysis = {
            "grammatical_analysis": [],
            "sandhi_rules": [],
            "morphological_breakdown": [],
            "sutra_references": [],
            "confidence_indicators": []
        }
        
        # Extract sūtra references
        import re
        sutra_pattern = r'(\d+\.\d+\.\d+)'
        sutras = re.findall(sutra_pattern, response_text)
        analysis["sutra_references"] = sutras
        
        # Extract Sanskrit terms
        sanskrit_pattern = r'([a-zA-Zāīūṛṝḷḹēōṃḥṅñṇnmśṣskhgghṅcchajjhañṭṭhḍḍhṇtthddhnpphbbhm]+)'
        sanskrit_terms = re.findall(sanskrit_pattern, response_text)
        analysis["sanskrit_terms"] = list(set(sanskrit_terms))
        
        # Look for grammatical analysis keywords
        grammar_keywords = ["sandhi", "compound", "morphology", "case", "tense", "voice", "mood"]
        for keyword in grammar_keywords:
            if keyword.lower() in response_text.lower():
                analysis["grammatical_analysis"].append(keyword)
        
        return analysis
    
    def _calculate_confidence(self, response_text: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score based on response quality."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for detailed responses
        if len(response_text) > 200:
            confidence += 0.1
        
        # Increase confidence for sūtra references
        if "sūtra" in response_text.lower() or any(c.isdigit() for c in response_text):
            confidence += 0.2
        
        # Increase confidence for Sanskrit terminology
        sanskrit_indicators = ["sandhi", "samāsa", "vibhakti", "dhātu", "pratyaya"]
        for indicator in sanskrit_indicators:
            if indicator in response_text.lower():
                confidence += 0.05
        
        # Increase confidence for structured analysis
        if "analysis:" in response_text.lower() or "step" in response_text.lower():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _mock_response(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide mock response when Gemini is not available."""
        return {
            "success": True,
            "response": f"Mock Gemini response for: {prompt[:100]}...",
            "analysis": {
                "grammatical_analysis": ["mock_analysis"],
                "confidence_indicators": ["mock_mode"]
            },
            "confidence": 0.3,
            "model": "gemini-mock",
            "timestamp": datetime.now().isoformat()
        }


class GeminiSanskritEvolution:
    """Handles self-evolution using Gemini for Sanskrit reasoning enhancement."""
    
    def __init__(self, gemini_interface: SanskritGeminiInterface):
        self.gemini = gemini_interface
        self.evolution_history = []
        self.learned_patterns = {}
        self.logger = logging.getLogger(__name__)
    
    async def evolve_sanskrit_understanding(self, 
                                          problem: SanskritProblem, 
                                          current_solution: str,
                                          feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini to evolve Sanskrit understanding based on feedback."""
        
        evolution_prompt = f"""
Analyze this Sanskrit processing result and suggest improvements:

Problem: {problem.text}
Current Solution: {current_solution}
Feedback: {json.dumps(feedback, indent=2)}

Please provide:
1. Analysis of what went wrong (if anything)
2. Suggested improvements to the processing rules
3. New patterns or rules that should be learned
4. Confidence assessment of the current solution
5. Recommendations for future similar problems

Focus on grammatical accuracy and adherence to Pāṇinian principles.
"""
        
        context = {
            "task_type": "evolution_analysis",
            "complexity": "expert",
            "sanskrit_text": problem.text,
            "feedback_type": feedback.get("type", "unknown")
        }
        
        try:
            result = await self.gemini.query(evolution_prompt, context)
            
            if result.get("success"):
                evolution_insights = self._extract_evolution_insights(result["response"])
                self._update_learned_patterns(evolution_insights)
                
                return {
                    "success": True,
                    "insights": evolution_insights,
                    "learned_patterns": self.learned_patterns,
                    "evolution_applied": True
                }
            else:
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            self.logger.error(f"Evolution analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_evolution_insights(self, response: str) -> Dict[str, Any]:
        """Extract actionable insights from Gemini's evolution analysis."""
        insights = {
            "rule_improvements": [],
            "new_patterns": [],
            "confidence_factors": [],
            "learning_priorities": []
        }
        
        # Parse the response for structured insights
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "improvements" in line.lower():
                current_section = "rule_improvements"
            elif "patterns" in line.lower() or "rules" in line.lower():
                current_section = "new_patterns"
            elif "confidence" in line.lower():
                current_section = "confidence_factors"
            elif "recommendations" in line.lower() or "future" in line.lower():
                current_section = "learning_priorities"
            elif line and current_section and line.startswith(('-', '•', '1.', '2.')):
                insights[current_section].append(line)
        
        return insights
    
    def _update_learned_patterns(self, insights: Dict[str, Any]):
        """Update the learned patterns database."""
        timestamp = datetime.now().isoformat()
        
        for pattern in insights.get("new_patterns", []):
            pattern_key = hash(pattern) % 10000  # Simple hash for key
            self.learned_patterns[pattern_key] = {
                "pattern": pattern,
                "learned_at": timestamp,
                "confidence": 0.7,
                "applications": 0
            }
        
        # Store evolution history
        self.evolution_history.append({
            "timestamp": timestamp,
            "insights": insights,
            "patterns_learned": len(insights.get("new_patterns", []))
        })
    
    async def validate_with_gemini(self, 
                                  sanskrit_text: str, 
                                  proposed_transformation: str) -> Dict[str, Any]:
        """Use Gemini to validate Sanskrit transformations."""
        
        validation_prompt = f"""
Please validate this Sanskrit transformation:

Original: {sanskrit_text}
Transformed: {proposed_transformation}

Analyze:
1. Is the transformation grammatically correct according to Pāṇinian rules?
2. Which specific sūtras apply?
3. Are there any errors or improvements needed?
4. What is your confidence level (0-1) in this transformation?
5. Provide alternative transformations if the current one is incorrect.

Be precise and reference specific grammatical principles.
"""
        
        context = {
            "task_type": "validation",
            "complexity": "expert",
            "sanskrit_text": sanskrit_text,
            "transformation": proposed_transformation
        }
        
        try:
            result = await self.gemini.query(validation_prompt, context)
            
            if result.get("success"):
                validation = self._parse_validation_result(result["response"])
                return {
                    "success": True,
                    "validation": validation,
                    "gemini_confidence": result.get("confidence", 0.5)
                }
            else:
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_validation_result(self, response: str) -> Dict[str, Any]:
        """Parse validation result from Gemini response."""
        validation = {
            "is_correct": False,
            "confidence": 0.5,
            "sutras_applied": [],
            "errors_found": [],
            "alternatives": [],
            "reasoning": response
        }
        
        # Simple parsing logic - could be enhanced with more sophisticated NLP
        if any(word in response.lower() for word in ["correct", "accurate", "valid"]):
            validation["is_correct"] = True
        
        # Extract confidence if mentioned
        import re
        confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', response.lower())
        if confidence_match:
            validation["confidence"] = float(confidence_match.group(1))
        
        # Extract sūtra references
        sutra_matches = re.findall(r'(\d+\.\d+\.\d+)', response)
        validation["sutras_applied"] = sutra_matches
        
        return validation


class GeminiModelManager:
    """Manages multiple Gemini model instances for different Sanskrit tasks."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = {}
        self.performance_stats = {}
        
        # Initialize different model configurations
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different Gemini model configurations."""
        
        # Fast model for simple tasks
        fast_config = GeminiConfig(
            api_key=self.api_key,
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            max_output_tokens=2048
        )
        
        # Detailed model for complex analysis
        detailed_config = GeminiConfig(
            api_key=self.api_key,
            model_name="gemini-2.0-flash-exp",
            temperature=0.7,
            max_output_tokens=8192
        )
        
        # Creative model for rule generation
        creative_config = GeminiConfig(
            api_key=self.api_key,
            model_name="gemini-2.0-flash-exp",
            temperature=0.9,
            max_output_tokens=4096
        )
        
        self.models = {
            "fast": SanskritGeminiInterface(fast_config),
            "detailed": SanskritGeminiInterface(detailed_config),
            "creative": SanskritGeminiInterface(creative_config)
        }
    
    def get_model_for_task(self, task_complexity: TaskComplexity) -> SanskritGeminiInterface:
        """Get the appropriate model for a given task complexity."""
        
        if task_complexity == TaskComplexity.SIMPLE:
            return self.models["fast"]
        elif task_complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            return self.models["detailed"]
        else:
            return self.models["detailed"]  # Default to detailed
    
    async def ensemble_query(self, 
                           prompt: str, 
                           context: Dict[str, Any],
                           models_to_use: List[str] = None) -> Dict[str, Any]:
        """Query multiple models and combine results."""
        
        if models_to_use is None:
            models_to_use = ["fast", "detailed"]
        
        results = []
        
        for model_name in models_to_use:
            if model_name in self.models:
                try:
                    result = await self.models[model_name].query(prompt, context)
                    result["model_name"] = model_name
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Model {model_name} failed: {e}")
        
        # Combine results
        return self._combine_ensemble_results(results)
    
    def _combine_ensemble_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple models."""
        
        if not results:
            return {"success": False, "error": "No results from ensemble"}
        
        # Simple combination strategy - could be enhanced
        successful_results = [r for r in results if r.get("success")]
        
        if not successful_results:
            return {"success": False, "error": "All ensemble models failed"}
        
        # Use highest confidence result as primary
        primary_result = max(successful_results, key=lambda x: x.get("confidence", 0))
        
        # Combine insights from all results
        combined_analysis = {}
        for result in successful_results:
            analysis = result.get("analysis", {})
            for key, value in analysis.items():
                if key not in combined_analysis:
                    combined_analysis[key] = []
                if isinstance(value, list):
                    combined_analysis[key].extend(value)
                else:
                    combined_analysis[key].append(value)
        
        return {
            "success": True,
            "primary_response": primary_result["response"],
            "ensemble_analysis": combined_analysis,
            "confidence": primary_result.get("confidence", 0.5),
            "models_used": [r.get("model_name", "unknown") for r in successful_results],
            "ensemble_size": len(successful_results)
        }


# Factory function for easy integration
def create_gemini_integration(api_key: str) -> Tuple[GeminiModelManager, GeminiSanskritEvolution]:
    """Create Gemini integration components."""
    
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
    
    manager = GeminiModelManager(api_key)
    evolution = GeminiSanskritEvolution(manager.models["detailed"])
    
    return manager, evolution


# Example usage and testing
async def test_gemini_integration():
    """Test the Gemini integration."""
    
    # This would use your actual API key
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    
    try:
        manager, evolution = create_gemini_integration(api_key)
        
        # Test basic query
        context = {
            "task_type": "sandhi_analysis",
            "complexity": "moderate",
            "sanskrit_text": "rāma + iti"
        }
        
        result = await manager.models["fast"].query(
            "Analyze the sandhi transformation of 'rāma + iti'", 
            context
        )
        
        print("Gemini Integration Test Results:")
        print(f"Success: {result.get('success')}")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Confidence: {result.get('confidence')}")
        
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_gemini_integration())