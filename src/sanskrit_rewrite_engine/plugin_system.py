"""
Plugin system implementation for the Sanskrit Rewrite Engine.

This module provides a comprehensive plugin system that allows for extensible
linguistic processing components, including example plugins and utilities
for plugin development.
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Type
from pathlib import Path
import importlib.util
import inspect
import json

from .interfaces import (
    ProcessingStage, ProcessingContext, PluginInterface,
    AdvancedToken, AdvancedRule
)
from .future_architecture import (
    LinguisticPlugin, AdvancedSanskritToken, ExtensibleSutraRule,
    SutraReference, SutraCategory
)


logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for plugin discovery and management."""
    name: str
    version: str
    author: str
    description: str
    supported_stages: List[ProcessingStage]
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = "main"
    license: Optional[str] = None
    homepage: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class PluginLoader:
    """Loader for discovering and loading plugins."""
    
    def __init__(self, plugin_directories: List[str]):
        self.plugin_directories = plugin_directories
        self._loaded_plugins: Dict[str, PluginInterface] = {}
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins in configured directories."""
        discovered = []
        
        for plugin_dir in self.plugin_directories:
            if not os.path.exists(plugin_dir):
                continue
            
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                # Check for plugin directory with metadata
                if os.path.isdir(item_path):
                    metadata_file = os.path.join(item_path, 'plugin.json')
                    if os.path.exists(metadata_file):
                        try:
                            metadata = self._load_plugin_metadata(metadata_file)
                            discovered.append(metadata)
                        except Exception as e:
                            logger.error(f"Failed to load plugin metadata from {metadata_file}: {e}")
                
                # Check for single Python file plugins
                elif item.endswith('.py') and not item.startswith('__'):
                    try:
                        metadata = self._discover_python_plugin(item_path)
                        if metadata:
                            discovered.append(metadata)
                    except Exception as e:
                        logger.error(f"Failed to discover plugin from {item_path}: {e}")
        
        return discovered
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[PluginInterface]:
        """Load a specific plugin by name."""
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name]
        
        # Find plugin metadata
        metadata = self._plugin_metadata.get(plugin_name)
        if not metadata:
            # Try to discover it
            discovered = self.discover_plugins()
            for meta in discovered:
                self._plugin_metadata[meta.name] = meta
                if meta.name == plugin_name:
                    metadata = meta
                    break
        
        if not metadata:
            logger.error(f"Plugin not found: {plugin_name}")
            return None
        
        try:
            plugin_instance = self._load_plugin_instance(metadata, config)
            if plugin_instance:
                self._loaded_plugins[plugin_name] = plugin_instance
                logger.info(f"Loaded plugin: {plugin_name}")
            return plugin_instance
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name in self._loaded_plugins:
            plugin = self._loaded_plugins[plugin_name]
            plugin.cleanup()
            del self._loaded_plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        return False
    
    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """Get all loaded plugins."""
        return self._loaded_plugins.copy()
    
    def _load_plugin_metadata(self, metadata_file: str) -> PluginMetadata:
        """Load plugin metadata from JSON file."""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return PluginMetadata(
            name=data['name'],
            version=data['version'],
            author=data['author'],
            description=data['description'],
            supported_stages=[ProcessingStage(stage) for stage in data['supported_stages']],
            dependencies=data.get('dependencies', []),
            config_schema=data.get('config_schema', {}),
            entry_point=data.get('entry_point', 'main'),
            license=data.get('license'),
            homepage=data.get('homepage'),
            tags=data.get('tags', [])
        )
    
    def _discover_python_plugin(self, python_file: str) -> Optional[PluginMetadata]:
        """Discover plugin metadata from Python file."""
        try:
            spec = importlib.util.spec_from_file_location("temp_plugin", python_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    
                    # Try to get metadata from class attributes
                    plugin_name = getattr(obj, 'PLUGIN_NAME', name)
                    plugin_version = getattr(obj, 'PLUGIN_VERSION', '1.0.0')
                    plugin_author = getattr(obj, 'PLUGIN_AUTHOR', 'Unknown')
                    plugin_description = getattr(obj, 'PLUGIN_DESCRIPTION', 'No description')
                    
                    # Create temporary instance to get supported stages
                    temp_instance = obj()
                    supported_stages = temp_instance.supported_stages
                    
                    return PluginMetadata(
                        name=plugin_name,
                        version=plugin_version,
                        author=plugin_author,
                        description=plugin_description,
                        supported_stages=supported_stages
                    )
            
        except Exception as e:
            logger.error(f"Error discovering plugin from {python_file}: {e}")
        
        return None
    
    def _load_plugin_instance(self, metadata: PluginMetadata, 
                            config: Optional[Dict[str, Any]]) -> Optional[PluginInterface]:
        """Load plugin instance from metadata."""
        # Find the plugin file
        plugin_file = None
        for plugin_dir in self.plugin_directories:
            # Check for directory-based plugin
            plugin_path = os.path.join(plugin_dir, metadata.name)
            if os.path.isdir(plugin_path):
                main_file = os.path.join(plugin_path, f"{metadata.entry_point}.py")
                if os.path.exists(main_file):
                    plugin_file = main_file
                    break
            
            # Check for single file plugin
            single_file = os.path.join(plugin_dir, f"{metadata.name}.py")
            if os.path.exists(single_file):
                plugin_file = single_file
                break
        
        if not plugin_file:
            logger.error(f"Plugin file not found for: {metadata.name}")
            return None
        
        # Load the plugin module
        spec = importlib.util.spec_from_file_location(f"plugin_{metadata.name}", plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find and instantiate plugin class
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, PluginInterface) and 
                obj != PluginInterface):
                
                plugin_instance = obj()
                plugin_instance.initialize(config or {})
                return plugin_instance
        
        return None


# Example plugin implementations

class SandhiAnalysisPlugin(LinguisticPlugin):
    """Example plugin for sandhi analysis."""
    
    PLUGIN_NAME = "sandhi_analysis"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "Sanskrit Engine Team"
    PLUGIN_DESCRIPTION = "Advanced sandhi analysis and splitting"
    
    def __init__(self):
        super().__init__(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            supported_stages=[ProcessingStage.MORPHOLOGICAL_ANALYSIS]
        )
        self._sandhi_patterns: List[Dict[str, Any]] = []
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with sandhi patterns."""
        super().initialize(config)
        
        # Load sandhi patterns from config or defaults
        self._sandhi_patterns = config.get('sandhi_patterns', [
            {
                'pattern': r'([aā])([iī])',
                'result': 'e',
                'type': 'vowel_sandhi',
                'description': 'a/ā + i/ī → e'
            },
            {
                'pattern': r'([aā])([uū])',
                'result': 'o',
                'type': 'vowel_sandhi',
                'description': 'a/ā + u/ū → o'
            }
        ])
    
    def process(self, data: Any, context: ProcessingContext) -> Any:
        """Process data for sandhi analysis."""
        if isinstance(data, AdvancedSanskritToken):
            return self._analyze_sandhi(data, context)
        elif isinstance(data, list):
            return [self.process(item, context) for item in data]
        return data
    
    def _analyze_sandhi(self, token: AdvancedSanskritToken, 
                       context: ProcessingContext) -> AdvancedSanskritToken:
        """Analyze sandhi in a token."""
        surface_form = token.surface_form
        
        # Check for sandhi patterns
        for pattern_info in self._sandhi_patterns:
            import re
            pattern = pattern_info['pattern']
            if re.search(pattern, surface_form):
                # Add sandhi analysis to token metadata
                token.token_metadata.morphological_features['sandhi_detected'] = True
                token.token_metadata.morphological_features['sandhi_type'] = pattern_info['type']
                token.token_metadata.morphological_features['sandhi_description'] = pattern_info['description']
                
                # Add confidence score
                token.token_metadata.confidence_scores['sandhi_analysis'] = 0.85
                
                break
        
        return token


class CompoundAnalysisPlugin(LinguisticPlugin):
    """Example plugin for compound word analysis."""
    
    PLUGIN_NAME = "compound_analysis"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "Sanskrit Engine Team"
    PLUGIN_DESCRIPTION = "Analysis of Sanskrit compound words"
    
    def __init__(self):
        super().__init__(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            supported_stages=[ProcessingStage.MORPHOLOGICAL_ANALYSIS, ProcessingStage.SYNTACTIC_ANALYSIS]
        )
        self._compound_types: List[str] = []
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with compound types."""
        super().initialize(config)
        
        self._compound_types = config.get('compound_types', [
            'tatpurusha',  # Determinative compound
            'bahuvrihi',   # Possessive compound
            'dvandva',     # Copulative compound
            'avyayibhava', # Adverbial compound
            'karmadharaya' # Descriptive compound
        ])
    
    def process(self, data: Any, context: ProcessingContext) -> Any:
        """Process data for compound analysis."""
        if isinstance(data, AdvancedSanskritToken):
            return self._analyze_compound(data, context)
        elif isinstance(data, list):
            return [self.process(item, context) for item in data]
        return data
    
    def _analyze_compound(self, token: AdvancedSanskritToken,
                         context: ProcessingContext) -> AdvancedSanskritToken:
        """Analyze compound structure in a token."""
        surface_form = token.surface_form
        
        # Simple heuristic: words longer than 6 characters might be compounds
        if len(surface_form) > 6:
            # Add compound analysis
            token.token_metadata.morphological_features['potential_compound'] = True
            token.token_metadata.morphological_features['compound_length'] = len(surface_form)
            
            # Try to identify compound boundaries (simplified)
            boundaries = self._find_compound_boundaries(surface_form)
            if boundaries:
                token.set_morpheme_boundaries(boundaries)
                token.token_metadata.morphological_features['compound_boundaries'] = boundaries
                token.token_metadata.confidence_scores['compound_analysis'] = 0.7
        
        return token
    
    def _find_compound_boundaries(self, word: str) -> List[int]:
        """Find potential compound boundaries in a word."""
        boundaries = []
        
        # Simple heuristic: look for vowel-consonant transitions
        for i in range(1, len(word) - 1):
            prev_char = word[i-1]
            curr_char = word[i]
            
            # Check for vowel followed by consonant (potential boundary)
            if (prev_char in 'aeiouāīūṛṝḷḹeo' and 
                curr_char not in 'aeiouāīūṛṝḷḹeo'):
                boundaries.append(i)
        
        return boundaries


class MeterAnalysisPlugin(LinguisticPlugin):
    """Example plugin for Sanskrit meter analysis."""
    
    PLUGIN_NAME = "meter_analysis"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "Sanskrit Engine Team"
    PLUGIN_DESCRIPTION = "Analysis of Sanskrit poetic meters"
    
    def __init__(self):
        super().__init__(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            supported_stages=[ProcessingStage.SEMANTIC_ANALYSIS]
        )
        self._meter_patterns: Dict[str, str] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with meter patterns."""
        super().initialize(config)
        
        self._meter_patterns = config.get('meter_patterns', {
            'anushtubh': 'SLSL SLSL',  # S=short, L=long
            'trishtubh': 'SLSL SLSL SLSL',
            'jagati': 'SLSL SLSL SLSL SL'
        })
    
    def process(self, data: Any, context: ProcessingContext) -> Any:
        """Process data for meter analysis."""
        if isinstance(data, AdvancedSanskritToken):
            return self._analyze_meter(data, context)
        elif isinstance(data, list):
            return [self.process(item, context) for item in data]
        return data
    
    def _analyze_meter(self, token: AdvancedSanskritToken,
                      context: ProcessingContext) -> AdvancedSanskritToken:
        """Analyze metrical properties of a token."""
        surface_form = token.surface_form
        
        # Calculate syllable weights (simplified)
        syllable_pattern = self._get_syllable_pattern(surface_form)
        
        if syllable_pattern:
            token.token_metadata.prosodic_features['syllable_pattern'] = syllable_pattern
            token.token_metadata.prosodic_features['syllable_count'] = len(syllable_pattern)
            
            # Try to match known meters
            for meter_name, meter_pattern in self._meter_patterns.items():
                if syllable_pattern in meter_pattern:
                    token.token_metadata.prosodic_features['potential_meter'] = meter_name
                    token.token_metadata.confidence_scores['meter_analysis'] = 0.6
                    break
        
        return token
    
    def _get_syllable_pattern(self, word: str) -> str:
        """Get syllable weight pattern (S=short, L=long)."""
        pattern = []
        
        # Simplified syllable analysis
        i = 0
        while i < len(word):
            char = word[i]
            
            if char in 'aeiouāīūṛṝḷḹeo':
                # Check if it's a long vowel
                if char in 'āīūṛṝḷḹ':
                    pattern.append('L')
                else:
                    # Check for consonant cluster after short vowel
                    if i + 1 < len(word) and word[i + 1] not in 'aeiouāīūṛṝḷḹeo':
                        pattern.append('L')  # Short vowel + consonant = long
                    else:
                        pattern.append('S')
            
            i += 1
        
        return ''.join(pattern)


# Plugin development utilities

class PluginTemplate:
    """Template generator for creating new plugins."""
    
    @staticmethod
    def generate_plugin_template(plugin_name: str, author: str, 
                               supported_stages: List[ProcessingStage],
                               output_dir: str) -> None:
        """Generate a plugin template."""
        template_code = f'''"""
{plugin_name} plugin for the Sanskrit Rewrite Engine.

This plugin provides {plugin_name.replace('_', ' ')} functionality.
"""

from sanskrit_rewrite_engine.future_architecture import LinguisticPlugin
from sanskrit_rewrite_engine.interfaces import ProcessingStage, ProcessingContext
from typing import Any, Dict


class {plugin_name.title().replace('_', '')}Plugin(LinguisticPlugin):
    """Plugin for {plugin_name.replace('_', ' ')}."""
    
    PLUGIN_NAME = "{plugin_name}"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "{author}"
    PLUGIN_DESCRIPTION = "{plugin_name.replace('_', ' ').title()} plugin"
    
    def __init__(self):
        super().__init__(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            supported_stages={[f"ProcessingStage.{stage.name}" for stage in supported_stages]}
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        super().initialize(config)
        # Add your initialization code here
    
    def process(self, data: Any, context: ProcessingContext) -> Any:
        """Process data."""
        # Add your processing logic here
        return data
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        super().cleanup()
        # Add your cleanup code here
'''
        
        # Create plugin directory
        plugin_dir = os.path.join(output_dir, plugin_name)
        os.makedirs(plugin_dir, exist_ok=True)
        
        # Write plugin code
        with open(os.path.join(plugin_dir, 'main.py'), 'w', encoding='utf-8') as f:
            f.write(template_code)
        
        # Create plugin metadata
        metadata = {
            'name': plugin_name,
            'version': '1.0.0',
            'author': author,
            'description': f'{plugin_name.replace("_", " ").title()} plugin',
            'supported_stages': [stage.value for stage in supported_stages],
            'entry_point': 'main',
            'dependencies': [],
            'config_schema': {}
        }
        
        with open(os.path.join(plugin_dir, 'plugin.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Plugin template created at: {plugin_dir}")


class PluginValidator:
    """Validator for plugin implementations."""
    
    @staticmethod
    def validate_plugin(plugin: PluginInterface) -> List[str]:
        """Validate a plugin implementation."""
        issues = []
        
        # Check required properties
        if not plugin.plugin_name:
            issues.append("Plugin name is required")
        
        if not plugin.plugin_version:
            issues.append("Plugin version is required")
        
        if not plugin.supported_stages:
            issues.append("Plugin must support at least one processing stage")
        
        # Check methods
        if not hasattr(plugin, 'process') or not callable(getattr(plugin, 'process')):
            issues.append("Plugin must implement process() method")
        
        # Check initialization
        try:
            plugin.initialize({})
        except Exception as e:
            issues.append(f"Plugin initialization failed: {e}")
        
        return issues


# Factory function for creating plugin system

def create_plugin_system(plugin_directories: List[str]) -> PluginLoader:
    """Create a plugin system with specified directories."""
    return PluginLoader(plugin_directories)


def get_example_plugins() -> List[Type[LinguisticPlugin]]:
    """Get list of example plugin classes."""
    return [
        SandhiAnalysisPlugin,
        CompoundAnalysisPlugin,
        MeterAnalysisPlugin
    ]