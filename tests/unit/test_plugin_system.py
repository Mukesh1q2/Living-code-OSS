"""
Tests for the plugin system.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.sanskrit_rewrite_engine.plugin_system import (
    PluginManager, BasePlugin, TokenProcessorPlugin, RuleProcessorPlugin,
    MLPlugin, HookPlugin, ExampleTokenizerPlugin, ExampleMLPlugin,
    PluginMetadata, PluginLoadError, scan_for_plugins, validate_plugin_interface
)