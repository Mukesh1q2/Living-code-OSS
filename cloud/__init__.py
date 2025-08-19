"""
Cloud integration module for Vidya Quantum Interface
"""

from .aws_integration import aws_integration, scalable_processor, AWSIntegration, ScalableAIProcessor

__all__ = ["aws_integration", "scalable_processor", "AWSIntegration", "ScalableAIProcessor"]