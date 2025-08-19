"""
Environment configuration management for Vidya Quantum Interface
"""
import os
from enum import Enum
from typing import Optional, List
from pydantic import BaseSettings, Field


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class BaseConfig(BaseSettings):
    """Base configuration class"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # Database
    database_url: str = Field(default="sqlite:///./vidya.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # AI Services
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Local AI
    local_model_path: str = Field(default="./models", env="LOCAL_MODEL_PATH")
    enable_local_inference: bool = Field(default=True, env="ENABLE_LOCAL_INFERENCE")
    
    # Sanskrit Engine
    sanskrit_data_path: str = Field(default="./data", env="SANSKRIT_DATA_PATH")
    sutra_rules_path: str = Field(default="./sutra_rules", env="SUTRA_RULES_PATH")
    
    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9000, env="METRICS_PORT")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Cloud Configuration
    cloud_provider: CloudProvider = Field(default=CloudProvider.LOCAL, env="CLOUD_PROVIDER")
    
    # Feature Flags
    enable_quantum_effects: bool = Field(default=True, env="ENABLE_QUANTUM_EFFECTS")
    enable_neural_network: bool = Field(default=True, env="ENABLE_NEURAL_NETWORK")
    enable_voice_interaction: bool = Field(default=False, env="ENABLE_VOICE_INTERACTION")
    enable_vr_mode: bool = Field(default=False, env="ENABLE_VR_MODE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    debug: bool = True
    log_level: str = "debug"
    enable_tracing: bool = True
    
    # Development-specific overrides
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]


class StagingConfig(BaseConfig):
    """Staging environment configuration"""
    
    debug: bool = False
    log_level: str = "info"
    enable_tracing: bool = True
    
    # Staging-specific settings
    cache_ttl: int = 1800  # 30 minutes
    max_workers: int = 2


class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    debug: bool = False
    log_level: str = "warning"
    enable_tracing: bool = False
    
    # Production-specific settings
    cache_ttl: int = 7200  # 2 hours
    max_workers: int = 8
    
    # Security enhancements
    cors_origins: List[str] = []  # Should be set via environment variable
    
    # Performance optimizations
    enable_caching: bool = True


class AWSConfig(BaseConfig):
    """AWS-specific configuration"""
    
    aws_region: str = Field(default="us-west-2", env="AWS_REGION")
    aws_s3_bucket: str = Field(default="vidya-storage", env="AWS_S3_BUCKET")
    aws_cloudwatch_log_group: str = Field(
        default="/aws/lambda/vidya", 
        env="AWS_CLOUDWATCH_LOG_GROUP"
    )
    
    # AWS-specific overrides
    database_url: str = Field(env="RDS_DATABASE_URL")
    redis_url: str = Field(env="ELASTICACHE_REDIS_URL")


class KubernetesConfig(BaseConfig):
    """Kubernetes-specific configuration"""
    
    kubernetes_namespace: str = Field(default="vidya", env="KUBERNETES_NAMESPACE")
    kubernetes_service_account: str = Field(
        default="vidya-service-account", 
        env="KUBERNETES_SERVICE_ACCOUNT"
    )
    
    # Kubernetes-specific overrides
    host: str = "0.0.0.0"
    port: int = 8000


def get_config() -> BaseConfig:
    """Get configuration based on environment"""
    
    env = os.getenv("ENVIRONMENT", "development").lower()
    cloud_provider = os.getenv("CLOUD_PROVIDER", "local").lower()
    
    if env == "development":
        return DevelopmentConfig()
    elif env == "staging":
        if cloud_provider == "aws":
            return AWSConfig(environment=Environment.STAGING)
        return StagingConfig()
    elif env == "production":
        if cloud_provider == "aws":
            return AWSConfig(environment=Environment.PRODUCTION)
        elif os.getenv("KUBERNETES_NAMESPACE"):
            return KubernetesConfig(environment=Environment.PRODUCTION)
        return ProductionConfig()
    else:
        return BaseConfig()


# Global configuration instance
config = get_config()