"""
Deployment Automation and CI/CD Pipeline for Sanskrit Rewrite Engine

This module provides comprehensive deployment automation, CI/CD pipelines,
containerization, and infrastructure management for the Sanskrit reasoning system.

Requirements: All requirements final validation - deployment automation and CI/CD
"""

import os
import sys
import json
import yaml
import asyncio
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import tarfile
import zipfile
import logging

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    version: str
    docker_registry: str = "localhost:5000"
    kubernetes_namespace: str = "sanskrit-engine"
    replicas: int = 3
    cpu_limit: str = "2000m"
    memory_limit: str = "4Gi"
    gpu_enabled: bool = True
    auto_scaling: bool = True
    health_check_path: str = "/health"
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class BuildArtifact:
    """Build artifact information."""
    name: str
    version: str
    path: Path
    size_bytes: int
    checksum: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentResult:
    """Deployment operation result."""
    deployment_id: str
    environment: DeploymentEnvironment
    version: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    artifacts: List[BuildArtifact] = field(default_factory=list)
    rollback_version: Optional[str] = None
    error_message: Optional[str] = None

class DockerBuilder:
    """Docker container builder for the Sanskrit engine."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.dockerfile_template = self._get_dockerfile_template()
    
    def _get_dockerfile_template(self) -> str:
        """Get Dockerfile template for Sanskrit engine."""
        return """
# Multi-stage build for Sanskrit Rewrite Engine
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY sanskrit_rewrite_engine/ ./sanskrit_rewrite_engine/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY *.py ./
COPY *.md ./

# Create non-root user
RUN useradd -m -u 1000 sanskrit && \\
    chown -R sanskrit:sanskrit /app
USER sanskrit

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "sanskrit_rewrite_engine.api_server"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY sanskrit_rewrite_engine/ ./sanskrit_rewrite_engine/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY *.py ./
COPY *.md ./

# Create user
RUN useradd -m -u 1000 sanskrit && \\
    chown -R sanskrit:sanskrit /app
USER sanskrit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "sanskrit_rewrite_engine.api_server"]
"""
    
    async def build_image(self, config: DeploymentConfig, gpu_enabled: bool = False) -> BuildArtifact:
        """Build Docker image."""
        logger.info(f"Building Docker image for {config.environment.value} environment")
        
        # Create build context
        build_context = self._create_build_context(config)
        
        try:
            # Generate Dockerfile
            dockerfile_path = build_context / "Dockerfile"
            dockerfile_content = self.dockerfile_template
            
            # Modify for GPU if needed
            if gpu_enabled:
                dockerfile_content += "\n# Enable GPU support\nENV NVIDIA_VISIBLE_DEVICES=all"
            
            dockerfile_path.write_text(dockerfile_content)
            
            # Build image
            image_name = f"sanskrit-engine:{config.version}"
            if config.docker_registry:
                image_name = f"{config.docker_registry}/{image_name}"
            
            build_cmd = [
                "docker", "build",
                "-t", image_name,
                "--target", "gpu" if gpu_enabled else "production",
                str(build_context)
            ]
            
            result = await self._run_command(build_cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"Docker build failed: {result.stderr}")
            
            # Get image info
            inspect_cmd = ["docker", "inspect", image_name]
            inspect_result = await self._run_command(inspect_cmd)
            
            if inspect_result.returncode == 0:
                image_info = json.loads(inspect_result.stdout)[0]
                size_bytes = image_info.get("Size", 0)
                image_id = image_info.get("Id", "")
            else:
                size_bytes = 0
                image_id = ""
            
            # Create artifact
            artifact = BuildArtifact(
                name=image_name,
                version=config.version,
                path=Path(f"docker://{image_name}"),
                size_bytes=size_bytes,
                checksum=image_id,
                created_at=datetime.now(),
                metadata={
                    "type": "docker_image",
                    "gpu_enabled": gpu_enabled,
                    "registry": config.docker_registry
                }
            )
            
            logger.info(f"Docker image built successfully: {image_name}")
            return artifact
            
        finally:
            # Cleanup build context
            shutil.rmtree(build_context, ignore_errors=True)
    
    def _create_build_context(self, config: DeploymentConfig) -> Path:
        """Create Docker build context."""
        build_context = Path(tempfile.mkdtemp(prefix="sanskrit_build_"))
        
        # Copy source files
        source_dirs = [
            "sanskrit_rewrite_engine",
            "examples",
            "tests"
        ]
        
        for dir_name in source_dirs:
            source_dir = self.base_directory / dir_name
            if source_dir.exists():
                shutil.copytree(source_dir, build_context / dir_name)
        
        # Copy root files
        root_files = [
            "requirements.txt",
            "README.md",
            "run_tests.py"
        ]
        
        for file_name in root_files:
            source_file = self.base_directory / file_name
            if source_file.exists():
                shutil.copy2(source_file, build_context / file_name)
        
        return build_context
    
    async def push_image(self, artifact: BuildArtifact) -> bool:
        """Push Docker image to registry."""
        if not artifact.metadata.get("registry"):
            logger.warning("No registry configured, skipping push")
            return True
        
        logger.info(f"Pushing image: {artifact.name}")
        
        push_cmd = ["docker", "push", artifact.name]
        result = await self._run_command(push_cmd)
        
        if result.returncode != 0:
            logger.error(f"Failed to push image: {result.stderr}")
            return False
        
        logger.info(f"Image pushed successfully: {artifact.name}")
        return True
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )

class KubernetesDeployer:
    """Kubernetes deployment manager."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.manifest_templates = self._get_manifest_templates()
    
    def _get_manifest_templates(self) -> Dict[str, str]:
        """Get Kubernetes manifest templates."""
        return {
            "namespace": """
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
  labels:
    app: sanskrit-engine
    environment: {environment}
""",
            "deployment": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sanskrit-engine
  namespace: {namespace}
  labels:
    app: sanskrit-engine
    version: {version}
    environment: {environment}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: sanskrit-engine
  template:
    metadata:
      labels:
        app: sanskrit-engine
        version: {version}
    spec:
      containers:
      - name: sanskrit-engine
        image: {image}
        ports:
        - containerPort: 8000
        env:
{environment_variables}
        resources:
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
{gpu_resources}
        livenessProbe:
          httpGet:
            path: {health_check_path}
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: {health_check_path}
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
{volumes}
""",
            "service": """
apiVersion: v1
kind: Service
metadata:
  name: sanskrit-engine-service
  namespace: {namespace}
  labels:
    app: sanskrit-engine
spec:
  selector:
    app: sanskrit-engine
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
""",
            "hpa": """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sanskrit-engine-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sanskrit-engine
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        }
    
    async def deploy(self, config: DeploymentConfig, artifact: BuildArtifact) -> DeploymentResult:
        """Deploy to Kubernetes."""
        deployment_id = f"deploy-{config.version}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            environment=config.environment,
            version=config.version,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now(),
            artifacts=[artifact]
        )
        
        try:
            logger.info(f"Starting Kubernetes deployment: {deployment_id}")
            
            # Create manifests
            manifests = self._generate_manifests(config, artifact)
            
            # Apply manifests
            for manifest_name, manifest_content in manifests.items():
                logger.info(f"Applying {manifest_name} manifest")
                
                apply_result = await self._apply_manifest(manifest_content)
                result.logs.append(f"Applied {manifest_name}: {apply_result}")
                
                if not apply_result.startswith("configured") and not apply_result.startswith("created"):
                    raise RuntimeError(f"Failed to apply {manifest_name}: {apply_result}")
            
            # Wait for deployment to be ready
            await self._wait_for_deployment(config.kubernetes_namespace)
            
            result.status = DeploymentStatus.SUCCESS
            result.end_time = datetime.now()
            
            logger.info(f"Deployment completed successfully: {deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            logger.error(f"Deployment failed: {deployment_id} - {e}")
        
        return result
    
    def _generate_manifests(self, config: DeploymentConfig, artifact: BuildArtifact) -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        manifests = {}
        
        # Environment variables
        env_vars = []
        for key, value in config.environment_variables.items():
            env_vars.append(f"        - name: {key}\n          value: \"{value}\"")
        env_vars_str = "\n".join(env_vars) if env_vars else "        []"
        
        # GPU resources
        gpu_resources = ""
        if config.gpu_enabled:
            gpu_resources = """            nvidia.com/gpu: 1"""
        
        # Volumes
        volumes_str = ""
        if config.volumes:
            volume_mounts = []
            volume_specs = []
            
            for volume in config.volumes:
                volume_mounts.append(f"""        volumeMounts:
        - name: {volume['name']}
          mountPath: {volume['mountPath']}""")
                
                volume_specs.append(f"""      volumes:
      - name: {volume['name']}
        {volume['type']}:
          {volume['spec']}""")
            
            volumes_str = "\n".join(volume_mounts + volume_specs)
        
        # Generate each manifest
        template_vars = {
            "namespace": config.kubernetes_namespace,
            "environment": config.environment.value,
            "version": config.version,
            "replicas": config.replicas,
            "image": artifact.name,
            "cpu_limit": config.cpu_limit,
            "memory_limit": config.memory_limit,
            "health_check_path": config.health_check_path,
            "environment_variables": env_vars_str,
            "gpu_resources": gpu_resources,
            "volumes": volumes_str,
            "min_replicas": max(1, config.replicas // 2),
            "max_replicas": config.replicas * 3
        }
        
        for manifest_name, template in self.manifest_templates.items():
            if manifest_name == "hpa" and not config.auto_scaling:
                continue
            
            manifests[manifest_name] = template.format(**template_vars)
        
        return manifests
    
    async def _apply_manifest(self, manifest_content: str) -> str:
        """Apply Kubernetes manifest."""
        # Write manifest to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(manifest_content)
            manifest_file = f.name
        
        try:
            # Apply manifest
            cmd = ["kubectl", "apply", "-f", manifest_file]
            if self.kubeconfig_path:
                cmd.extend(["--kubeconfig", self.kubeconfig_path])
            
            result = await self._run_command(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"kubectl apply failed: {result.stderr}")
            
            return result.stdout.strip()
            
        finally:
            # Cleanup temporary file
            Path(manifest_file).unlink(missing_ok=True)
    
    async def _wait_for_deployment(self, namespace: str, timeout: int = 300):
        """Wait for deployment to be ready."""
        cmd = [
            "kubectl", "rollout", "status",
            f"deployment/sanskrit-engine",
            f"--namespace={namespace}",
            f"--timeout={timeout}s"
        ]
        
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        
        result = await self._run_command(cmd)
        
        if result.returncode != 0:
            raise RuntimeError(f"Deployment rollout failed: {result.stderr}")
    
    async def rollback(self, config: DeploymentConfig, target_version: str) -> DeploymentResult:
        """Rollback deployment to previous version."""
        deployment_id = f"rollback-{target_version}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            environment=config.environment,
            version=config.version,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now(),
            rollback_version=target_version
        )
        
        try:
            logger.info(f"Starting rollback to version {target_version}")
            
            # Rollback deployment
            cmd = [
                "kubectl", "rollout", "undo",
                f"deployment/sanskrit-engine",
                f"--namespace={config.kubernetes_namespace}"
            ]
            
            if self.kubeconfig_path:
                cmd.extend(["--kubeconfig", self.kubeconfig_path])
            
            rollback_result = await self._run_command(cmd)
            
            if rollback_result.returncode != 0:
                raise RuntimeError(f"Rollback failed: {rollback_result.stderr}")
            
            # Wait for rollback to complete
            await self._wait_for_deployment(config.kubernetes_namespace)
            
            result.status = DeploymentStatus.ROLLED_BACK
            result.end_time = datetime.now()
            
            logger.info(f"Rollback completed successfully: {deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            logger.error(f"Rollback failed: {deployment_id} - {e}")
        
        return result
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )

class CIPipeline:
    """Continuous Integration pipeline."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.test_results: List[Dict[str, Any]] = []
    
    async def run_tests(self) -> bool:
        """Run comprehensive test suite."""
        logger.info("Running test suite...")
        
        test_commands = [
            # Unit tests
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            
            # Integration tests
            ["python", "-m", "pytest", "tests/test_comprehensive_integration.py", "-v"],
            
            # Performance tests
            ["python", "tests/benchmark_reasoning_core.py"],
            
            # Security tests
            ["python", "-m", "pytest", "tests/test_mcp_security.py", "-v"],
            
            # Linting
            ["python", "-m", "flake8", "sanskrit_rewrite_engine/", "--max-line-length=120"],
            
            # Type checking
            ["python", "-m", "mypy", "sanskrit_rewrite_engine/", "--ignore-missing-imports"]
        ]
        
        all_passed = True
        
        for cmd in test_commands:
            try:
                result = await self._run_command(cmd)
                
                test_result = {
                    "command": " ".join(cmd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "passed": result.returncode == 0
                }
                
                self.test_results.append(test_result)
                
                if result.returncode != 0:
                    all_passed = False
                    logger.error(f"Test failed: {' '.join(cmd)}")
                    logger.error(f"Error: {result.stderr}")
                else:
                    logger.info(f"Test passed: {' '.join(cmd)}")
                    
            except Exception as e:
                logger.error(f"Test execution failed: {' '.join(cmd)} - {e}")
                all_passed = False
        
        return all_passed
    
    async def run_security_scan(self) -> bool:
        """Run security vulnerability scan."""
        logger.info("Running security scan...")
        
        security_commands = [
            # Dependency vulnerability scan
            ["python", "-m", "safety", "check"],
            
            # Code security scan
            ["python", "-m", "bandit", "-r", "sanskrit_rewrite_engine/", "-f", "json"]
        ]
        
        all_passed = True
        
        for cmd in security_commands:
            try:
                result = await self._run_command(cmd)
                
                if result.returncode != 0:
                    all_passed = False
                    logger.warning(f"Security scan found issues: {' '.join(cmd)}")
                    logger.warning(f"Output: {result.stdout}")
                else:
                    logger.info(f"Security scan passed: {' '.join(cmd)}")
                    
            except Exception as e:
                logger.warning(f"Security scan failed to run: {' '.join(cmd)} - {e}")
        
        return all_passed
    
    async def generate_documentation(self) -> bool:
        """Generate project documentation."""
        logger.info("Generating documentation...")
        
        try:
            # Generate API documentation
            doc_cmd = [
                "python", "-m", "sphinx.cmd.build",
                "-b", "html",
                "docs/source",
                "docs/build/html"
            ]
            
            result = await self._run_command(doc_cmd)
            
            if result.returncode != 0:
                logger.warning(f"Documentation generation had issues: {result.stderr}")
                return False
            
            logger.info("Documentation generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return False
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.base_directory
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )

class DeploymentManager:
    """Main deployment manager coordinating all deployment activities."""
    
    def __init__(self, base_directory: Path):
        self.base_directory = base_directory
        self.docker_builder = DockerBuilder(base_directory)
        self.k8s_deployer = KubernetesDeployer()
        self.ci_pipeline = CIPipeline(base_directory)
        self.deployment_history: List[DeploymentResult] = []
    
    async def full_deployment_pipeline(self, config: DeploymentConfig) -> DeploymentResult:
        """Run complete deployment pipeline."""
        deployment_id = f"pipeline-{config.version}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            environment=config.environment,
            version=config.version,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Starting full deployment pipeline: {deployment_id}")
            
            # Step 1: Run CI pipeline
            if config.environment != DeploymentEnvironment.DEVELOPMENT:
                logger.info("Running CI pipeline...")
                
                tests_passed = await self.ci_pipeline.run_tests()
                if not tests_passed:
                    raise RuntimeError("CI tests failed")
                
                security_passed = await self.ci_pipeline.run_security_scan()
                if not security_passed and config.environment == DeploymentEnvironment.PRODUCTION:
                    raise RuntimeError("Security scan failed for production deployment")
                
                docs_generated = await self.ci_pipeline.generate_documentation()
                if not docs_generated:
                    logger.warning("Documentation generation failed, continuing...")
            
            # Step 2: Build Docker image
            logger.info("Building Docker image...")
            artifact = await self.docker_builder.build_image(config, config.gpu_enabled)
            result.artifacts.append(artifact)
            
            # Step 3: Push image to registry
            if config.docker_registry:
                logger.info("Pushing image to registry...")
                push_success = await self.docker_builder.push_image(artifact)
                if not push_success:
                    raise RuntimeError("Failed to push image to registry")
            
            # Step 4: Deploy to Kubernetes
            logger.info("Deploying to Kubernetes...")
            k8s_result = await self.k8s_deployer.deploy(config, artifact)
            
            if k8s_result.status != DeploymentStatus.SUCCESS:
                raise RuntimeError(f"Kubernetes deployment failed: {k8s_result.error_message}")
            
            result.logs.extend(k8s_result.logs)
            
            # Step 5: Run post-deployment tests
            if config.environment in [DeploymentEnvironment.STAGING, DeploymentEnvironment.PRODUCTION]:
                logger.info("Running post-deployment tests...")
                # Add post-deployment health checks here
                await asyncio.sleep(30)  # Wait for services to stabilize
            
            result.status = DeploymentStatus.SUCCESS
            result.end_time = datetime.now()
            
            logger.info(f"Deployment pipeline completed successfully: {deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            logger.error(f"Deployment pipeline failed: {deployment_id} - {e}")
            
            # Attempt rollback for non-development environments
            if config.environment != DeploymentEnvironment.DEVELOPMENT and self.deployment_history:
                logger.info("Attempting automatic rollback...")
                last_successful = self._get_last_successful_deployment(config.environment)
                if last_successful:
                    rollback_result = await self.k8s_deployer.rollback(config, last_successful.version)
                    if rollback_result.status == DeploymentStatus.ROLLED_BACK:
                        result.rollback_version = last_successful.version
                        logger.info(f"Automatic rollback completed to version {last_successful.version}")
        
        # Store deployment history
        self.deployment_history.append(result)
        
        return result
    
    def _get_last_successful_deployment(self, environment: DeploymentEnvironment) -> Optional[DeploymentResult]:
        """Get the last successful deployment for an environment."""
        for deployment in reversed(self.deployment_history):
            if (deployment.environment == environment and 
                deployment.status == DeploymentStatus.SUCCESS):
                return deployment
        return None
    
    async def rollback_deployment(self, environment: DeploymentEnvironment, 
                                target_version: Optional[str] = None) -> DeploymentResult:
        """Rollback deployment to a specific version or last successful."""
        if not target_version:
            last_successful = self._get_last_successful_deployment(environment)
            if not last_successful:
                raise ValueError(f"No successful deployment found for {environment.value}")
            target_version = last_successful.version
        
        # Create rollback config
        config = DeploymentConfig(
            environment=environment,
            version=target_version,
            kubernetes_namespace=f"sanskrit-engine-{environment.value}"
        )
        
        return await self.k8s_deployer.rollback(config, target_version)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment."""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None
    
    def list_deployments(self, environment: Optional[DeploymentEnvironment] = None) -> List[DeploymentResult]:
        """List deployments, optionally filtered by environment."""
        if environment:
            return [d for d in self.deployment_history if d.environment == environment]
        return self.deployment_history.copy()


# Configuration templates for different environments
DEPLOYMENT_CONFIGS = {
    DeploymentEnvironment.DEVELOPMENT: DeploymentConfig(
        environment=DeploymentEnvironment.DEVELOPMENT,
        version="dev",
        replicas=1,
        cpu_limit="1000m",
        memory_limit="2Gi",
        gpu_enabled=False,
        auto_scaling=False,
        kubernetes_namespace="sanskrit-engine-dev"
    ),
    
    DeploymentEnvironment.TESTING: DeploymentConfig(
        environment=DeploymentEnvironment.TESTING,
        version="test",
        replicas=2,
        cpu_limit="1500m",
        memory_limit="3Gi",
        gpu_enabled=True,
        auto_scaling=False,
        kubernetes_namespace="sanskrit-engine-test"
    ),
    
    DeploymentEnvironment.STAGING: DeploymentConfig(
        environment=DeploymentEnvironment.STAGING,
        version="staging",
        replicas=3,
        cpu_limit="2000m",
        memory_limit="4Gi",
        gpu_enabled=True,
        auto_scaling=True,
        kubernetes_namespace="sanskrit-engine-staging"
    ),
    
    DeploymentEnvironment.PRODUCTION: DeploymentConfig(
        environment=DeploymentEnvironment.PRODUCTION,
        version="latest",
        replicas=5,
        cpu_limit="2000m",
        memory_limit="4Gi",
        gpu_enabled=True,
        auto_scaling=True,
        kubernetes_namespace="sanskrit-engine-prod"
    )
}


async def deploy_to_environment(environment: DeploymentEnvironment, 
                              version: str,
                              base_directory: Optional[Path] = None) -> DeploymentResult:
    """Deploy Sanskrit engine to specified environment."""
    if base_directory is None:
        base_directory = Path.cwd()
    
    # Get base config and update version
    config = DEPLOYMENT_CONFIGS[environment]
    config.version = version
    
    # Create deployment manager and run pipeline
    manager = DeploymentManager(base_directory)
    return await manager.full_deployment_pipeline(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Sanskrit Rewrite Engine")
    parser.add_argument("environment", choices=["dev", "test", "staging", "prod"],
                       help="Deployment environment")
    parser.add_argument("version", help="Version to deploy")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(),
                       help="Base directory of the project")
    
    args = parser.parse_args()
    
    # Map environment names
    env_map = {
        "dev": DeploymentEnvironment.DEVELOPMENT,
        "test": DeploymentEnvironment.TESTING,
        "staging": DeploymentEnvironment.STAGING,
        "prod": DeploymentEnvironment.PRODUCTION
    }
    
    environment = env_map[args.environment]
    
    # Run deployment
    async def main():
        try:
            result = await deploy_to_environment(environment, args.version, args.base_dir)
            
            if result.status == DeploymentStatus.SUCCESS:
                print(f"✅ Deployment successful: {result.deployment_id}")
                print(f"Environment: {result.environment.value}")
                print(f"Version: {result.version}")
                print(f"Duration: {result.end_time - result.start_time}")
            else:
                print(f"❌ Deployment failed: {result.deployment_id}")
                print(f"Error: {result.error_message}")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Deployment error: {e}")
            sys.exit(1)
    
    asyncio.run(main())ment_history.append(result)
        
        return result
    
    def _get_last_successful_deployment(self, environment: DeploymentEnvironment) -> Optional[DeploymentResult]:
        """Get the last successful deployment for an environment."""
        for deployment in reversed(self.deployment_history):
            if (deployment.environment == environment and 
                deployment.status == DeploymentStatus.SUCCESS):
                return deployment
        return None
    
    async def create_deployment_config(self, environment: str, version: str, **kwargs) -> DeploymentConfig:
        """Create deployment configuration for environment."""
        env = DeploymentEnvironment(environment.lower())
        
        # Environment-specific defaults
        defaults = {
            DeploymentEnvironment.DEVELOPMENT: {
                "replicas": 1,
                "cpu_limit": "500m",
                "memory_limit": "1Gi",
                "auto_scaling": False,
                "gpu_enabled": False
            },
            DeploymentEnvironment.TESTING: {
                "replicas": 2,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "auto_scaling": False,
                "gpu_enabled": False
            },
            DeploymentEnvironment.STAGING: {
                "replicas": 2,
                "cpu_limit": "1500m",
                "memory_limit": "3Gi",
                "auto_scaling": True,
                "gpu_enabled": True
            },
            DeploymentEnvironment.PRODUCTION: {
                "replicas": 3,
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "auto_scaling": True,
                "gpu_enabled": True
            }
        }
        
        config_dict = defaults[env].copy()
        config_dict.update(kwargs)
        config_dict["environment"] = env
        config_dict["version"] = version
        
        return DeploymentConfig(**config_dict)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status by ID."""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None
    
    def get_deployment_history(self, environment: Optional[DeploymentEnvironment] = None) -> List[DeploymentResult]:
        """Get deployment history, optionally filtered by environment."""
        if environment:
            return [d for d in self.deployment_history if d.environment == environment]
        return self.deployment_history.copy()

# Global deployment manager
_deployment_manager: Optional[DeploymentManager] = None

def get_deployment_manager(base_directory: Optional[Path] = None) -> DeploymentManager:
    """Get or create the global deployment manager."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager(base_directory or Path.cwd())
    return _deployment_manager

# Convenience functions
async def deploy_to_environment(environment: str, version: str, **kwargs) -> DeploymentResult:
    """Deploy to specified environment."""
    manager = get_deployment_manager()
    config = await manager.create_deployment_config(environment, version, **kwargs)
    return await manager.full_deployment_pipeline(config)

async def rollback_deployment(environment: str, target_version: str) -> DeploymentResult:
    """Rollback deployment to target version."""
    manager = get_deployment_manager()
    env = DeploymentEnvironment(environment.lower())
    
    # Create minimal config for rollback
    config = DeploymentConfig(environment=env, version=target_version)
    
    return await manager.k8s_deployer.rollback(config, target_version)