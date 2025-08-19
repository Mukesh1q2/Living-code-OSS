"""
Advanced performance profiling utilities for Sanskrit Rewrite Engine.

This module provides detailed profiling capabilities including:
- Function-level performance analysis
- Memory usage tracking
- Bottleneck identification
- Performance regression detection
- Optimization recommendations
"""

import cProfile
import pstats
import io
import time
import functools
import threading
import gc
import sys
import os
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json
import logging
import weakref

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProfileData:
    """Performance profile data for a function or code block."""
    name: str
    total_time: float
    call_count: int
    average_time: float
    min_time: float
    max_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    bottleneck_score: float = 0.0
    call_stack: List[str] = field(default_factory=list)
    line_profile: Optional[Dict[int, Tuple[int, float]]] = None  # line_number: (hits, time)


@dataclass
class ProfilingSession:
    """A complete profiling session with multiple profiles."""
    session_id: str
    start_time: float
    end_time: float
    profiles: Dict[str, ProfileData] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AdvancedProfiler:
    """Advanced profiler with multiple profiling backends."""
    
    def __init__(self, enable_line_profiling: bool = False, 
                 enable_memory_profiling: bool = True):
        self.enable_line_profiling = enable_line_profiling and LINE_PROFILER_AVAILABLE
        self.enable_memory_profiling = enable_memory_profiling
        
        # Profiling data
        self.active_profiles = {}
        self.completed_profiles = {}
        self.call_stack = deque(maxlen=100)
        
        # Memory tracking
        if self.enable_memory_profiling:
            tracemalloc.start()
        
        # Line profiler
        if self.enable_line_profiling:
            self.line_profiler = line_profiler.LineProfiler()
        else:
            self.line_profiler = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Advanced profiler initialized (line: {self.enable_line_profiling}, "
                   f"memory: {self.enable_memory_profiling})")
    
    def profile_function(self, func_name: Optional[str] = None, 
                        include_line_profile: bool = False):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            # Add to line profiler if requested
            if include_line_profile and self.line_profiler:
                self.line_profiler.add_function(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(name, func, args, kwargs)
            
            return wrapper
        return decorator
    
    def profile_block(self, block_name: str):
        """Context manager for profiling code blocks."""
        return ProfiledBlock(self, block_name)
    
    def _profile_execution(self, name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with profiling."""
        with self._lock:
            # Start profiling
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            start_cpu = self._get_cpu_usage()
            
            # Track call stack
            self.call_stack.append(name)
            
            # Initialize profile data
            if name not in self.active_profiles:
                self.active_profiles[name] = {
                    'call_count': 0,
                    'total_time': 0.0,
                    'times': [],
                    'memory_samples': [],
                    'cpu_samples': []
                }
            
            profile = self.active_profiles[name]
            profile['call_count'] += 1
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # End profiling
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self._lock:
                # Update profile data
                profile['total_time'] += execution_time
                profile['times'].append(execution_time)
                profile['memory_samples'].append(memory_delta)
                profile['cpu_samples'].append(end_cpu - start_cpu)
                
                # Remove from call stack
                if self.call_stack and self.call_stack[-1] == name:
                    self.call_stack.pop()
            
            return result
            
        except Exception as e:
            # Clean up on exception
            with self._lock:
                if self.call_stack and self.call_stack[-1] == name:
                    self.call_stack.pop()
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.enable_memory_profiling and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024  # Convert to MB
        elif PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=None)
            except Exception:
                pass
        return 0.0
    
    def finalize_profiles(self) -> Dict[str, ProfileData]:
        """Finalize all active profiles and return profile data."""
        with self._lock:
            finalized_profiles = {}
            
            for name, profile in self.active_profiles.items():
                if profile['call_count'] > 0:
                    times = profile['times']
                    memory_samples = profile['memory_samples']
                    cpu_samples = profile['cpu_samples']
                    
                    # Calculate statistics
                    total_time = profile['total_time']
                    call_count = profile['call_count']
                    average_time = total_time / call_count
                    min_time = min(times) if times else 0.0
                    max_time = max(times) if times else 0.0
                    
                    # Memory statistics
                    avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0.0
                    peak_memory = max(memory_samples) if memory_samples else 0.0
                    
                    # CPU statistics
                    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
                    
                    # Calculate bottleneck score
                    bottleneck_score = self._calculate_bottleneck_score(
                        total_time, call_count, average_time, peak_memory
                    )
                    
                    # Get line profile if available
                    line_profile = self._get_line_profile(name)
                    
                    finalized_profiles[name] = ProfileData(
                        name=name,
                        total_time=total_time,
                        call_count=call_count,
                        average_time=average_time,
                        min_time=min_time,
                        max_time=max_time,
                        memory_usage_mb=avg_memory,
                        peak_memory_mb=peak_memory,
                        cpu_percent=avg_cpu,
                        bottleneck_score=bottleneck_score,
                        call_stack=list(self.call_stack),
                        line_profile=line_profile
                    )
            
            # Store completed profiles
            self.completed_profiles.update(finalized_profiles)
            
            # Clear active profiles
            self.active_profiles.clear()
            
            return finalized_profiles
    
    def _calculate_bottleneck_score(self, total_time: float, call_count: int, 
                                  average_time: float, peak_memory: float) -> float:
        """Calculate bottleneck score for a function."""
        # Weighted score based on multiple factors
        time_weight = 0.4
        frequency_weight = 0.3
        memory_weight = 0.2
        latency_weight = 0.1
        
        # Normalize factors (these thresholds may need tuning)
        time_score = min(total_time / 1.0, 1.0)  # 1 second threshold
        frequency_score = min(call_count / 1000, 1.0)  # 1000 calls threshold
        memory_score = min(peak_memory / 100, 1.0)  # 100 MB threshold
        latency_score = min(average_time / 0.1, 1.0)  # 0.1 second threshold
        
        bottleneck_score = (
            time_weight * time_score +
            frequency_weight * frequency_score +
            memory_weight * memory_score +
            latency_weight * latency_score
        )
        
        return bottleneck_score
    
    def _get_line_profile(self, func_name: str) -> Optional[Dict[int, Tuple[int, float]]]:
        """Get line-by-line profile data if available."""
        if not self.line_profiler:
            return None
        
        try:
            # This is a simplified version - actual implementation would need
            # to extract line profile data from line_profiler
            return None
        except Exception:
            return None
    
    def identify_bottlenecks(self, threshold: float = 0.5) -> List[str]:
        """Identify performance bottlenecks based on profile data."""
        bottlenecks = []
        
        for name, profile in self.completed_profiles.items():
            if profile.bottleneck_score >= threshold:
                bottlenecks.append(
                    f"{name}: score {profile.bottleneck_score:.2f} "
                    f"(time: {profile.total_time:.4f}s, calls: {profile.call_count})"
                )
        
        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: float(x.split("score ")[1].split(" ")[0]), reverse=True)
        
        return bottlenecks
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profile data."""
        recommendations = []
        
        for name, profile in self.completed_profiles.items():
            # High frequency, low individual time -> consider caching
            if profile.call_count > 100 and profile.average_time < 0.01:
                recommendations.append(
                    f"Consider caching results for {name} "
                    f"(called {profile.call_count} times)"
                )
            
            # High individual time -> optimize algorithm
            elif profile.average_time > 0.1:
                recommendations.append(
                    f"Optimize algorithm for {name} "
                    f"(avg time: {profile.average_time:.4f}s)"
                )
            
            # High memory usage -> optimize memory
            elif profile.peak_memory_mb > 50:
                recommendations.append(
                    f"Optimize memory usage for {name} "
                    f"(peak: {profile.peak_memory_mb:.1f}MB)"
                )
            
            # High total time -> major bottleneck
            elif profile.total_time > 1.0:
                recommendations.append(
                    f"Major bottleneck: {name} "
                    f"(total time: {profile.total_time:.4f}s)"
                )
        
        return recommendations
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for profiling context."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': os.cpu_count()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                    'available_memory_gb': psutil.virtual_memory().available / 1024**3,
                    'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                })
            except Exception:
                pass
        
        return info
    
    def create_session(self, session_id: str) -> ProfilingSession:
        """Create a new profiling session."""
        profiles = self.finalize_profiles()
        bottlenecks = self.identify_bottlenecks()
        recommendations = self.generate_recommendations()
        
        session = ProfilingSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=time.time(),
            profiles=profiles,
            system_info=self.get_system_info(),
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
        
        return session
    
    def save_session(self, session: ProfilingSession, output_file: str) -> None:
        """Save profiling session to file."""
        # Convert ProfileData objects to dictionaries for JSON serialization
        session_data = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'system_info': session.system_info,
            'bottlenecks': session.bottlenecks,
            'recommendations': session.recommendations,
            'profiles': {
                name: {
                    'name': profile.name,
                    'total_time': profile.total_time,
                    'call_count': profile.call_count,
                    'average_time': profile.average_time,
                    'min_time': profile.min_time,
                    'max_time': profile.max_time,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'peak_memory_mb': profile.peak_memory_mb,
                    'cpu_percent': profile.cpu_percent,
                    'bottleneck_score': profile.bottleneck_score,
                    'call_stack': profile.call_stack,
                    'line_profile': profile.line_profile
                }
                for name, profile in session.profiles.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Profiling session saved: {output_file}")
    
    def cleanup(self) -> None:
        """Cleanup profiler resources."""
        if self.enable_memory_profiling and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        self.active_profiles.clear()
        self.completed_profiles.clear()
        self.call_stack.clear()


class ProfiledBlock:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: AdvancedProfiler, block_name: str):
        self.profiler = profiler
        self.block_name = block_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.perf_counter()
            execution_time = end_time - self.start_time
            
            # Record the block execution
            with self.profiler._lock:
                if self.block_name not in self.profiler.active_profiles:
                    self.profiler.active_profiles[self.block_name] = {
                        'call_count': 0,
                        'total_time': 0.0,
                        'times': [],
                        'memory_samples': [],
                        'cpu_samples': []
                    }
                
                profile = self.profiler.active_profiles[self.block_name]
                profile['call_count'] += 1
                profile['total_time'] += execution_time
                profile['times'].append(execution_time)


class PerformanceRegression:
    """Performance regression detection and analysis."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_data = {}
        
        if baseline_file and Path(baseline_file).exists():
            self._load_baseline()
    
    def _load_baseline(self) -> None:
        """Load baseline performance data."""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            logger.info(f"Loaded baseline data from {self.baseline_file}")
        except Exception as e:
            logger.warning(f"Failed to load baseline data: {e}")
    
    def save_baseline(self, session: ProfilingSession) -> None:
        """Save current session as baseline."""
        if not self.baseline_file:
            return
        
        baseline_data = {
            'timestamp': session.end_time,
            'system_info': session.system_info,
            'profiles': {
                name: {
                    'total_time': profile.total_time,
                    'call_count': profile.call_count,
                    'average_time': profile.average_time,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'peak_memory_mb': profile.peak_memory_mb
                }
                for name, profile in session.profiles.items()
            }
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.baseline_data = baseline_data
        logger.info(f"Baseline saved to {self.baseline_file}")
    
    def detect_regressions(self, current_session: ProfilingSession, 
                          threshold: float = 0.1) -> List[str]:
        """Detect performance regressions compared to baseline."""
        if not self.baseline_data or 'profiles' not in self.baseline_data:
            return ["No baseline data available for regression detection"]
        
        regressions = []
        baseline_profiles = self.baseline_data['profiles']
        
        for name, current_profile in current_session.profiles.items():
            if name not in baseline_profiles:
                continue
            
            baseline_profile = baseline_profiles[name]
            
            # Check processing time regression
            baseline_time = baseline_profile.get('average_time', 0)
            current_time = current_profile.average_time
            
            if baseline_time > 0:
                time_regression = (current_time - baseline_time) / baseline_time
                if time_regression > threshold:
                    regressions.append(
                        f"{name}: {time_regression:.1%} slower "
                        f"({baseline_time:.4f}s -> {current_time:.4f}s)"
                    )
            
            # Check memory regression
            baseline_memory = baseline_profile.get('peak_memory_mb', 0)
            current_memory = current_profile.peak_memory_mb
            
            if baseline_memory > 0:
                memory_regression = (current_memory - baseline_memory) / baseline_memory
                if memory_regression > threshold:
                    regressions.append(
                        f"{name}: {memory_regression:.1%} more memory "
                        f"({baseline_memory:.1f}MB -> {current_memory:.1f}MB)"
                    )
        
        return regressions


def create_profiler(enable_line_profiling: bool = False, 
                   enable_memory_profiling: bool = True) -> AdvancedProfiler:
    """Create an advanced profiler instance."""
    return AdvancedProfiler(enable_line_profiling, enable_memory_profiling)


# Convenience decorators
def profile(func_name: Optional[str] = None, include_line_profile: bool = False):
    """Convenience decorator for profiling functions."""
    _default_profiler = create_profiler()
    return _default_profiler.profile_function(func_name, include_line_profile)


def profile_block(block_name: str):
    """Convenience context manager for profiling code blocks."""
    _default_profiler = create_profiler()
    return _default_profiler.profile_block(block_name)


if __name__ == "__main__":
    # Example usage
    profiler = create_profiler(enable_line_profiling=True, enable_memory_profiling=True)
    
    @profiler.profile_function("example_function")
    def example_function(n: int) -> int:
        """Example function for profiling."""
        result = 0
        for i in range(n):
            result += i * i
        return result
    
    # Run example
    for i in range(10):
        example_function(1000)
    
    # Create session and analyze
    session = profiler.create_session("example_session")
    
    print("Bottlenecks:")
    for bottleneck in session.bottlenecks:
        print(f"  {bottleneck}")
    
    print("\nRecommendations:")
    for recommendation in session.recommendations:
        print(f"  {recommendation}")
    
    # Save session
    profiler.save_session(session, "example_profile.json")
    
    # Cleanup
    profiler.cleanup()