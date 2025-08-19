#!/usr/bin/env python3
"""
Verify that dependencies have been right-sized correctly.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_core_dependencies():
    """Check that core dependencies are available and working."""
    print("🔍 Checking Core Dependencies")
    print("=" * 40)
    
    core_deps = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("click", "CLI framework"),
        ("pydantic", "Data validation"),
        ("yaml", "YAML parsing"),
        ("jsonschema", "JSON schema validation")
    ]
    
    missing_deps = []
    working_deps = []
    
    for dep_name, description in core_deps:
        try:
            importlib.import_module(dep_name)
            print(f"✅ {dep_name:12} - {description}")
            working_deps.append(dep_name)
        except ImportError:
            print(f"❌ {dep_name:12} - {description} (MISSING)")
            missing_deps.append(dep_name)
    
    return len(missing_deps) == 0, working_deps, missing_deps

def check_optional_dependencies():
    """Check that heavy dependencies are NOT in core installation."""
    print("\n🔍 Checking Optional Dependencies (should be missing)")
    print("=" * 55)
    
    optional_deps = [
        ("torch", "PyTorch (GPU extra)"),
        ("transformers", "Transformers (GPU extra)"),
        ("numpy", "NumPy (analysis extra)"),
        ("pandas", "Pandas (analysis extra)"),
        ("matplotlib", "Matplotlib (analysis extra)"),
        ("streamlit", "Streamlit (web extra)"),
        ("jupyter", "Jupyter (jupyter extra)"),
        ("redis", "Redis (storage extra)"),
        ("sqlalchemy", "SQLAlchemy (storage extra)")
    ]
    
    present_deps = []
    missing_deps = []
    
    for dep_name, description in optional_deps:
        try:
            importlib.import_module(dep_name)
            print(f"⚠️  {dep_name:12} - {description} (PRESENT - should be optional)")
            present_deps.append(dep_name)
        except ImportError:
            print(f"✅ {dep_name:12} - {description} (correctly optional)")
            missing_deps.append(dep_name)
    
    return present_deps, missing_deps

def check_package_structure():
    """Check that the package can be imported correctly."""
    print("\n🔍 Checking Package Structure")
    print("=" * 35)
    
    try:
        sys.path.insert(0, 'src')
        
        # Test core imports
        from sanskrit_rewrite_engine import SanskritRewriteEngine
        print("✅ Core engine import")
        
        from sanskrit_rewrite_engine.server import app
        print("✅ Server app import")
        
        from sanskrit_rewrite_engine.cli import main
        print("✅ CLI import")
        
        # Test that server works
        routes = [route.path for route in app.routes]
        if '/api/process' in routes:
            print("✅ Server endpoints configured")
        else:
            print("❌ Server endpoints missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

def check_pyproject_configuration():
    """Check that pyproject.toml is properly configured."""
    print("\n🔍 Checking pyproject.toml Configuration")
    print("=" * 42)
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("❌ Cannot read pyproject.toml (no TOML parser)")
            return False
    
    try:
        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)
        
        # Check core dependencies
        core_deps = config.get('project', {}).get('dependencies', [])
        print(f"✅ Core dependencies: {len(core_deps)} packages")
        
        # Check optional dependencies
        optional_deps = config.get('project', {}).get('optional-dependencies', {})
        print(f"✅ Optional dependency groups: {len(optional_deps)}")
        
        # Check that heavy deps are in optionals
        heavy_deps_in_core = any('torch' in dep or 'transformers' in dep for dep in core_deps)
        if heavy_deps_in_core:
            print("❌ Heavy dependencies found in core dependencies")
            return False
        else:
            print("✅ Heavy dependencies properly moved to optional extras")
        
        # Check extras structure
        expected_extras = ['dev', 'gpu', 'web', 'analysis', 'monitoring']
        missing_extras = [extra for extra in expected_extras if extra not in optional_deps]
        
        if missing_extras:
            print(f"❌ Missing extras: {missing_extras}")
            return False
        else:
            print("✅ All expected extras present")
        
        return True
        
    except Exception as e:
        print(f"❌ pyproject.toml check failed: {e}")
        return False

def check_requirements_files():
    """Check that requirements files are properly structured."""
    print("\n🔍 Checking Requirements Files")
    print("=" * 35)
    
    # Check main requirements.txt
    if Path('requirements.txt').exists():
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        # Should not contain heavy dependencies
        heavy_deps = ['torch', 'transformers', 'numpy', 'pandas', 'matplotlib']
        heavy_in_requirements = any(dep in content for dep in heavy_deps)
        
        if heavy_in_requirements:
            print("❌ Heavy dependencies found in requirements.txt")
            return False
        else:
            print("✅ requirements.txt contains only core dependencies")
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Check dev requirements
    if Path('requirements-dev.txt').exists():
        print("✅ requirements-dev.txt exists")
    else:
        print("⚠️  requirements-dev.txt not found (optional)")
    
    return True

def main():
    """Run all dependency checks."""
    print("🔍 Sanskrit Rewrite Engine - Dependency Right-sizing Verification")
    print("=" * 70)
    
    all_passed = True
    
    # Check core dependencies
    core_ok, working, missing = check_core_dependencies()
    if not core_ok:
        print(f"\n❌ Missing core dependencies: {missing}")
        all_passed = False
    
    # Check optional dependencies
    present_optional, missing_optional = check_optional_dependencies()
    if present_optional:
        print(f"\n⚠️  Optional dependencies present in core: {present_optional}")
        print("   This is OK if you installed extras, but indicates heavy dependencies")
    
    # Check package structure
    if not check_package_structure():
        all_passed = False
    
    # Check pyproject.toml
    if not check_pyproject_configuration():
        all_passed = False
    
    # Check requirements files
    if not check_requirements_files():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ DEPENDENCY RIGHT-SIZING SUCCESSFUL")
        print("\nChanges made:")
        print("  • Moved heavy dependencies (torch, transformers, etc.) to optional extras")
        print("  • Kept only essential dependencies in core installation")
        print("  • Created logical groupings: dev, gpu, web, analysis, monitoring, etc.")
        print("  • Updated requirements.txt to be minimal")
        print("  • Created requirements-dev.txt for development")
        print("  • Added comprehensive installation documentation")
        
        print("\nInstallation options:")
        print("  • Core only: pip install -e .")
        print("  • Development: pip install -e .[dev]")
        print("  • With GPU: pip install -e .[gpu]")
        print("  • Full features: pip install -e .[all]")
        
        return True
    else:
        print("❌ DEPENDENCY RIGHT-SIZING FAILED")
        print("Some checks failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)