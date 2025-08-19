#!/usr/bin/env python3
"""
Verification script for the Sanskrit Rewrite Engine Frontend
Checks file structure and basic syntax without requiring npm
"""

import os
import json
from pathlib import Path

def check_file_exists(file_path, description=""):
    """Check if a file exists and print status"""
    if file_path.exists():
        print(f"✅ {description or file_path.name}")
        return True
    else:
        print(f"❌ {description or file_path.name}")
        return False

def check_json_syntax(file_path):
    """Check if JSON file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def main():
    """Main verification function"""
    frontend_dir = Path(__file__).parent / "frontend"
    
    print("Verifying Sanskrit Rewrite Engine Frontend Structure...")
    print("=" * 60)
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Check core files
    print("\n📁 Core Files:")
    core_files = [
        (frontend_dir / "package.json", "package.json (dependencies)"),
        (frontend_dir / "tsconfig.json", "tsconfig.json (TypeScript config)"),
        (frontend_dir / "public" / "index.html", "public/index.html (main HTML)"),
        (frontend_dir / "src" / "index.tsx", "src/index.tsx (entry point)"),
        (frontend_dir / "src" / "App.tsx", "src/App.tsx (main component)"),
    ]
    
    all_core_exist = True
    for file_path, description in core_files:
        if not check_file_exists(file_path, description):
            all_core_exist = False
    
    # Check JSON syntax
    print("\n🔍 JSON Syntax:")
    json_files = [
        frontend_dir / "package.json",
        frontend_dir / "tsconfig.json",
        frontend_dir / "public" / "manifest.json",
    ]
    
    json_valid = True
    for json_file in json_files:
        if json_file.exists():
            if check_json_syntax(json_file):
                print(f"✅ {json_file.name} syntax valid")
            else:
                json_valid = False
        else:
            print(f"⚠️ {json_file.name} not found")
    
    # Check component structure
    print("\n🧩 Components:")
    components_dir = frontend_dir / "src" / "components"
    component_files = [
        "Header.tsx",
        "ChatInterface.tsx", 
        "CodeEditor.tsx",
        "DiagramCanvas.tsx",
        "MessageBubble.tsx",
        "MessageInput.tsx",
        "MessageList.tsx",
        "ConversationList.tsx",
    ]
    
    components_exist = True
    for component in component_files:
        component_path = components_dir / component
        if not check_file_exists(component_path, f"components/{component}"):
            components_exist = False
    
    # Check CSS files
    print("\n🎨 Styles:")
    css_files = [
        "src/index.css",
        "src/App.css",
        "src/components/Header.css",
        "src/components/ChatInterface.css",
        "src/components/CodeEditor.css",
        "src/components/DiagramCanvas.css",
    ]
    
    styles_exist = True
    for css_file in css_files:
        css_path = frontend_dir / css_file
        if not check_file_exists(css_path, css_file):
            styles_exist = False
    
    # Check contexts
    print("\n🔄 Contexts:")
    contexts_dir = frontend_dir / "src" / "contexts"
    context_files = [
        "ChatContext.tsx",
        "WebSocketContext.tsx",
    ]
    
    contexts_exist = True
    for context in context_files:
        context_path = contexts_dir / context
        if not check_file_exists(context_path, f"contexts/{context}"):
            contexts_exist = False
    
    # Check services
    print("\n🌐 Services:")
    services_dir = frontend_dir / "src" / "services"
    service_files = [
        "apiService.ts",
    ]
    
    services_exist = True
    for service in service_files:
        service_path = services_dir / service
        if not check_file_exists(service_path, f"services/{service}"):
            services_exist = False
    
    # Check tests
    print("\n🧪 Tests:")
    test_files = [
        "src/App.test.tsx",
        "src/components/__tests__/ChatInterface.test.tsx",
        "src/components/__tests__/CodeEditor.test.tsx", 
        "src/components/__tests__/DiagramCanvas.test.tsx",
    ]
    
    tests_exist = True
    for test_file in test_files:
        test_path = frontend_dir / test_file
        if not check_file_exists(test_path, test_file):
            tests_exist = False
    
    # Check package.json dependencies
    print("\n📦 Dependencies Check:")
    package_json_path = frontend_dir / "package.json"
    if package_json_path.exists():
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            required_deps = [
                "react", "react-dom", "typescript", "@monaco-editor/react",
                "axios", "d3", "socket.io-client", "react-router-dom"
            ]
            
            dependencies = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
            
            deps_ok = True
            for dep in required_deps:
                if dep in dependencies:
                    print(f"✅ {dep}")
                else:
                    print(f"❌ {dep} (missing)")
                    deps_ok = False
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            deps_ok = False
    else:
        deps_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY:")
    print(f"Core Files: {'✅ PASS' if all_core_exist else '❌ FAIL'}")
    print(f"JSON Syntax: {'✅ PASS' if json_valid else '❌ FAIL'}")
    print(f"Components: {'✅ PASS' if components_exist else '❌ FAIL'}")
    print(f"Styles: {'✅ PASS' if styles_exist else '❌ FAIL'}")
    print(f"Contexts: {'✅ PASS' if contexts_exist else '❌ FAIL'}")
    print(f"Services: {'✅ PASS' if services_exist else '❌ FAIL'}")
    print(f"Tests: {'✅ PASS' if tests_exist else '❌ FAIL'}")
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    
    overall_pass = all([
        all_core_exist, json_valid, components_exist, styles_exist,
        contexts_exist, services_exist, tests_exist, deps_ok
    ])
    
    print(f"\n🎯 OVERALL: {'✅ PASS - Frontend is ready!' if overall_pass else '❌ FAIL - Issues found'}")
    
    if overall_pass:
        print("\n🚀 Next Steps:")
        print("1. Install dependencies: cd frontend && npm install")
        print("2. Start development server: npm start")
        print("3. Run tests: npm test")
        print("4. Build for production: npm run build")
    
    return overall_pass

if __name__ == "__main__":
    main()