#!/usr/bin/env python3
"""
TanukiMCP Vision Server - Dependency Check
Verifies all required dependencies are properly installed.
"""

import sys
import importlib
from typing import List, Tuple

def check_dependency(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a dependency is available"""
    try:
        importlib.import_module(module_name)
        return True, f"✓ {package_name or module_name} - OK"
    except ImportError as e:
        return False, f"✗ {package_name or module_name} - MISSING ({e})"

def main():
    """Check all dependencies"""
    print("TanukiMCP Vision Server - Dependency Check")
    print("=" * 50)
    
    # Core dependencies
    core_deps = [
        ("fastmcp", "fastmcp"),
        ("PIL", "pillow"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pyautogui", "pyautogui"),
        ("mss", "mss"),
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("pydantic", "pydantic"),
        ("uvicorn", "uvicorn"),
        ("starlette", "starlette"),
        ("multipart", "python-multipart")
    ]
    
    # Video processing dependencies
    video_deps = [
        ("ffmpeg", "ffmpeg-python"),
        ("moviepy.editor", "moviepy"),
        ("imageio", "imageio")
    ]
    
    print("\nCore Dependencies:")
    core_missing = []
    for module, package in core_deps:
        success, message = check_dependency(module, package)
        print(f"  {message}")
        if not success:
            core_missing.append(package)
    
    print("\nVideo Processing Dependencies:")
    video_missing = []
    for module, package in video_deps:
        success, message = check_dependency(module, package)
        print(f"  {message}")
        if not success:
            video_missing.append(package)
    
    print("\n" + "=" * 50)
    
    if core_missing:
        print(f"❌ CRITICAL: Missing core dependencies: {', '.join(core_missing)}")
        print("   The server will NOT work without these.")
        print(f"   Install with: pip install {' '.join(core_missing)}")
        
    if video_missing:
        print(f"⚠️  WARNING: Missing video dependencies: {', '.join(video_missing)}")
        print("   Video recording features will be disabled.")
        print(f"   Install with: pip install {' '.join(video_missing)}")
    
    if not core_missing and not video_missing:
        print("🎉 ALL DEPENDENCIES INSTALLED!")
        print("   The server is ready to run with full functionality.")
    elif not core_missing:
        print("✅ Core dependencies OK - Server will run with basic functionality.")
    
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
    
    return len(core_missing) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 