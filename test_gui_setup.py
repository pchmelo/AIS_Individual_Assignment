"""
GUI Test Script
Verifies that the GUI components are properly set up
Run this before launching the GUI for the first time
"""

import sys
import os
from pathlib import Path

def check_color(status):
    """Return colored status indicator"""
    return "✅" if status else "❌"

def test_imports():
    """Test if all required imports are available"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    tests = {
        "streamlit": False,
        "pandas": False,
        "numpy": False,
        "matplotlib": False,
        "seaborn": False
    }
    
    for module in tests.keys():
        try:
            __import__(module)
            tests[module] = True
            print(f"{check_color(True)} {module:20s} OK")
        except ImportError:
            print(f"{check_color(False)} {module:20s} MISSING")
    
    all_passed = all(tests.values())
    print(f"\nImport Test: {check_color(all_passed)} {'PASSED' if all_passed else 'FAILED'}")
    
    if not all_passed:
        print("\n⚠️  Install missing packages:")
        print("   pip install -r requirements-gui.txt")
    
    return all_passed

def test_file_structure():
    """Test if all required files exist"""
    print("\n" + "="*60)
    print("TESTING FILE STRUCTURE")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    required_files = {
        "src/main.py": base_dir / "src" / "main.py",
        "src/gui_app.py": base_dir / "src" / "gui_app.py",
        "src/pipeline.py": base_dir / "src" / "pipeline.py",
        "requirements-gui.txt": base_dir / "requirements-gui.txt",
        "src/data": base_dir / "src" / "data",
        "reports": base_dir / "reports"
    }
    
    tests = {}
    for name, path in required_files.items():
        exists = path.exists()
        tests[name] = exists
        status = "EXISTS" if exists else "MISSING"
        print(f"{check_color(exists)} {name:30s} {status}")
    
    all_passed = all(tests.values())
    print(f"\nFile Structure Test: {check_color(all_passed)} {'PASSED' if all_passed else 'FAILED'}")
    
    if not tests["src/data"]:
        print("\n⚠️  Create data directory:")
        print("   mkdir src/data")
    
    if not tests["reports"]:
        print("\n⚠️  Create reports directory:")
        print("   mkdir reports")
    
    return all_passed

def test_datasets():
    """Test if datasets are available"""
    print("\n" + "="*60)
    print("TESTING DATASETS")
    print("="*60)
    
    data_dir = Path(__file__).parent / "src" / "data"
    
    if not data_dir.exists():
        print(f"{check_color(False)} Data directory does not exist")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"{check_color(False)} No CSV files found in src/data/")
        print("\n⚠️  Add datasets to src/data/ folder")
        return False
    
    print(f"{check_color(True)} Found {len(csv_files)} dataset(s):")
    for csv in csv_files:
        print(f"   • {csv.name}")
    
    print(f"\nDataset Test: {check_color(True)} PASSED")
    return True

def test_python_version():
    """Test Python version"""
    print("\n" + "="*60)
    print("TESTING PYTHON VERSION")
    print("="*60)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    is_valid = version.major == 3 and version.minor >= 8
    
    print(f"Python Version: {version_str}")
    print(f"Required: Python 3.8+")
    print(f"\nPython Version Test: {check_color(is_valid)} {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid:
        print("\n⚠️  Please upgrade to Python 3.8 or higher")
    
    return is_valid

def test_main_py_mode():
    """Test if main.py has correct mode setting"""
    print("\n" + "="*60)
    print("TESTING MAIN.PY CONFIGURATION")
    print("="*60)
    
    main_py = Path(__file__).parent / "src" / "main.py"
    
    if not main_py.exists():
        print(f"{check_color(False)} main.py not found")
        return False
    
    content = main_py.read_text()
    
    has_run_mode = 'RUN_MODE' in content
    print(f"{check_color(has_run_mode)} RUN_MODE variable found")
    
    if has_run_mode:
        # Try to extract the mode
        for line in content.split('\n'):
            if 'RUN_MODE' in line and '=' in line and not line.strip().startswith('#'):
                mode_line = line.strip()
                if '"terminal"' in mode_line:
                    print(f"   Current mode: TERMINAL")
                elif '"gui"' in mode_line:
                    print(f"   Current mode: GUI")
                else:
                    print(f"   Current mode: UNKNOWN")
                break
    
    has_imports = 'import subprocess' in content and 'import os' in content
    print(f"{check_color(has_imports)} Required imports present")
    
    has_gui_function = 'def run_gui_mode' in content
    print(f"{check_color(has_gui_function)} GUI mode function defined")
    
    all_passed = has_run_mode and has_imports and has_gui_function
    print(f"\nMain.py Configuration Test: {check_color(all_passed)} {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

def test_gui_app_imports():
    """Test if gui_app.py can be imported"""
    print("\n" + "="*60)
    print("TESTING GUI_APP.PY")
    print("="*60)
    
    gui_app = Path(__file__).parent / "src" / "gui_app.py"
    
    if not gui_app.exists():
        print(f"{check_color(False)} gui_app.py not found")
        return False
    
    # Add src to path
    sys.path.insert(0, str(gui_app.parent))
    
    try:
        # Try to read the file
        content = gui_app.read_text()
        
        checks = {
            "Streamlit import": "import streamlit" in content,
            "Pipeline import": "from pipeline import" in content,
            "Main function": "def main()" in content,
            "Page config": "st.set_page_config" in content,
            "Session state": "st.session_state" in content
        }
        
        for check_name, passed in checks.items():
            print(f"{check_color(passed)} {check_name}")
        
        all_passed = all(checks.values())
        print(f"\nGUI App Test: {check_color(all_passed)} {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed
    
    except Exception as e:
        print(f"{check_color(False)} Error reading gui_app.py: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, result in results.items():
        print(f"{check_color(result)} {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{check_color(True)} ALL TESTS PASSED!")
        print("\n✨ You're ready to launch the GUI!")
        print("\nTo start:")
        print("1. Set RUN_MODE = 'gui' in src/main.py")
        print("2. Run: python src/main.py")
        print("   OR")
        print("   Run: streamlit run src/gui_app.py")
    else:
        print(f"\n{check_color(False)} SOME TESTS FAILED")
        print("\n⚠️  Please fix the issues above before running the GUI")
        print("\nCommon fixes:")
        print("• Install dependencies: pip install -r requirements-gui.txt")
        print("• Create directories: mkdir src/data reports")
        print("• Add datasets to src/data/ folder")
    
    print(f"{'='*60}\n")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GUI SETUP VERIFICATION")
    print("="*60)
    print("\nThis script checks if your GUI environment is properly configured")
    print("Running tests...\n")
    
    results = {
        "Python Version": test_python_version(),
        "Required Imports": test_imports(),
        "File Structure": test_file_structure(),
        "Datasets Available": test_datasets(),
        "Main.py Configuration": test_main_py_mode(),
        "GUI App Structure": test_gui_app_imports()
    }
    
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
