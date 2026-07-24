import os

def modernize_file(file_path):
    print(f"Modernizing workflow: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace deprecated actions
    content = content.replace("actions/checkout@v2", "actions/checkout@v4")
    content = content.replace("actions/setup-python@v2", "actions/setup-python@v5")
    
    # Replace host runner version (ubuntu-18.04 is deprecated and removed)
    content = content.replace("runs-on: ubuntu-18.04", "runs-on: ubuntu-20.04")
    
    # Update Python 3.7 to Python 3.9
    content = content.replace("python-version: [3.7]", "python-version: [3.9]")
    content = content.replace("python-version: 3.7", "python-version: 3.9")
    content = content.replace("python-version: [3.7, 3.8, 3.9]", "python-version: [3.8, 3.9]")
    
    # Check lint.yml or other hardcoded Python 3.7 names
    content = content.replace("Set up Python 3.7", "Set up Python 3.9")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    workflows_dir = ".github/workflows"
    for filename in os.listdir(workflows_dir):
        if filename.endswith(".yml"):
            modernize_file(os.path.join(workflows_dir, filename))
            
if __name__ == "__main__":
    main()
