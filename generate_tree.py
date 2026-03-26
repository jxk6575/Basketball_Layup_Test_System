import os
from pathlib import Path

def generate_tree(startpath, output_file):
    startpath_str = str(startpath)
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath_str):
            level = root.replace(startpath_str, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

def main():
    project_root = Path(__file__).parent
    output_file = project_root / "project_structure.txt"
    generate_tree(project_root, output_file)
    print(f"Project structure saved to {output_file}")

if __name__ == "__main__":
    main() 