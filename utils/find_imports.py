import os
import re
import sys
import argparse
import subprocess
import json
from stdlib_list import stdlib_list

# Add a mapping of import names to package names
IMPORT_TO_PACKAGE_MAP = {
    'yaml': 'PyYAML',
    'ruamel': 'ruamel.yaml',
}

def find_imports_in_file(filepath):
    imports = set()
    if filepath.endswith('.py'):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            import_matches = re.findall(r'^\s*import\s+([\w.]+)', content, re.MULTILINE)
            from_import_matches = re.findall(r'^\s*from\s+([\w.]+)\s+import', content, re.MULTILINE)
            imports.update(match.split('.')[0] for match in import_matches if not match.startswith('.'))
            imports.update(match.split('.')[0] for match in from_import_matches if not match.startswith('.'))
    elif filepath.endswith('.ipynb'):
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            for cell in notebook.get('cells', []):
                if cell['cell_type'] == 'code':
                    content = ''.join(cell['source'])
                    import_matches = re.findall(r'^\s*import\s+([\w.]+)', content, re.MULTILINE)
                    from_import_matches = re.findall(r'^\s*from\s+([\w.]+)\s+import', content, re.MULTILINE)
                    imports.update(match.split('.')[0] for match in import_matches if not match.startswith('.'))
                    imports.update(match.split('.')[0] for match in from_import_matches if not match.startswith('.'))
    return imports

def find_imports(root_dir):
    imports = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                filepath = os.path.join(root, file)
                imports.update(find_imports_in_file(filepath))
    return imports

def get_installed_version(package):
    try:
        result = subprocess.run(['pip', 'show', package], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            version_line = re.search(r'^Version: (.+)$', result.stdout, re.MULTILINE)
            if version_line:
                return version_line.group(1)
    except Exception as e:
        print(f"Error retrieving version for {package}: {e}")
    return "Version not found"

def generate_requirements(imports, find_versions=False):
    python_version = ".".join(map(str, sys.version_info[:2]))
    standard_libs = set(stdlib_list(python_version))

    requirements = []
    for package in sorted(imports):
        if package not in standard_libs:
            package_name = IMPORT_TO_PACKAGE_MAP.get(package, package)
            if find_versions:
                version = get_installed_version(package_name)
                requirements.append(f"{package_name}=={version}" if version != "Version not found" else f"{package_name} (Version not found)")
            else:
                requirements.append(package_name)
    return requirements

def parse_arguments():
    parser = argparse.ArgumentParser(description='Find and analyze imports in Python and Jupyter Notebook files.')
    parser.add_argument('path', nargs='?', default=os.getcwd(), help='Path to the directory or file')
    parser.add_argument('--output', help='Output file to save the results')
    parser.add_argument('--versions', action='store_true', help='Find installed versions of the packages')
    parser.add_argument('--imports-file', help='File containing a list of imports to check versions')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.imports_file:
        with open(args.imports_file, 'r') as f:
            imports = {line.strip() for line in f}
    else:
        imports = find_imports(args.path)

    requirements = generate_requirements(imports, find_versions=args.versions)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write("\n".join(requirements))
    else:
        print("\n".join(requirements))

if __name__ == "__main__":
    main()
