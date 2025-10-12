#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:15:20 2025

@author: dev
"""


import os
import re
import pkg_resources


PACKAGE_MAPPING = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow"
}


def filter_installed_packages(imports: set):
    filtered_imports = set()
    for lib in imports:
        if lib not in os.listdir():
            filtered_imports.add(PACKAGE_MAPPING.get(lib, lib))
        if lib == 'pandas':
            filtered_imports.add('openpyxl')
    return filtered_imports


def get_library_versions(imports: set):
    library_versions = {}
    for lib in imports:
        try:
            library_versions[lib] = pkg_resources.get_distribution(lib).version
        except pkg_resources.DistributionNotFound:
            print(f"Warning: Library '{lib}' is not installed.")
        except Exception as e:
            print(f"Error while fetching version for '{lib}': {e}")
    return library_versions


def get_all_py_files(root_dir="."):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".py"):
                py_files.append(os.path.join(dirpath, f))
    return py_files


def extract_imports(file_path: str):
    imports = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Remove comments
    code = re.sub(r'#.*', '', code)

    # Collapse bracketed multiline imports into a single line
    code = re.sub(r'\(\s*([^)]*?)\s*\)',
                  lambda m: m.group(1).replace("\n", " "), code, flags=re.DOTALL)

    # Match `import module` and `from module import ...`
    pattern = re.compile(r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)', re.MULTILINE)
    for match in pattern.findall(code):
        imports.add(match.split('.')[0])

    return imports


def create_requirements_file(library_versions: dict,
                             output_path: str = 'requirements.txt'):
    """
    Creates a requirements.txt file with the specified library versions.
    If the file already exists, it checks for mismatches before overwriting.

    Parameters
    ----------
    library_versions : dict
        A dictionary of library names as keys and their versions as values.
    output_path : str, optional
        The file path where the requirements file should be created. Default is 'requirements.txt'.

    Returns
    -------
    None.
    """
    new_content = "\n".join(f"{lib}=={version}" for lib,
                            version in library_versions.items())

    print(new_content)

    if os.path.exists(output_path):
        with open(output_path, 'r') as req_file:
            existing_content = req_file.read().strip()

        if existing_content == new_content:
            print(
                f"No changes detected in {output_path}. File remains unchanged.")
            return

    with open(output_path, 'w') as req_file:
        req_file.write(new_content + "\n")

    print(f"requirements.txt created/updated at {output_path}")


def main():
    py_files = get_all_py_files(".")
    all_imports = set()

    for py_file in py_files:
        imports = [
            i.strip()
            for i in extract_imports(py_file)
            if i.strip()
        ]
        all_imports.update(imports)

    filtered_imports = filter_installed_packages(all_imports)
    library_versions = get_library_versions(filtered_imports)
    create_requirements_file(library_versions)


if __name__ == '__main__':
    main()
