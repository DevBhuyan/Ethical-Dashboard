#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:15:20 2025

@author: dev
"""


import sys
import os
import re
import pkg_resources


PACKAGE_MAPPING = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def extract_imports(file_path: str):
    '''


    Parameters
    ----------
    file_path : str
        DESCRIPTION.

    Returns
    -------
    imports : TYPE
        DESCRIPTION.

    '''

    imports = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Match `import module` or `from module import ...`
            # TODO: Need to add support for bracket multiline imports
            match = re.match(r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)', line)
            if match:
                module = match.group(1).split('.')[0]
                imports.add(module)

    return imports


'''
# FIXME: 
    ```
    Warning: Library 'os' is not installed.
    Warning: Library 're' is not installed.
    Warning: Library 'pkg_resources' is not installed.
    Warning: Library 'sklearn' is not installed.
    ```
    
    Need to be able to handle python base packages and mapping
'''


def filter_installed_packages(imports: set):
    """
    Filters out standard library modules from imports.
    Maps known mismatched library names.
    """

    # Python 3.10+ provides sys.stdlib_module_names; fallback for older versions
    built_in_modules = sys.stdlib_module_names if hasattr(
        sys, 'stdlib_module_names') else set()

    filtered_imports = set()
    for lib in imports:
        if lib not in built_in_modules:  # Ignore built-in modules
            filtered_imports.add(PACKAGE_MAPPING.get(lib, lib))

    return filtered_imports


def get_library_versions(imports: set):
    '''


    Parameters
    ----------
    imports : set
        DESCRIPTION.

    Returns
    -------
    library_versions : TYPE
        DESCRIPTION.

    '''

    library_versions = {}
    for lib in imports:
        try:
            version = pkg_resources.get_distribution(lib).version
            library_versions[lib] = version
        except pkg_resources.DistributionNotFound:
            print(f"Warning: Library '{lib}' is not installed.")
            # TODO: Need to add mapping of lib-names. For example, it cannot detect sklearn, which is installed as scikit-learn in pip
        except Exception as e:
            print(f"Error while fetching version for '{lib}': {e}")

    return library_versions


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
    '''


    Returns
    -------
    None.

    '''

    py_files = [f
                for f in os.listdir('.')
                if f.endswith('.py')]
    all_imports = set()

    for py_file in py_files:
        imports = extract_imports(py_file)
        all_imports.update(imports)

    filtered_imports = filter_installed_packages(all_imports)
    library_versions = get_library_versions(filtered_imports)
    library_versions = get_library_versions(all_imports)

    create_requirements_file(library_versions)


if __name__ == '__main__':
    main()
