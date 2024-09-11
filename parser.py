from pathlib import Path
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
def parse_doxygen_output(output_dir: Path) -> Tuple[Dict, List[Tuple[str, str, str]]]:
    parsed_data = {
        "classes": {},
        "functions": {},
        "namespaces": {}
    }
    relationships = []

    for xml_file in output_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for compound in root.findall('compounddef'):
            kind = compound.get('kind')
            if kind in ['class', 'struct']:
                parse_class(compound, parsed_data, relationships)
            elif kind == 'namespace':
                parse_namespace(compound, parsed_data, relationships)

    return parsed_data, relationships
def parse_class(class_elem, parsed_data: Dict, relationships: List):
    class_name = class_elem.find('compoundname').text
    parsed_data['classes'][class_name] = {
        'details': extract_details(class_elem),
        'file': class_elem.find('location').get('file'),
        'line': class_elem.find('location').get('line'),
        'declaration': extract_declaration(class_elem)
    }

    # Extract inheritance relationships
    for base_class in class_elem.findall('.//basecompoundref'):
        relationships.append((class_name, 'INHERITS_FROM', base_class.text))

    # Extract member functions
    for member in class_elem.findall('sectiondef/memberdef[@kind="function"]'):
        function_name = f"{class_name}::{member.find('name').text}"
        parsed_data['functions'][function_name] = {
            'details': extract_details(member),
            'file': member.find('location').get('file'),
            'line': member.find('location').get('line'),
            'declaration': extract_declaration(member)
        }
        relationships.append((class_name, 'HAS_METHOD', function_name))

def parse_namespace(namespace_elem, parsed_data: Dict, relationships: List):
    namespace_name = namespace_elem.find('compoundname').text
    parsed_data['namespaces'][namespace_name] = {
        'details': extract_details(namespace_elem),
        'file': namespace_elem.find('location').get('file'),
        'line': namespace_elem.find('location').get('line')
    }

    # Extract namespace members (classes, functions)
    for member in namespace_elem.findall('innerclass'):
        class_name = member.text
        relationships.append((namespace_name, 'CONTAINS', class_name))

    for member in namespace_elem.findall('sectiondef/memberdef[@kind="function"]'):
        function_name = f"{namespace_name}::{member.find('name').text}"
        parsed_data['functions'][function_name] = {
            'details': extract_details(member),
            'file': member.find('location').get('file'),
            'line': member.find('location').get('line'),
            'declaration': extract_declaration(member)
        }
        relationships.append((namespace_name, 'CONTAINS', function_name))

def extract_details(elem) -> str:
    brief = elem.find('briefdescription')
    detailed = elem.find('detaileddescription')
    
    details = []
    if brief is not None and brief.text:
        details.append(brief.text.strip())
    if detailed is not None:
        for para in detailed.findall('.//para'):
            if para.text:
                details.append(para.text.strip())
    
    return "\n".join(details) if details else "No description available."

def extract_declaration(elem) -> str:
    declaration = ""
    definition = elem.find('definition')
    if definition is not None and definition.text:
        declaration += definition.text
    
    argsstring = elem.find('argsstring')
    if argsstring is not None and argsstring.text:
        declaration += argsstring.text
    
    return declaration.strip()

import os
import re
import subprocess

def run_cmake_trace(cmake_path):
    result = subprocess.run(['cmake', '--trace-expand', cmake_path], 
                            capture_output=True, text=True)
    return result.stdout

def parse_cmake_trace(trace_output):
    modules = {}
    current_file = ""
    
    for line in trace_output.split('\n'):
        file_match = re.search(r'CMakeLists\.txt:(\d+)\s*\((.*?)\)', line)
        if file_match:
            current_file = os.path.dirname(file_match.group(2))
        
        if current_file not in modules:
            modules[current_file] = set()
        
        if 'add_library' in line or 'add_executable' in line:
            match = re.search(r'add_\w+\s*\(([\w_-]+)', line)
            if match:
                modules[current_file].add(match.group(1))
        
        if 'add_subdirectory' in line:
            match = re.search(r'add_subdirectory\s*\(([\w_-]+)', line)
            if match:
                modules[current_file].add(f"subdir:{match.group(1)}")
    
    return modules

def extract_modules(root_dir):
    cmake_trace = run_cmake_trace(root_dir)
    return parse_cmake_trace(cmake_trace)

def get_basic_structure_by_cmake(root_directory):
    module_structure = extract_modules(root_directory)
    cmake_structure_str = ''
    for directory, modules in module_structure.items():
        cmake_structure_str += f"Directory: {directory}\n"
        cmake_structure_str +="Modules:\n"
        for module in modules:
            if module.startswith("subdir:"):
                cmake_structure_str += f"  - Subdirectory: {module[7:]}"
            else:
                cmake_structure_str += f"  - {module}"
    return cmake_structure_str


import os

def generate_directory_structure(startpath, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = set()
    if exclude_files is None:
        exclude_files = set()

    tree = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = '│   ' * level + '├── '
        for file in sorted(files):
            if file not in exclude_files:
                tree.append(f"{subindent}{file}")

    return '\n'.join(tree)