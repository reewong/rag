from pathlib import Path
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
def parse_doxygen_output(output_dir: Path) -> Tuple[Dict, List[Tuple[str, str, str]]]:
    parsed_data = {
        "classes": {},
        "functions": {},
        "namespaces": {},
        "variables": {}
    }
    relationships = []

    for xml_file in output_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for compound in root.findall('compounddef'):
                kind = compound.get('kind')
                if kind in ['class', 'struct']:
                    parse_class(compound, parsed_data, relationships)
                elif kind == 'namespace':
                    parse_namespace(compound, parsed_data, relationships)
                elif kind == 'file':
                    parse_file(compound, parsed_data, relationships)

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            continue

    return parsed_data, relationships

def parse_class(class_elem, parsed_data: Dict, relationships: List):
    class_name = class_elem.find('compoundname').text
    class_details = class_elem.find('briefdescription').text or ""
    class_details += "\n" + (class_elem.find('detaileddescription').text or "")
    parsed_data['classes'][class_name] = {
        'methods': [],
        'attributes': [],
        'details': class_details.strip()
    }

    for section in class_elem.findall('sectiondef'):
        for member in section.findall('memberdef'):
            member_kind = member.get('kind')
            member_name = member.find('name').text
            if member_kind == 'function':
                parsed_data['classes'][class_name]['methods'].append(member_name)
                parse_function(member, parsed_data, relationships, f"{class_name}::{member_name}")
            elif member_kind == 'variable':
                parsed_data['classes'][class_name]['attributes'].append(member_name)

def parse_namespace(namespace_elem, parsed_data: Dict, relationships: List):
    namespace_name = namespace_elem.find('compoundname').text
    namespace_details = namespace_elem.find('briefdescription').text or ""
    namespace_details += "\n" + (namespace_elem.find('detaileddescription').text or "")
    parsed_data['namespaces'][namespace_name] = {
        'functions': [],
        'classes': [],
        'details': namespace_details.strip()
    }

    for inner_class in namespace_elem.findall('innerclass'):
        class_name = inner_class.text
        parsed_data['namespaces'][namespace_name]['classes'].append(class_name)

    for section in namespace_elem.findall('sectiondef'):
        for member in section.findall('memberdef'):
            if member.get('kind') == 'function':
                function_name = member.find('name').text
                full_function_name = f"{namespace_name}::{function_name}"
                parsed_data['namespaces'][namespace_name]['functions'].append(full_function_name)
                parse_function(member, parsed_data, relationships, full_function_name)

def parse_file(file_elem, parsed_data: Dict, relationships: List):
    for section in file_elem.findall('sectiondef'):
        for member in section.findall('memberdef'):
            if member.get('kind') == 'function':
                parse_function(member, parsed_data, relationships)
            elif member.get('kind') == 'variable':
                parse_variable(member, parsed_data, relationships)

def parse_variable(var_elem, parsed_data: Dict, relationships: List):
    var_name = var_elem.find('name').text
    var_type = var_elem.find('type').text
    var_details = var_elem.find('briefdescription').text or ""
    var_details += "\n" + (var_elem.find('detaileddescription').text or "")
    
    parsed_data['variables'][var_name] = {
        'type': var_type,
        'details': var_details.strip()
    }

def parse_function(function_elem, parsed_data: Dict, relationships: List, full_function_name: str = None):
    function_name = full_function_name or function_elem.find('name').text
    function_details = function_elem.find('briefdescription').text or ""
    function_details += "\n" + (function_elem.find('detaileddescription').text or "")
    parsed_data['functions'][function_name] = {
        'params': [],
        'return_type': function_elem.find('type').text,
        'details': function_details.strip()
     }
    
    for param in function_elem.findall('param'):
        param_name = param.find('declname')
        param_type = param.find('type')
        if param_name is not None and param_type is not None:
            parsed_data['functions'][function_name]['params'].append({
                'name': param_name.text,
                'type': param_type.text
            })

    def is_likely_function(ref_elem):
        if ref_elem.get('kindref') == 'member':
            return True
        ref_name = ref_elem.text
        if ref_name in parsed_data['functions']:
            return True
        if '(' in ref_name and ')' in ref_name:
            return True
        return False

    for references in function_elem.findall('references'):
        if is_likely_function(references):
            ref_name = references.text
            relationships.append((function_name, 'CALLS', ref_name))

    for referencedby in function_elem.findall('referencedby'):
        if is_likely_function(referencedby):
            ref_name = referencedby.text
            relationships.append((ref_name, 'CALLS', function_name))

import os
import re

def parse_cmakelists(file_path):
    modules = set()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        
        # Find add_library and add_executable
        modules.update(re.findall(r'add_library\s*\(([\w_-]+)', content))
        modules.update(re.findall(r'add_executable\s*\(([\w_-]+)', content))
        
        # Find add_subdirectory
        subdirs = re.findall(r'add_subdirectory\s*\(([\w_-]+)', content)
        modules.update(f"subdir:{subdir}" for subdir in subdirs)
    
    return modules

def get_basic_structure_by_cmake(root_directory):
    cmake_structure = {}
    
    for root, dirs, files in os.walk(root_directory):
        if root == f"{root_directory}/doxygen_output":
            continue
        # print(files)
        if 'CMakeLists.txt' in files:
            relative_path = os.path.relpath(root, root_directory)
            cmake_file = os.path.join(root, 'CMakeLists.txt')
            modules = parse_cmakelists(cmake_file)
            if modules:
                cmake_structure[relative_path] = modules
    
    cmake_structure_str = ''
    for directory, modules in cmake_structure.items():
        cmake_structure_str += f"Directory: {directory}\n"
        cmake_structure_str += "Modules:\n"
        for module in modules:
            if module.startswith("subdir:"):
                cmake_structure_str += f"  - Subdirectory: {module[7:]}\n"
            else:
                cmake_structure_str += f"  - {module}\n"
        cmake_structure_str += "\n"
    
    return cmake_structure_str.strip()


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