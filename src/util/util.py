import ast
import os
import re
import io
import hashlib
import numpy as np
import pandas as pd
import difflib


def extract(message: str, key: str, sep=None):
    """从 message 中提取以三颗星(***key***) 包裹的内容。

    OpenAI 等模型的回复格式可能存在以下差异:
        1. `***python_code:` 与冒号后紧跟换行
        2. `***python_code` 不带冒号

    为了提升鲁棒性, 这里对冒号做了可选处理, 并允许 key 前后存在空白或换行。

    参数:
        message (str): 待搜索的完整文本。
        key (str): 需要提取的关键字段, 例如 "python_code"。
        sep (str, optional): 如果提供, 在返回前按此分隔符对提取结果再做一次 split。

    返回:
        Union[str, list[str], None]:
            - 如果找到内容且未指定 sep, 返回去除首尾空白后的字符串。
            - 如果找到内容且指定 sep, 返回列表。
            - 如果在 stars 块中显式返回了 "None"/"none", 返回 None。
            - 如果未找到, 返回 [] (当指定 sep) 或 None (未指定)。
    """
    # print("[qwq begin]\n{message}\n[qwq end]\n")

    # 统一换行, 防止因连续空行导致正则不匹配
    message = message.replace("\r\n", "\n").replace("\n\n", "\n")

    # 构造若干正则模板, 对冒号设置为可选(:?)
    patterns = [
        rf"\*\*\*\s*{key}:?(.*?)\*\*\*",               # ***python_code: ... *** 或 ***python_code ... ***
        rf" \*\*\*{key}:?(.*?)\*\*\*",                   # 以空格开头的 ***python_code***
        rf"\*\*\*\n{key}:?(.*?)\*\*\*",                # 换行后紧跟 key
        rf"\*.*{key}:?(.*?)\*",                            # 单星包装
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.DOTALL)
        if match:
            value = match.group(1).strip()
            # 处理显式的 None 标记
            if value in ["None", "none", "none."]:
                return None
            if sep:
                return value.split(sep)
            return value

    # 额外: 针对 python_code 关键字, 可能使用 ```python ``` 代码块而非星号包装
    if key == "python_code":
        code_block = re.search(r"```python(.*?)```", message, re.DOTALL)
        if code_block:
            value = code_block.group(1).strip()
            if sep:
                return value.split(sep)
            return value

    # 未找到匹配项, 根据是否要求分割返回默认值
    return [] if sep else None

def find_closest_match(input_string, string_list):
    matches = difflib.get_close_matches(input_string, string_list, n=1, cutoff=0.0)
    return matches[0] if matches else None 

def parse_text_to_dict(text):
    lines = text.split("\n")
    result = {}
    current_key = None
    current_content = []
    for line in lines:
        if len(line) > 0 and line[0] == "-" and ":" in line:
            if current_key:
                result[current_key.replace(" ", "")] = "\n".join(current_content).strip()
            current_key = line[1:].split(":")[0]
            current_content = []
            if len(line.split(":")) > 0:
                current_content = [line.split(":")[1]]
        elif current_key:
            current_content.append(line)
    if current_key:
        result[current_key.replace(" ", "")] = "\n".join(current_content).strip()
    return result

def load_function(file:str, problem: str="base", function_name: str=None) -> callable:
    if not "\n" in file:
        if not file.endswith(".py"):
            # File name
            file += ".py"
        file_path = search_file(file, problem)
        assert file_path is not None
        code = open(file_path, "r").read()
    else:
        # code only
        code = file

    if function_name is None:
        function_name = file.split(os.sep)[-1].split(".")[0]

    exec(code, globals())
    assert function_name in globals()
    return eval(function_name)

def load_framework_description(component_code: str) -> tuple[str, str]:
    """ Load framework description for the problem from source code, including solution design and operators design."""
    def get_method_source(method_node):
        """Convert the method node to source component_code, ensuring correct indentation."""
        source_lines = ast.unparse(method_node).split('\n')
        indented_source = '\n'.join(['    ' + line for line in source_lines])  # Indent the source component_code
        return indented_source

    tree = ast.parse(component_code)
    solution_str = ""
    operator_str = ""

    # Traverse the AST to find the Solution and Operator classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(base.id == 'BaseSolution' for base in node.bases if isinstance(base, ast.Name)):
                # Extract Solution class with only __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        solution_str += f"class {node.name}:\n"
                        solution_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                        solution_str += get_method_source(item) + "\n"
            elif any(base.id == 'BaseOperator' for base in node.bases if isinstance(base, ast.Name)):
                # Extract Operator class with only __init__ and run methods
                operator_str += f"class {node.name}(BaseOperator):\n"
                operator_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in ['__init__', 'run']:
                        operator_str += get_method_source(item) + "\n"

    return solution_str.strip(), operator_str.strip()

def extract_function_with_docstring(code_str, function_name):
    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?)\"\"\""
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def filter_dict_to_str(dicts: list[dict], content_threshold: int=None) -> str:
    if isinstance(dicts, dict):
        dicts = [dicts]
    total_dict = {k: v for d in dicts for k, v in d.items()}
    strs = []
    for key, value in total_dict.items():
        if callable(value):
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if "\n" in str(value):
            key_value_str = str(key) + ":\n" + str(value)
        else:
            key_value_str = str(key) + ":" + str(value)
        if content_threshold is None or len(key_value_str) <= content_threshold:
            strs.append(key_value_str)
    return "\n".join(strs)

def find_key_value(source_dict: dict, key: object) -> object:
    if key in source_dict:
        return source_dict[key]
    else:
        for k, v in source_dict.items():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
    return None

def extract_function_with_short_docstring(code_str, function_name):

    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?).*Args"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        string = match.group(0)
        function_name = string.split("(")[0].strip()
        parameters = string.split("algorithm_data: dict")[1].split(", **kwargs")[0].strip()
        if parameters[:2] == ", ":
            parameters = parameters[2:]
        introduction = string.split("\"\"\"")[1].split("Args")[0].strip()
        introduction = re.sub(r'\s+', ' ', introduction)
        return f"{function_name}({parameters}): {introduction}"
    else:
        print(f"[qwq] {function_name}")
        print(f"[qwq] {code_str}")
        return None

def parse_paper_to_dict(content: str, level=0):
    if level == 0:
        pattern = r'\\section\{(.*?)\}(.*?)((?=\\section)|\Z)'
    elif level == 1:
        pattern = r'\\subsection\{(.*?)\}(.*?)((?=\\subsection)|(?=\\section)|\Z)'
    elif level == 2:
        pattern = r'\\subsubsection\{(.*?)\}(.*?)((?=\\subsubsection)|(?=\\subsection)|(?=\\section)|\Z)'
    else:
        raise ValueError("Unsupported section level")
    sections = re.findall(pattern, content, re.DOTALL)
    section_dict = {}
    for title, body, _ in sections:
        body = body.strip()
        if level < 2:
            sub_dict = parse_paper_to_dict(body, level + 1)
            if sub_dict:
                section_dict[title] = sub_dict
            else:
                section_dict[title] = body
        else:
            section_dict[title] = body
    if level == 0:
        if "\\begin{abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{abstract}")[-1].split("\\end{abstract}")[0]
        if "\\begin{Abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{Abstract}")[-1].split("\\end{Abstract}")[0]
        if "\\title{" in content:
            section_dict["Title"] = content.split("\\title{")[-1].split("}")[0]
    return dict(section_dict)

def replace_strings_in_dict(source_dict: dict, replace_value: str="...") -> dict:
    for key in source_dict:
        if isinstance(source_dict[key], str):
            source_dict[key] = replace_value 
        elif isinstance(source_dict[key], dict):
            source_dict[key] = replace_strings_in_dict(source_dict[key])
    return source_dict

def search_file(file_name: str, problem: str="base") -> str:
    def find_file_in_folder(folder_path, file_name):
        return next((os.path.join(root, file_name) for root, dirs, files in os.walk(folder_path) if file_name in files or file_name in dirs), None)

    if os.path.exists(file_name):
        return file_name

    file_path = find_file_in_folder(os.path.join("src", "problems", problem), file_name)
    if file_path:
        return file_path

    if os.getenv("AMLT_DATA_DIR"):
        output_dir = os.getenv("AMLT_DATA_DIR")
    else:
        output_dir = "output"

    file_path = find_file_in_folder(os.path.join(output_dir, problem, "data"), file_name)
    if file_path:
        return file_path

    file_path = find_file_in_folder(os.path.join(output_dir, problem, "heuristics"), file_name)
    if file_path:
        return file_path

    file_path = find_file_in_folder(os.path.join(output_dir, problem), file_name)
    if file_path:
        return file_path
    return None

def df_to_str(df: pd.DataFrame) -> str:
    return df.to_csv(sep="\t", index=False).replace("\r\n", "\n").strip()

def str_to_df(string: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(string), sep="\t")

def compress_numbers(s):
    def format_float(match):
        number = match.group()
        if '.' in number and len(number.split('.')[1]) > 2:
            return "{:.2f}".format(float(number))
        return number
    return re.sub(r'\d+\.\d{3,}', format_float, s)

def sanitize_function_name(name: str, id_str: str="None"):
    s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
    sanitized_name = re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()

    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(" ", "_").replace("__", "_")

    # Remove invalid characters
    sanitized_name = "".join(char for char in sanitized_name if char.isalnum() or char == '_')

    # Ensure it doesn't start with a digit
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name

    suffix_str = hashlib.sha256(id_str.encode()).hexdigest()[:4]
    # Add uuid to avoid duplicated name
    sanitized_name = sanitized_name + "_" + suffix_str

    return sanitized_name