import os
import sys
import time
import json
import yaml
import pickle
import inspect
import datetime

import numpy as np
from tqdm import tqdm
from uuid import UUID
from nlpka.env import env
from types import SimpleNamespace
from collections.abc import KeysView, ValuesView, ItemsView

# RICH installations --------------------------------------------------------------------------------------------------------------------------------------
from rich import print as rprint
from rich import pretty
pretty.install()

show_locals = bool(int(env['SHOW_LOCALS']))
from rich.traceback import install
def install_traceback(show_locals = show_locals, width = 120, extra_lines = 1):
    install(show_locals=show_locals, width=width, extra_lines=extra_lines)
install_traceback()

def print_traceback(show_locals = show_locals, width = 120, extra_lines = 1):
    install_traceback(show_locals, width, extra_lines)
    raise Exception("Ephemeral Exception")

# --------------------------------------------------------------------------------------------------------------------------------------
from enum import Enum
class SimpleEnum(Enum):
    def __get__(self, instance, owner):
        return self.value

# --------------------------------------------------------------------------------------------------------------------------------------
def info(file = __file__, name = __name__, package = __package__):

    info                = {}
    info['file      ']  = file 
    info['module    ']  = name 
    info['package   ']  = package 

    print('Module info:', json_dumps(info))

# --------------------------------------------------------------------------------------------------------------------------------------
def tik(tok, key, callback, params = ()):
    
    start       = time.perf_counter()
    res         = callback(*params)
    end         = time.perf_counter()
    tok[key]    = format_seconds(end - start)

    return res

def format_seconds(n):
    return str(datetime.timedelta(seconds = n))

def get_time_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------------------------------------------------------------------------------------------------------------------
def go_up_to_dir(path, dir_name):

    curr_dir_name       = os.path.basename(path)

    while curr_dir_name and curr_dir_name != dir_name:

        path            = os.path.dirname(path)
        curr_dir_name   = os.path.basename(path)

    return path if curr_dir_name == dir_name else False

# --------------------------------------------------------------------------------------------------------------------------------------
def root_path():
    # return os.path.dirname(__file__)
    return go_up_to_dir(__file__, env['PACKAGE_NAME'])

# --------------------------------------------------------------------------------------------------------------------------------------
def root_to_search_paths():
    sys.path.append(root_path())

# --------------------------------------------------------------------------------------------------------------------------------------
def absolute_to_root_relative(absolute_path):
    rp = root_path()
    root_dir = rp.split("/")[-1:][0]
    root_relative_path = os.path.join(root_dir, os.path.relpath(absolute_path, rp))
    return root_relative_path

# --------------------------------------------------------------------------------------------------------------------------------------
def get_root_relative_module_location(module):
    return absolute_to_root_relative(get_module_location(module))

# --------------------------------------------------------------------------------------------------------------------------------------
def get_module_location(module):
    return get_file_location(module.__file__, True)
    
# --------------------------------------------------------------------------------------------------------------------------------------
def get_file_location(file, ignor_symlink = False):
    if not ignor_symlink:
        return os.path.dirname(os.path.abspath(file))
    else:
        return os.path.dirname(os.path.abspath(os.path.realpath(file)))

# --------------------------------------------------------------------------------------------------------------------------------------
def get_relative_module_location(module):
    return os.path.relpath(get_module_location(module), root_path())

# --------------------------------------------------------------------------------------------------------------------------------------
def get_dir_items(dir_path, only_dirs=False, only_files=False):
    '''List all files and directories in the specified directory'''
    res_items = []
    # Check if the specified directory exists
    if os.path.exists(dir_path):
        # List all files and directories in the specified directory
        items = os.listdir(dir_path)
        for item in items:
            item_path = os.path.join(dir_path, item)
            # Check if it is a file (not a directory)
            if os.path.isfile(item_path) and not only_dirs:
                res_items.append(item)
            # Check if it is a directory (not a file)
            if os.path.isdir(item_path) and not only_files:
                res_items.append(f'{item}/')
    return res_items

from tqdm import tqdm

def find_dir_path(root_dir, dir_prefix):
    '''
        Find directory's path in the specified root directory 
        that starts with the specified prefix
    '''
    matching_dirs = []
    print(root_dir, dir_prefix)
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc="Walking through directories"):
        for dirname in dirnames:
            if dirname.startswith(dir_prefix):
                full_path = os.path.join(dirpath, dirname)
                matching_dirs.append(full_path)
    return matching_dirs


def read_jsons_in_folder(folder_path):
    ''' loads all .json files inside given folder and returns their list'''
    files = [folder_path+'/'+name for name in os.listdir(folder_path) 
             if name[-5:]=='.json']
    jss = []
    for file in files:
        f = open(file, "r")
        jss.append(json.load(f))
        f.close()
    return jss

# --------------------------------------------------------------------------------------------------------------------------------------
def print_list(list):
    for element in list:
        print(element)

# def print_gpu_memory():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")

# --------------------------------------------------------------------------------------------------------------------------------------
def json_dumps(object, **kwargs):
    return json.dumps(object, sort_keys=False, indent=4, ensure_ascii=False, **kwargs)

def json_dumps_numpy(object, **kwargs):
    return json_dumps(object, cls=NumpyEncoder, **kwargs)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
# Simple Namespace ---------------------------------------------------------------------------------------------------------------------
def json_dumps_simple_nsp(simple_nsp): # dump
    return json_dumps(simple_nsp, default=safe_json_default)

def safe_json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif hasattr(o, "__dict__"):
        return o.__dict__
    else:
        return str(o)  # fallback for enums, etc.

def json_load_simple_nsp(json_string): # laod
    return json.loads(json_string, object_hook=lambda d: SimpleNamespace(**d))

def copy_simple_nsp(simple_nsp): # dump and load 
    return json_load_simple_nsp(json_dumps_simple_nsp(simple_nsp))

def dict_to_simple_nsp(dictionary = {}): # dump and load
    return json_load_simple_nsp(json_dumps(scientific_notation_to_float(dictionary)))

def simple_nsp_to_dict(simple_nsp): # dump and laod
    return json.loads(json_dumps_simple_nsp(simple_nsp))

def simple_nsps_to_params(*simple_nsps):
    params = {}
    # Merging each SimpleNamespace into the params dictionary
    for simple_nsp in simple_nsps:
        params.update(vars(simple_nsp))
    return params

def update_simple_nsp(simple_nsps, updates):
    for key, value in vars(updates).items():
        setattr(simple_nsps, key, value)

# Read Write File ----------------------------------------------------------------------------------------------------------------------

# Read
def json_file_to_dict(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)

def json_file_to_simple_nsp(json_file_path):
    return dict_to_simple_nsp(json_file_to_dict(json_file_path))
        
def yaml_file_to_dict(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.safe_load(yaml_file)
    
def yaml_file_to_simple_nsp(yaml_file_path):
    return dict_to_simple_nsp(yaml_file_to_dict(yaml_file_path))

# Write
def dict_to_json_file(dict, json_file_path):
    with open(json_file_path, "w") as json_file:
        json_file.write(json_dumps(dict))

def simple_nsp_to_json_file(object, json_file_path):
    with open(json_file_path, "w") as json_file:
        json_file.write(json_dumps_simple_nsp(object))
    
def dict_to_yaml_file(dict, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(dict, yaml_file, default_flow_style=False)

def simple_nsp_to_yaml_file(object, yaml_file_path):
    dict_to_yaml_file(simple_nsp_to_dict(object), yaml_file_path)
    
# Convert string "5e-5" to float 0.00005 -----------------------------------------------------------------------------------------------
def scientific_notation_to_float(item):
    if isinstance(item, dict):
        return {k: scientific_notation_to_float(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [scientific_notation_to_float(v) for v in item]
    elif isinstance(item, str):
        try:
            return float(item) if 'e' in item else item
        except ValueError:
            return item
    else:
        return item
    
# --------------------------------------------------------------------------------------------------------------------------------------
def get_placeholder(name):
    return '<PLACEHOLDER:' + name + '>'

# --------------------------------------------------------------------------------------------------------------------------------------
def get_script_param(len, number, default = None, p = False):
    param = default if len <= number else sys.argv[number]
    param = type(default)(param) if param is not None and default is not None else param
    print(param) if p else None
    return param    
    
# --------------------------------------------------------------------------------------------------------------------------------------
def file_len(path, print_lines=False):
    with open(path, 'r') as f:
        l = sum(1 for _ in tqdm(f))
    if print_lines: print(f'{l} lines in {path}')
    return l

def sizeof_file(file):
    return format_size(os.path.getsize(file)) if os.path.exists(file) else 0

def sizeof_object(object):
    return format_size(sys.getsizeof(object))

def format_size(num, suffix = 'B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# --------------------------------------------------------------------------------------------------------------------------------------
def pickle_save(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def pickle_load(path):    
    with open(path, 'rb') as f:
        return pickle.load(f)
# --------------------------------------------------------------------------------------------------------------------------------------
def is_sublist(smaller_list, larger_list):
    """
    Check if smaller_list is a sublist of larger_list.

    Args:
    - smaller_list (list or np.array): The list to be checked as a sublist.
    - larger_list (list or np.array): The list in which to search for the sublist.

    Returns:
    - bool: True if smaller_list is a sublist of larger_list, False otherwise.
    """
    
    # Convert to numpy arrays if they are not already
    if not isinstance(smaller_list, np.ndarray):
        smaller_list = np.array(smaller_list)
    if not isinstance(larger_list, np.ndarray):
        larger_list = np.array(larger_list)
    
    len_smaller = len(smaller_list)
    len_larger = len(larger_list)

    # Edge case: if smaller list is empty, it's considered a sublist
    if len_smaller == 0:
        return True
    
    # Edge case: if smaller list is larger than the larger list, it can't be a sublist
    if len_smaller > len_larger:
        return False
    
    # Sliding window approach with numpy
    for i in range(len_larger - len_smaller + 1):
        if np.array_equal(larger_list[i:i + len_smaller], smaller_list):
            return True
    
    return False
# --------------------------------------------------------------------------------------------------------------------------------------
def try_set_add(set, element):
    """
    Attempts to add an element to the set.
    Returns True if the element was added (it did not exist in the set).
    Returns False if the element was not added (it already existed in the set).
    """
    l = len(set)
    set.add(element)
    return True if l != len(set) else False
# --------------------------------------------------------------------------------------------------------------------------------------
def filter_kwargs_by_method_signature(method, kwargs):
    """Filter kwargs to only include valid parameters for the given method."""
    signature = inspect.signature(method)
    valid_params = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in valid_params}

# --------------------------------------------------------------------------------------------------------------------------------------
def to_utf8_if_binary(text):
    if isinstance(text, list) and isinstance(text[0], bytes):
        text = [t.decode('utf-8') for t in text]
    elif isinstance(text, bytes):
        text = text.decode('utf-8')
    return text

def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test

# Parse Config Name Argument From Script Run -------------------------------------------------------------------------------------------
def parse_script_args(ap = None):
    '''
    Extention Example:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arg_2', type=str, help='Named Flag Argument')
    args, config_name = parse_script_args(ap)
    '''
    import argparse
    config_name_only = False if ap else True
    ap = ap or argparse.ArgumentParser()
    ap.add_argument('--config', required = True, help = 'Configuration File Name')
    args = ap.parse_args()
    return args.config if config_name_only else (args, args.config)

def parse_config_name():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('config_name', help = 'Configuration File Name')
    args = ap.parse_args()
    return args.config_name

def print_stack():
    print("Call stack:")
    for frame in inspect.stack():
        print(f"  {frame.filename}:{frame.lineno} - {frame.function}")

# --------------------------------------------------------------------------------------------------------------------------------------
def p(*objects, end='\n', sep=' '):
    """
    Pretty prints multiple objects in a readable format using appropriate dump functions.
    
    Args:
        *objects: One or more objects of any type to be printed
        end: String to append after the last value (default: newline)
        sep: Separator between objects (default: space)
    """
    result = []
    
    for obj in objects:
        if isinstance(obj, SimpleNamespace):
            result.append(json_dumps_simple_nsp(obj))
        elif isinstance(obj, np.ndarray):
            result.append(json_dumps_numpy(obj))
        elif isinstance(obj, dict):
            result.append(json_dumps(obj))
        elif isinstance(obj, (KeysView, ValuesView, ItemsView, set, frozenset)):
            result.append(json_dumps(list(obj)))
        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            result.append(json_dumps(obj))
        elif isinstance(obj, (int, float, str, bool, type(None))):
            result.append(str(obj))
        else:
            try:
                # Try to convert to dict if object has __dict__ attribute
                if hasattr(obj, '__dict__'):
                    result.append(json_dumps(obj.__dict__))
                else:
                    result.append(str(obj))
            except:
                result.append(str(obj))
    
    rprint(sep.join(result), end=end)

# --------------------------------------------------------------------------------------------------------------------------------------
def dict_diff(d1, d2, print_diff=True):
    """
    Compare two dictionaries and show the differences between their keys and values.
    
    Args:
        d1: First dictionary
        d2: Second dictionary
        print_diff: If True, print the differences (default: True)
        
    Returns:
        Dictionary containing the differences
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        if print_diff:
            p("One or both inputs are not dictionaries")
        return {"error": "One or both inputs are not dictionaries"}
    
    diff = {}
    
    # Find keys in d1 but not in d2
    only_in_d1 = set(d1.keys()) - set(d2.keys())
    if only_in_d1:
        diff["only_in_first"] = {k: d1[k] for k in only_in_d1}
    
    # Find keys in d2 but not in d1
    only_in_d2 = set(d2.keys()) - set(d1.keys())
    if only_in_d2:
        diff["only_in_second"] = {k: d2[k] for k in only_in_d2}
    
    # Find keys with different values
    common_keys = set(d1.keys()) & set(d2.keys())
    diff_values = {}
    
    for k in common_keys:
        # If values are dictionaries, recursively compare them
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            sub_diff = dict_diff(d1[k], d2[k], print_diff=False)
            if sub_diff and not (len(sub_diff) == 1 and "error" in sub_diff):
                diff_values[k] = sub_diff
        # Otherwise directly compare values
        elif d1[k] != d2[k]:
            diff_values[k] = {"first": d1[k], "second": d2[k]}
    
    if diff_values:
        diff["different_values"] = diff_values
    
    # Print the differences if requested
    if print_diff:
        if not diff:
            p("No differences found between the dictionaries")
        else:
            p(diff)
    
    return diff

def monkey_patch_globally(name: str, new_obj, verbose=False):
    count = 0
    for module in list(sys.modules.values()):
        if module and hasattr(module, "__dict__") and name in module.__dict__:
            module.__dict__[name] = new_obj
            count += 1
            if verbose:
                print(f"🔁 Patched '{name}' in {module.__name__}")
    return count