import sys
import os
from os import path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        try:
            os.environ["PYTHONPATH"] = path + ":" + os.environ["PYTHONPATH"]
        except KeyError:
            os.environ["PYTHONPATH"] = path

this_dir = osp.dirname(__file__)
repo_dir = osp.join(this_dir, '..')
add_path(repo_dir)