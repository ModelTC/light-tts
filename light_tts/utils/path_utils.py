import os


def trans_relative_to_abs_path(base_dir, dest_path):
    if dest_path.startswith("."):
        abs_path = os.path.normpath(os.path.join(base_dir, dest_path))
    else:
        abs_path = dest_path
    return abs_path

