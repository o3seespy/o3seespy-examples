
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
TEMP_PATH = ROOT_DIR + "temp/"
MODULE_DATA_PATH = ROOT_DIR + "data/"

# add project specific paths here

try:
    import user_paths as up  # create a file in the root directory called user_paths.py and set the following paths
    if hasattr(up, "OP_PATH"):  # Location to save output files
        OP_PATH = up.OP_PATH
    else:
        OP_PATH = "<set path to OP_PATH>"
except ModuleNotFoundError:
    OP_PATH = TEMP_PATH
    if not os.path.exists(OP_PATH):
        os.makedirs(OP_PATH)
