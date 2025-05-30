import os

# Get the project root directory (parent of config directory)
_CONFIG_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CONFIG_DIR_PATH)

# Base directories
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

# Data subdirectories
DEMOS_DIR = os.path.join(DATA_DIR, "demos")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
WORKING_DIR = os.path.join(DATA_DIR, "working")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
PIPELINE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pipeline")

# Working files
WORKING_IMAGE_PATH = os.path.join(WORKING_DIR, "working_image.png")
WORKING_CONFIG_PATH = os.path.join(WORKING_DIR, "working_config.yaml")
CONFIG_DOWNLOAD_PATH = os.path.join(WORKING_DIR, "config_download.yaml")

# Output files
OUTPUT_PNML_PATH = os.path.join(OUTPUT_DIR, "output.pnml")
OUTPUT_PETRIOBJ_PATH = os.path.join(OUTPUT_DIR, "output.petriobj")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "output.json")
OUTPUT_PNG_PATH = os.path.join(OUTPUT_DIR, "output.png")
OUTPUT_GV_PATH = os.path.join(OUTPUT_DIR, "output.gv")

# Pipeline data files
PLACES_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "places.pkl")
TRANSITIONS_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "transitions.pkl")
ARCS_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "arcs.pkl")
PLACES_FIXED_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "places_fixed.pkl")
TRANSITIONS_FIXED_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "transitions_fixed.pkl")
ARCS_FIXED_PKL_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "arcs_fixed.pkl")

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
SUPPORTED_CONFIG_EXTENSIONS = ('.yaml', '.yml')

# Default configuration
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "algorithm_config", "config.yaml")

def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    directories = [
        OUTPUT_DIR,
        VISUALIZATIONS_DIR,
        PIPELINE_OUTPUT_DIR,
        DEMOS_DIR,
        WORKING_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_working_image_path(extension=".png"):
    """Get the working image path with the specified extension."""
    return os.path.join(WORKING_DIR, f"working_image{extension}")

def get_working_config_path(extension=".yaml"):
    """Get the working config path with the specified extension."""
    return os.path.join(WORKING_DIR, f"working_config{extension}")

def get_visualization_path(name):
    """Get the path for a visualization file."""
    return os.path.join(VISUALIZATIONS_DIR, f"{name}.png")

def get_output_file_path(filename):
    """Get the path for an output file."""
    return os.path.join(OUTPUT_DIR, filename)

def get_config_download_path():
    """Get the path for config download file."""
    return CONFIG_DOWNLOAD_PATH
