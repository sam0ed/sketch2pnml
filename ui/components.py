import gradio as gr
import yaml
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

def create_component(param_def: Dict) -> Any:
    """Create appropriate Gradio component based on parameter definition"""
    comp_type = param_def.get("type", "text")
    
    if comp_type == "slider":
        return gr.Slider(
            label=param_def["label"],
            minimum=param_def.get("min", 0),
            maximum=param_def.get("max", 1),
            step=param_def.get("step", 0.1),
            value=param_def.get("default")
        )
    elif comp_type == "checkbox":
        return gr.Checkbox(label=param_def["label"], value=param_def.get("default", False))
    elif comp_type == "number":
        return gr.Number(label=param_def["label"], value=param_def.get("default", 0), precision=param_def.get("precision", 0))
    elif comp_type == "password":
        return gr.Textbox(label=param_def["label"], value=param_def.get("default", ""), type="password")
    else:  # text is the default
        return gr.Textbox(label=param_def["label"], value=param_def.get("default", ""))

def get_from_nested_dict(data: Dict, path: str, default: Any = None) -> Any:
    """Access nested dictionary using dot notation path"""
    keys = path.split('.')
    result = data
    
    for key in keys:
        if not isinstance(result, dict) or key not in result:
            return default
        result = result[key]
    
    return result

def set_in_nested_dict(data: Dict, path: str, value: Any) -> Dict:
    """Set value in nested dictionary using dot notation path"""
    keys = path.split('.')
    current = data
    
    # Navigate to the final dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    return data

def load_config(config_path: str) -> tuple:
    """Load configuration from yaml file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f), True, f"Successfully loaded {config_path}"
    except Exception as e:
        return {}, False, f"Error loading config: {str(e)}"

def save_config(config: Dict, config_path: str, custom_filename: Optional[str] = None) -> tuple:
    """Save configuration to yaml file with timestamp or custom name"""
    try:
        base_dir = os.path.dirname(config_path)
        
        if custom_filename:
            # Use custom filename if provided
            if not custom_filename.endswith(('.yaml', '.yml')):
                custom_filename += '.yaml'
            new_path = os.path.join(base_dir, custom_filename)
        else:
            # Otherwise use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.splitext(config_path)[0]
            ext = os.path.splitext(config_path)[1]
            new_path = f"{base_path}_{timestamp}{ext}"
        
        with open(new_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return True, f"Configuration saved to {new_path}", new_path
    except Exception as e:
        return False, f"Error saving config: {str(e)}", None

def create_download_file(content: str, filename: str, file_extension: str, output_dir: str) -> str:
    """Create a temporary file with the given content for download"""
    try:
        # Create filename with proper extension
        if not filename.endswith(file_extension):
            filename = f"{filename}{file_extension}"
        
        file_path = os.path.join(output_dir, filename)
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return file_path
    except Exception as e:
        print(f"Error creating download file: {e}")
        return "" 