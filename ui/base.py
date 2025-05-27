import gradio as gr
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

class BaseUI(ABC):
    """Base class for UI components with shared functionality"""
    
    def __init__(self):
        self.state_vars = {}
    
    def create_state_var(self, name: str, default_value: Any = None) -> gr.State:
        """Create and track a Gradio state variable"""
        state_var = gr.State(default_value)
        self.state_vars[name] = state_var
        return state_var
    
    def get_state_var(self, name: str) -> gr.State:
        """Get a tracked state variable"""
        return self.state_vars.get(name)
    
    def handle_error(self, error: Exception, context: str = "") -> str:
        """Common error handling pattern"""
        error_msg = f"Error {context}: {str(error)}" if context else f"Error: {str(error)}"
        print(error_msg)  # For now, keeping print for consistency
        return error_msg
    
    def create_file_upload(self, label: str, file_types: List[str], file_count: str = "single") -> gr.File:
        """Create a standardized file upload component"""
        return gr.File(
            label=label,
            file_types=file_types,
            file_count=file_count
        )
    
    def create_button(self, text: str, variant: str = "secondary", size: str = "md") -> gr.Button:
        """Create a standardized button component"""
        return gr.Button(text, variant=variant, size=size)
    
    @abstractmethod
    def create_interface(self) -> gr.TabItem:
        """Create the main interface for this UI component"""
        pass
    
    @abstractmethod
    def setup_event_handlers(self):
        """Setup event handlers for this UI component"""
        pass 