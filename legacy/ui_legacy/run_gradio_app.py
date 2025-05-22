#!/usr/bin/env python
import os
import sys

# Make sure we're in the correct directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the Gradio app
from gradio_app import app

if __name__ == "__main__":
    app.launch() 