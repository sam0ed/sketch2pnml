#!/usr/bin/env python
"""
Run the Merged Petri Net Converter Suite
"""
import os
import sys

# Make sure we're in the correct directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the merged app
from merged_app import app

if __name__ == "__main__":
    print("Starting Petri Net Converter Suite...")
    app.launch() 