#!/usr/bin/env python
"""
Run the Petri Net Configuration UI (Optimized Version)
"""
from config_ui_optimized import create_ui

if __name__ == "__main__":
    app = create_ui()
    print("Starting Petri Net Configuration UI (Optimized Version)...")
    app.launch(share=False) 