---
title: Petri Net Converter Suite
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Petri Net Converter Suite

A Gradio-based application that converts hand-drawn Petri net sketches into digital formats using computer vision and OCR. 

## Demo

🚀 **[Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/sam0ed/sketch2pnml)**

## What it does

The application processes images of Petri net diagrams and automatically:
- Detects places, transitions, and arcs using computer vision
- Extracts text labels using OCR
- Converts the detected elements into standard formats (PNML, JSON, GraphViz, etc.)
- Provides visual feedback of the detection process

## Setup

### Requirements

Python 3.8+ and the dependencies listed in `requirements.txt`.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd petri-net-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, for full functionality):
```bash
cp env.example .env
# Edit .env and add your API keys:
# GROQ_API_KEY=your_groq_api_key_here
# ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

4. Run the application:
```bash
python app.py
```

The application will start on `http://localhost:7860`.

## Usage

1. **Image Processor Tab**: Upload an image of a Petri net sketch and a YAML configuration file, then click "Process"
2. **Configuration Editor Tab**: Create or modify YAML configuration files for processing parameters
3. **Petri Net Converter Tab**: Convert processed results into various output formats

## Supported Formats

- **Input**: PNG, JPG, JPEG images + YAML configuration files
- **Output**: PNML (XML), PetriObj, JSON, GraphViz DOT, PNG visualizations

## License

MIT License 