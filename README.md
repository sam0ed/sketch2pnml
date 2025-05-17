# Petri Net to PNML Converter

This project is a tool to convert Petri net definitions into PNML (Petri Net Markup Language) format.

## Description

(Detailed description of the project, its purpose, and features will go here.)

## Setup

1.  Clone the repository (if applicable).
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Gradio application:
```bash
python app.py
```

Then, open your web browser and navigate to the address provided by Gradio (usually `http://127.0.0.1:7860`).

## Project Structure

(Brief overview of the project structure as discussed.)

```diploma_bachelor/
├── petri_net_converter/            # Main package for your conversion logic
│   ├── elements/                   # Petri net element definitions
│   ├── templates/                  # Jinja2 templates (pnml_template.jinja2)
│   └── services/                   # Services (pnml_render_service.py)
├── app.py                          # Gradio application entry point
├── tests/                          # Unit and integration tests
├── examples/                       # Example input/output files
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Contributing

(Guidelines for contributing, if any.)

## License

(Specify the license for your project.) 