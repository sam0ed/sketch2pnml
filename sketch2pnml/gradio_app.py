import os
import gradio as gr
import base64
from endpoints.converter import fix_petri_net, render_diagram_to, render_to_graphviz, render_to_json
from pipeline.commons import here


def process_and_display():
    """Run the converter pipeline and return results for display"""
    try:
        # Ensure output directories exist
        os.makedirs(here("data/output"), exist_ok=True)
        os.makedirs(here("data/output/pipeline"), exist_ok=True)
        os.makedirs(here("data/output/visualizations"), exist_ok=True)
        
        # Run the pipeline functions
        fix_petri_net()
        render_diagram_to("pnml")
        render_diagram_to("petriobj")
        render_to_graphviz()
        render_to_json()
        
        # Get the output files
        pnml_path = here("data/output/output.pnml")
        petriobj_path = here("data/output/output.petriobj")
        json_path = here("data/output/output.json")
        png_path = here("data/output/output.png")
        gv_path = here("data/output/output.gv")
        
        # Read file contents
        with open(pnml_path, "r", encoding="utf-8") as f:
            pnml_content = f.read()
        
        with open(petriobj_path, "r", encoding="utf-8") as f:
            petriobj_content = f.read()
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        
        with open(gv_path, "r", encoding="utf-8") as f:
            gv_content = f.read()
        
        # Return all results for the tabs
        return pnml_content, petriobj_content, json_content, png_path, gv_content, pnml_path, petriobj_path, json_path, gv_path, png_path
    except Exception as e:
        error_message = f"Error during processing: {str(e)}"
        print(error_message)
        empty_path = ""
        return f"Error: {error_message}", f"Error: {error_message}", f"Error: {error_message}", None, f"Error: {error_message}", empty_path, empty_path, empty_path, empty_path, empty_path


# Create Gradio interface
with gr.Blocks(title="Petri Net Converter") as app:
    gr.Markdown("# Petri Net Sketch Converter")
    gr.Markdown("Press the Translate button to convert the working image to a Petri net")
    
    with gr.Row():
        translate_button = gr.Button("Translate", variant="primary", size="lg")
    
    # Hidden components to store file paths for downloads
    pnml_path_comp = gr.State("")
    petriobj_path_comp = gr.State("")
    json_path_comp = gr.State("")
    gv_path_comp = gr.State("")
    png_path_comp = gr.State("")
    
    with gr.Tabs():
        with gr.TabItem("PNML"):
            pnml_output = gr.Textbox(label="PNML Output", lines=20)
            pnml_download = gr.File(label="Download PNML File", interactive=False)
        
        with gr.TabItem("PetriObj"):
            petriobj_output = gr.Textbox(label="PetriObj Output", lines=20)
            petriobj_download = gr.File(label="Download PetriObj File", interactive=False)
        
        with gr.TabItem("JSON"):
            json_output = gr.Textbox(label="JSON Output", lines=20)
            json_download = gr.File(label="Download JSON File", interactive=False)
        
        with gr.TabItem("Visualization"):
            image_output = gr.Image(label="Petri Net Visualization", type="filepath")
            png_download = gr.File(label="Download PNG File", interactive=False)
        
        with gr.TabItem("GraphViz"):
            gv_output = gr.Textbox(label="GraphViz Output", lines=20)
            gv_download = gr.File(label="Download GraphViz File", interactive=False)
    
    # Connect the button to the processing function
    translate_button.click(
        fn=process_and_display,
        outputs=[
            pnml_output, petriobj_output, json_output, image_output, gv_output,
            pnml_path_comp, petriobj_path_comp, json_path_comp, gv_path_comp, png_path_comp
        ]
    ).then(
        lambda path: path,
        inputs=pnml_path_comp,
        outputs=pnml_download
    ).then(
        lambda path: path,
        inputs=petriobj_path_comp,
        outputs=petriobj_download
    ).then(
        lambda path: path,
        inputs=json_path_comp,
        outputs=json_download
    ).then(
        lambda path: path,
        inputs=gv_path_comp,
        outputs=gv_download
    ).then(
        lambda path: path,
        inputs=png_path_comp,
        outputs=png_download
    )


if __name__ == "__main__":
    app.launch()
    # pnml_path = here("../data/output/output.pnml")
    # print(pnml_path)
