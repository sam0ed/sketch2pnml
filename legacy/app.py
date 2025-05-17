import gradio as gr
import time  # Optional: To simulate a delay

def load_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"Error reading file: {e}"

def save_file(text_content, file_path):
    try:
        # Simulate a short delay for demonstration
        time.sleep(0.5)
        with open(file_path, 'w') as f:
            f.write(text_content)
        return f"File updated at {time.strftime('%H:%M:%S')}"
    except Exception as e:
        return f"Error writing to file: {e}"

def on_text_change(new_content, file_path):
    save_message = save_file(new_content, file_path)
    return save_message

if __name__ == "__main__":
    file_path = "example.txt"  # Replace with the actual path to your file

    # Create a dummy file if it doesn't exist
    try:
        with open(file_path, 'x') as f:
            f.write("Initial content of the file.")
    except FileExistsError:
        pass

    initial_content = load_file(file_path)

    with gr.Blocks() as demo:
        textbox = gr.Textbox(value=initial_content, label="File Content", interactive=True)
        save_status = gr.Textbox(label="Save Status", interactive=False)

        textbox.change(
            fn=on_text_change,
            inputs=[textbox, gr.State(file_path)],
            outputs=[save_status]
        )

    demo.launch()

# import gradio as gr
# import os

# FILE_PATH = "example.txt"

# def load_file_content_on_start():
#     if not os.path.exists(FILE_PATH):
#         with open(FILE_PATH, "w", encoding="utf-8") as f:
#             f.write("")
#         return ""
#     else:
#         with open(FILE_PATH, "r", encoding="utf-8") as f:
#             return f.read()

# def save_content_and_display_status(text_from_editor):
#     try:
#         with open(FILE_PATH, "w", encoding="utf-8") as f:
#             f.write(text_from_editor)
#         return "✓ Saved"
#     except Exception as e:
#         return f"Error: {str(e)}"

# with gr.Blocks(title="Real-time File Editor") as demo:
#     gr.Markdown("## File Editor\nModify the text below. Changes are saved automatically when you click outside the text area.")
    
#     with gr.Row():
#         editor_area = gr.Textbox(
#             value=load_file_content_on_start,
#             label="Edit File",
#             show_label=False,
#             lines=20,
#             interactive=True,
#             elem_id="file_editor_textbox"
#         )
        
#         status_indicator = gr.Markdown(
#             value="",
#             elem_id="save_status_display"
#         )

#     editor_area.blur(
#         fn=save_content_and_display_status,
#         inputs=editor_area,
#         outputs=status_indicator
#     )

# if __name__ == "__main__":
#     demo.launch()


