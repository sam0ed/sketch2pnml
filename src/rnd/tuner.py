import jedi
import cv2

def get_canny_parameters_info():
    """
    Uses jedi to get comprehensive information about the parameters of the cv2.Canny function.

    Returns:
        list: A list of dictionaries, where each dictionary contains all available 
              information about a parameter.
    """
    code = """import cv2
cv2.Canny("""
    
    script = jedi.Script(code)
    signatures = script.get_signatures(line=2, column=len("cv2.Canny("))
    
    parameter_info = []
    if signatures:
        signature = signatures[0]
        
        # Get signature-level information
        signature_info = {
            'name': getattr(signature, 'name', ''),
            'docstring': getattr(signature, 'docstring', ''),
            'description': getattr(signature, 'description', '')
        }
        parameter_info.append({"signature_info": signature_info})
        
        # Get parameter-specific information
        for param in signature.params:
            # Explore all available attributes
            param_attrs = dir(param)
            
            param_info = {
                'name': param.name,
                'description': getattr(param, 'description', ''),
                'kind': getattr(param, 'kind', ''),
                'type_annotation': getattr(param, 'annotation', ''),
                'default': getattr(param, 'default', None),
                'has_default': getattr(param, 'has_default', None),
                'infer_default': getattr(param, 'infer_default', None),
                'all_attributes': param_attrs
            }
            parameter_info.append(param_info)
    
    return parameter_info

if __name__ == "__main__":
    parameters_info = get_canny_parameters_info()
    
    # Print signature information
    if parameters_info and "signature_info" in parameters_info[0]:
        sig_info = parameters_info[0]["signature_info"]
        print(f"Function: {sig_info['name']}")
        print(f"Documentation: {sig_info['docstring']}")
        print("\nParameters:")
    
    # Print parameter information
    for info in parameters_info[1:]:  # Skip the signature info
        print(f"\n{info['name']}:")
        for key, value in info.items():
            if key != 'name' and key != 'all_attributes' and value:
                print(f"  {key}: {value}")