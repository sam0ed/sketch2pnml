import cv2
import numpy as np
import inspect
import functools # Needed for passing arguments to callback

class InteractiveTuner:
    def __init__(self, func, image, param_config, window_name="Tuner"):
        """
        Initializes the interactive tuner.

        Parameters:
            func (function): The computer vision function to tune.
            image (numpy.ndarray): The input image.
            param_config (dict): Configuration for each hyperparameter.
                Example: {
                    'param_name': {'default': 100, 'range': (0, 255)},
                    'float_param': {'default': 1.2, 'range': (0.1, 5.0), 'scale': 10}, # scale=10 -> trackbar 1-50
                    'odd_param': {'default': 3, 'range': (3, 7), 'constraint': 'odd'}
                }
            window_name (str): Name of the OpenCV window.
        """
        self.func = func
        self.image = image.copy() # Work on a copy
        self.original_image = image.copy() # Keep original if needed
        self.param_config = self._validate_and_fill_config(func, param_config)
        self.window_name = window_name
        self.current_values = {name: config['default'] for name, config in self.param_config.items()}
        self.trackbar_params = {} # Map trackbar name to actual param name and scale

        cv2.namedWindow(self.window_name)
        self._create_trackbars()
        self.update_display() # Initial display

    def _validate_and_fill_config(self, func, user_config):
        """Gets signature, merges with user config, sets defaults."""
        sig = inspect.signature(func)
        params = list(sig.parameters.items())[1:] # Skip first param (image)
        
        final_config = {}
        for name, param in params:
            if name not in user_config:
                # Auto-generate basic config if not provided by user
                default_val = param.default if param.default is not inspect.Parameter.empty else 0
                user_config[name] = {'default': default_val}

            config = user_config[name]
            
            # Ensure essential keys exist
            if 'default' not in config:
                 config['default'] = param.default if param.default is not inspect.Parameter.empty else 0
                 
            if 'range' not in config:
                 # Basic default range logic (improved slightly)
                 default_val = config['default']
                 if isinstance(default_val, float):
                     scale = config.get('scale', 1) # Default scale of 1 might be bad
                     if scale == 1 and default_val != 0: scale = 10**(int(np.log10(abs(default_val)))+1) # Guess a scale
                     config['scale'] = scale
                     min_val, max_val = 0, int(default_val * scale * 2) if default_val != 0 else int(10*scale)
                     config['range'] = (min_val / scale, max_val / scale) # Store float range
                 elif isinstance(default_val, int):
                     min_val, max_val = 0, max(1, default_val * 2) if default_val != 0 else 255 # Ensure max > 0
                     config['range'] = (min_val, max_val)
                 else: # Default for others
                      config['range'] = (0, 255)

            # Add scale default for floats if missing
            if isinstance(config['default'], float) and 'scale' not in config:
                 config['scale'] = 100 # Default scale factor for floats

            final_config[name] = config
            
        return final_config

    def _create_trackbars(self):
        """Creates trackbars based on param_config."""
        for name, config in self.param_config.items():
            min_val, max_val = config['range']
            default_val = config['default']
            scale = config.get('scale', 1)
            
            trackbar_name = name
            initial_pos = default_val
            trackbar_max = max_val

            if isinstance(default_val, float):
                initial_pos = int(default_val * scale)
                trackbar_max = int(max_val * scale)
                # Handle min_val for floats slightly differently if needed, assuming min>=0 here
                if min_val * scale < 0: print(f"Warning: Min value < 0 for float {name}, trackbar starts at 0")

            elif isinstance(default_val, int):
                 # Ensure range makes sense for int trackbar
                 if not (isinstance(min_val, int) and isinstance(max_val, int)):
                     print(f"Warning: Non-integer range for int param {name}. Using defaults.")
                     min_val, max_val = 0, 255
                 # Trackbar min is always 0, adjust default and max accordingly
                 if min_val != 0:
                     initial_pos = max(0, default_val - min_val) # Position relative to min
                     trackbar_max = max(1, max_val - min_val) # Range width, ensure > 0
                 else:
                      initial_pos = default_val
                      trackbar_max = max(1, max_val) # Ensure max > 0


            # Store mapping info for the callback
            self.trackbar_params[trackbar_name] = {'name': name, 'scale': scale, 'min_val': min_val, 'constraint': config.get('constraint')}

            # Use functools.partial to pass the trackbar_name to the callback
            callback = functools.partial(self._on_trackbar_change, trackbar_name)
            
            cv2.createTrackbar(trackbar_name, self.window_name, initial_pos, trackbar_max, callback)


    def _on_trackbar_change(self, trackbar_name, value):
        """Callback function for trackbar changes."""
        param_info = self.trackbar_params[trackbar_name]
        param_name = param_info['name']
        scale = param_info['scale']
        min_val = param_info['min_val']
        constraint = param_info.get('constraint')

        # Apply scaling for floats or adjust for non-zero min_val
        actual_value = value
        if scale != 1: # Likely a float
            actual_value = value / scale
        elif min_val != 0: # Handle integer offset if min_val wasn't 0
             actual_value = value + min_val

        # Apply constraints
        if constraint == 'odd':
            actual_value = int(actual_value) | 1 # Force odd
            # Optional: Adjust slider position if value changed due to constraint
            # This part is tricky with standard cv2 trackbars - might jump
            
        elif constraint == 'even':
             actual_value = int(actual_value) & ~1 # Force even

        # Add more constraints as needed (e.g., 'multiple_of_5')

        self.current_values[param_name] = actual_value
        self.update_display()

    def update_display(self):
        """Calls the function and updates the image display."""
        try:
            # Filter out incompatible args if necessary (though inspect should handle it)
            valid_args = {k: v for k, v in self.current_values.items()}
            result = self.func(self.original_image, **valid_args) # Use original image each time
            
            # Display current values in the window title (optional)
            title_str = self.window_name + " | "
            title_str += ", ".join([f"{name}={val:.2f}" if isinstance(val, float) else f"{name}={val}" 
                                    for name, val in self.current_values.items()])
            cv2.setWindowTitle(self.window_name, title_str)

            cv2.imshow(self.window_name, result)
        except Exception as e:
            print(f"Error during function execution: {e}")
            # Optionally display an error image or message
            error_img = np.zeros_like(self.original_image)
            cv2.putText(error_img, "Error executing function", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(self.window_name, error_img)


    def run(self):
        """Runs the main loop waiting for user input."""
        print("Press ESC to quit.")
        while True:
            # The callback handles updates, so waitKey just checks for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC key
                break
            # Check if window was closed manually
            try:
                 # Property is available in recent OpenCV versions
                 if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                      break
            except Exception: # Fallback if property not available
                 pass # Continue loop until ESC

        cv2.destroyAllWindows()
        print("Tuner finished. Final parameters:")
        print(self.current_values)
        return self.current_values


# --- DEMO ---
def interactive_tuner_demo_enhanced():
    image_path = "./assets/local/simple_petri_2.png" # Update path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error loading image!")
        exit(1)
        
    # Apply Gaussian Blur first for better Hough Circle detection
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)


    # ## Canny Edge Detector Example
    def canny_filter_wrapper(img, threshold1=100, threshold2=200, apertureSize=3):
        # apertureSize must be odd. The tuner will handle enforcing this.
        return cv2.Canny(img, threshold1, threshold2, apertureSize=apertureSize)

    canny_config = {
        "threshold1": {'default': 50, 'range': (0, 255)},
        "threshold2": {'default': 150, 'range': (0, 255)},
        "apertureSize": {'default': 3, 'range': (3, 7), 'constraint': 'odd'} # Range 3-7, only odd values used
    }
    
    # print("--- Starting Canny Tuner ---")
    # canny_tuner = InteractiveTuner(canny_filter_wrapper, image, canny_config, window_name="Canny Tuner")
    # final_canny_params = canny_tuner.run()


    ## Hough Transform Example
    def hough_circle_visualizer_wrapper(img, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
        # Ensure integer parameters are passed as integers
        minDist_int = int(minDist)
        param1_int = int(param1)
        param2_int = int(param2)
        minRadius_int = int(minRadius)
        maxRadius_int = int(maxRadius)

        # Detect circles using the blurred image for better results
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist_int,
                                     param1=param1_int, param2=param2_int,
                                     minRadius=minRadius_int, maxRadius=maxRadius_int)
        
        # Create a color image for display from the original gray image
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Use the input img which is blurred
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2) # outer circle
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)      # center
        
        return output

    hough_config = {
        "dp": {'default': 1.0, 'range': (0.1, 5.0), 'scale': 10}, # Trackbar 1-50 represents 0.1-5.0
        "minDist": {'default': 20, 'range': (1, 100)},
        "param1": {'default': 50, 'range': (1, 200)}, # Upper threshold for Canny
        "param2": {'default': 30, 'range': (1, 100)}, # Accumulator threshold
        "minRadius": {'default': 0, 'range': (0, 100)},
        "maxRadius": {'default': 0, 'range': (0, 200)}
    }
    
    print("\n--- Starting Hough Circle Tuner (using blurred image) ---")
    # Using blurred_image as input for Hough
    hough_tuner = InteractiveTuner(hough_circle_visualizer_wrapper, blurred_image, hough_config, window_name="Hough Circle Tuner") 
    final_hough_params = hough_tuner.run()

# Run the enhanced demo
# Ensure you have an image at the specified path or update it.
# If running in an environment where OpenCV windows don't work well (like some remote notebooks), this might fail.
interactive_tuner_demo_enhanced()