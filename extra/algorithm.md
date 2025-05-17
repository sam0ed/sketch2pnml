do the preprocessing(upscales, otsu, sharpen)

detect text
remove text pixels from the image

dialate once to close up the low resolution rectangles and circles

State 1: 
Possible circles: empty, filled
Possible rectangles: empty, filled, line
Possilbe edges: arrows 

Step 1
find all contours
detect circles in contours
detect rectangles in contours
cluster the shapes 
perform cluster selection by checking if they satisfy criteria:
- circle cluster has to be the largest available cluster. If clusters are the same size select cluster with features of the largest size.
- rectangle cluster is(research needed):
    - the largest size cluster contours of which enclose not more then 70 percent of black pixels
    - cluster whose features are the most similar to the selected circle cluster. 

create mask for detected shapes
substract mask from the image

State 2:
Possible circles: filled
Possible rectangles: filled, line
Possible edges: arrows

Step 2:
erode the image until arrows diappear
find all contours
detect circles and rectangles in contours
dialate shapes back to the state before erosion
create mask for detected shapes
substract mask from image from step 1

State 3: 
Possible circles: none
Possible rectangles: line
possible edges: arrows

<!-- Step 3:
determine key points in the image
check which key points correspond to junctions -->


State 4: 
Possible circles: none
Possible rectangles: none
possible edges: arrows




## Text removal
- detect bounding boxes
- merge overlapping bounding boxes
- detect full contours inside the bounding box
- create a mask for the full contours inside bounding box
- substract that mask from the image. 
