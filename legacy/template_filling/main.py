import uuid
from jinja2 import Environment, FileSystemLoader
import os
import inspect
import pickle
from pipeline.commons import here



# def here(resource: str):
#     """Utils that given a relative path returns the corresponding absolute path, independently from the environment

#     Parameters
#     ----------
#     resource: str
#         The relative path of the given resource

#     Returns
#     -------
#     str
#         The absolute path of the give resource
#     """
#     stack = inspect.stack()
#     caller_frame = stack[1][0]
#     caller_module = inspect.getmodule(caller_frame)
#     return os.path.abspath(
#         os.path.join(os.path.dirname(caller_module.__file__), resource)
#     )

# class Point:
#     def __init__(self, x, y):
#         self.x = int(x) # Ensure integer coordinates if they represent pixels
#         self.y = int(y)

#         self.proximity_node = None # Placeholder for proximity node assignment
#         self.is_arrow = False # Placeholder for entry point assignment

#     def __repr__(self):
#         return f"Point({self.x}, {self.y})"

#     def __eq__(self, other):
#         if not isinstance(other, Point):
#             return NotImplemented
#         return self.x == other.x and self.y == other.y

#     def __hash__(self):
#         """Allows Point objects to be added to sets or used as dictionary keys."""
#         return hash((self.x, self.y))

# class Place:
#     def __init__(
#         self,
#         circle: tuple[int, int, int], # (x, y, radius)
#         original_detection_data=None, # Placeholder for any original detection data
#     ):
#         self.id = str(uuid.uuid4())
#         self.center = Point(circle[0], circle[1])
#         self.radius = circle[2]
#         self.center.part_of = self # Link back to the Place object

#         self.text = [] # Placeholder for any text associated with this place
#         self.original_detection_data = original_detection_data 

#         self.markers = 0 # Placeholder for markers associated with this place

#     def __repr__(self):
#         return f"Place(center={self.center}, radius={self.radius})"


if __name__ == "__main__":
    # place = Place((100, 100, 50))
    # place.markers = 3
    # place.text = []
    # print(place)

    with open(here("../../data/output/places.pkl"), "rb") as f:
        places = pickle.load(f)

    with open(here("../../data/output/transitions.pkl"), "rb") as f:
        transitions = pickle.load(f)

    with open(here("../../data/output/arcs.pkl"), "rb") as f:
        arcs = pickle.load(f)


    place = places[0]

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(here("."))) # Assumes template is in current dir
    template = env.get_template('template.jinja')

    # Render the template
    output = template.render({"places":places, "transitions":transitions, "arcs":arcs})
    print(output)
    output_file_path = here("output.pnml")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(output)