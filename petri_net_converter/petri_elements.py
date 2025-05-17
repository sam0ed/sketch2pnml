from dataclasses import dataclass, field
from typing import List

from jinja2 import Environment, BaseLoader


class Element:
    """Parent class for all the elements that can be put within a Petri diagram.

    Parameters
    ----------
    id : str
        Unique identifier of the Petri net Element
    """
    def __init__(
        self,
        id: str,
    ):
        self.id = id
        self.name = []
        self.jinja_environment = Environment(loader=BaseLoader())
        self.incoming = []
        self.outgoing = []

    def render_element(self):
        """Returns the xml string associated to this kind of element"""

    def get_name(self):
        """Returns the text of the element as a string"""
        return " ".join([text.text for text in self.name])

    def render_shape(self):
        pass


class Place(Element):
    """Represents a Petri net place.

    Parameters
    ----------
    id : str
        Unique identifier of the BPMN Element.
    
    center_x : float
        X coordinate of the center of the place.
    center_y : float
        Y coordinate of the center of the place.
    radius : float
        Radius of the place.
    """
    def __init__(
        self,
        id: str,
        center_x: float,
        center_y: float,
        radius: float,
    ):
        super(self).__init__(id)

    def render_element(self):
        pass


class Transition(Element):
    """Represents a Petri net transition.

    Parameters
    ----------
    id : str
        Unique identifier of the BPMN Element.
    
    center_x : float
        X coordinate of the center of the place.
    center_y : float
        Y coordinate of the center of the place.
    radius : float
        Radius of the place.
    """
    def __init__(
        self,
        id: str,
        center: tuple,
        radius: float,
    ):
        super(self).__init__(id)

    def render_element(self):
        pass


class TextAnnotation(Element):
    """Represents a BPMN Text Association.

        Parameters
        ----------
        id : str
            Unique identifier of the BPMN Element.
        bbox : list of float
            The bounding box of the text annotation.
        text : str
            The text of the annotation.
    """
    def __init__(self, id: str):
        super(self).__init__(id)

    def render_element(self):
        pass


@dataclass()
class Diagram:
    """Represents a BPMN Diagram which contains all the information used to write the xml file.

    Parameters
    ----------
    id : str
        Unique identifier of the BPMN Element.
    definition_id : str
        Unique identifier of the BPMN Definition tag.
    processes : list of Process
        The list of Process to include in the xml file.
    collaboration : Collaboration
        The collaboration object to include in the xml file.
    """
    id: str
    definition_id: str
    
