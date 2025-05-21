from pipeline.workflow import recognize_graph
from pipeline.commons import here
import os
import jinja2
import pickle
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from collections import Counter, defaultdict
from typing import List, Tuple
from pipeline.models import Place, Transition, Arc
from groq import Groq
import os
import base64

def process_elements(places: List[Place], transitions: List[Transition], arcs: List[Arc]) -> Tuple[List[Place], List[Transition], List[Arc]]:
    # Process places to remove those with no connected arcs
    places_to_remove = set()
    for place in places:
        if not any(arc.source == place or arc.target == place for arc in arcs):
            places_to_remove.add(place)
    new_places = [p for p in places if p not in places_to_remove]
    # Remove arcs connected to removed places
    arcs_after_places = [arc for arc in arcs if arc.source not in places_to_remove and arc.target not in places_to_remove]
    
    # Process transitions to remove those with less than two connected arcs
    transitions_to_remove = set()
    arcs_to_remove = set()
    for transition in transitions:
        connected_arcs = [arc for arc in arcs_after_places if arc.source == transition or arc.target == transition]
        if len(connected_arcs) < 2:
            transitions_to_remove.add(transition)
            arcs_to_remove.update(connected_arcs)
    new_transitions = [t for t in transitions if t not in transitions_to_remove]
    arcs_after_transitions = [arc for arc in arcs_after_places if arc not in arcs_to_remove]
    
    # Adjust transitions to have both incoming and outgoing arcs
    for transition in new_transitions:
        connected_arcs = [arc for arc in arcs_after_transitions if arc.source == transition or arc.target == transition]
        outgoing = sum(1 for arc in connected_arcs if arc.source == transition)
        incoming = sum(1 for arc in connected_arcs if arc.target == transition)
        
        if outgoing == 0 and incoming >= 1:
            # Flip one incoming arc to outgoing
            for arc in connected_arcs:
                if arc.target == transition:
                    arc.source, arc.target = arc.target, arc.source
                    break
        elif incoming == 0 and outgoing >= 1:
            # Flip one outgoing arc to incoming
            for arc in connected_arcs:
                if arc.source == transition:
                    arc.source, arc.target = arc.target, arc.source
                    break
    
    return new_places, new_transitions, arcs_after_transitions



def fix_petri_net():
    """Method that checks for all the errors in the petri net, logs the errors and applies fixes, if readily available."""
    with open(here("../data/output/pipeline/places.pkl"), "rb") as f:
        places = pickle.load(f)
    with open(here("../data/output/pipeline/transitions.pkl"), "rb") as f:
        transitions = pickle.load(f)
    with open(here("../data/output/pipeline/arcs.pkl"), "rb") as f:
        arcs = pickle.load(f)

    ### Remove duplicate ids across places, transitions and arcs
    all_ids = []
    all_ids.extend(place.id for place in places)
    all_ids.extend(transition.id for transition in transitions) 
    all_ids.extend(arc.id for arc in arcs)

    id_duplicates = [id for id, count in Counter(all_ids).items() if count > 1]

    for duplicate_id in id_duplicates:
        duplicate_elements = []
        duplicate_elements.extend([place for place in places if place.id == duplicate_id])
        duplicate_elements.extend([transition for transition in transitions if transition.id == duplicate_id]) 
        duplicate_elements.extend([arc for arc in arcs if arc.id == duplicate_id])
        print(f"Duplicate ID {duplicate_id} found in elements: {duplicate_elements}")

    ### Remove cycles, remove same type connections
    arcs = [arc for arc in arcs if type(arc.source) != type(arc.target)]
    ### Fix weights if any are less than 1
    for arc in arcs:
        if arc.weight < 1:
            print(f"Arc found with weight less than 1: {arc}")
            print(f"Applying fix to set the weight to 1")
            arc.weight = 1

    ### find arcs in arcs list, that have same source and the same target, and merge them into one arc, with the sum of the weights
    # Group arcs by their source and target
    arc_groups = defaultdict(list)
    for arc in arcs:  # Create a copy of the list to safely modify original
        key = (arc.source.id, arc.target.id)
        arc_groups[key].append(arc)

    # For each group of arcs with same source/target, merge them
    for (source_id, target_id), group in arc_groups.items():
        if len(group) > 1:
            print(f"Found {len(group)} parallel arcs between same source and target: {source_id} -> {target_id}")
            
            total_weight = sum(arc.weight for arc in group)
            
            merged_arc = group[0]
            merged_arc.weight = total_weight
            
            # Remove other arcs from the original list
            for arc in group[1:]:
                if arc in arcs:
                    arcs.remove(arc)


    ### There should be no hanging places, every place must have at least one arc
    ### Every transition must have at least one input and one output arc
    
    # places, transitions, arcs = sanitize_petri_net(places, transitions, arcs)
    places, transitions, arcs = process_elements(places, transitions, arcs)

    print(f"len(places): {len(places)}")
    print(f"len(transitions): {len(transitions)}")
    print(f"len(arcs): {len(arcs)}")
    # for place in places:
    #     print(place)
    # for transition in transitions:
    #     print(transition)
    # for arc in arcs:
    #     print(arc)


    ### save results as pickles 
    output_dir = here("../data/output")
    os.makedirs(output_dir, exist_ok=True)
    with open(here("../data/output/pipeline/places_fixed.pkl"), "wb") as f:
        pickle.dump(places, f)
    with open(here("../data/output/pipeline/transitions_fixed.pkl"), "wb") as f:
        pickle.dump(transitions, f)
    with open(here("../data/output/pipeline/arcs_fixed.pkl"), "wb") as f:
        pickle.dump(arcs, f)
    




def run_and_save_pipeline(config_path: str, image_path: str):
    result = recognize_graph(image_path, config_path)
    
    # Access the results
    places = result["places"]
    transitions = result["transitions"]
    arcs = result["arcs"]
    
    # Saving logic
    output_dir = here("../data/output")
    os.makedirs(output_dir, exist_ok=True)
    for name, img in result["visualizations"].items():
        img.save(f"{output_dir}/visualizations/{name}.png")
    with open(f"{output_dir}/pipeline/places.pkl", "wb") as f:
        pickle.dump(places, f)
    with open(f"{output_dir}/pipeline/transitions.pkl", "wb") as f:
        pickle.dump(transitions, f)
    with open(f"{output_dir}/pipeline/arcs.pkl", "wb") as f:
        pickle.dump(arcs, f)

    print(f"Recognition complete. Found {len(places)} places, {len(transitions)} transitions, and {len(arcs)} arcs.")


def render_diagram_to(file_type: str):
    """Method that renders elements into the final pnml string

    Parameters
    ----------
    type: str
        the type of template to use
    Returns
    -------
    str
        The string representing the final pnml model
    """
    with open(here("../data/output/pipeline/places_fixed.pkl"), "rb") as f:
        places = pickle.load(f)
    with open(here("../data/output/pipeline/transitions_fixed.pkl"), "rb") as f:
        transitions = pickle.load(f)
    with open(here("../data/output/pipeline/arcs_fixed.pkl"), "rb") as f:
        arcs = pickle.load(f)

    template_loader = jinja2.FileSystemLoader(
        searchpath=here("../data/templates/")
    )
    template_env = jinja2.Environment(loader=template_loader)

    if file_type == "pnml":
        template = template_env.get_template(f"template.{file_type}.jinja")
        output_text = template.render({"places": places, "transitions": transitions, "arcs": arcs})

        output_file_path = here(f"../data/output/output.{file_type}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(output_text)
    
    elif file_type == "petriobj":
        template = template_env.get_template(f"template.{file_type}.jinja")

        place_to_index = {place.id: index for index, place in enumerate(places)}
        transition_to_index = {transition.id: index for index, transition in enumerate(transitions)}


        output_text = template.render({"places": places, "transitions": transitions, "arcs": arcs, "place_to_index": place_to_index, "transition_to_index": transition_to_index})

        output_file_path = here(f"../data/output/output.{file_type}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(output_text)

    else:
        raise ValueError(f"Invalid file type: {file_type}")

    return output_text

def render_to_graphviz():
    net, im, fm = pm4py.read_pnml(here("../data/output/output.pnml"))

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz, here('../data/output/output.gv'))

    pm4py.save_vis_petri_net(net, im, fm, here('../data/output/output.png'))
    # pm4py.view_petri_net(net, im, fm, format='../data/output/converted_petri_net.gv')
    # # pm4py.view_petri_net(net, im, fm, format='gv')

def render_to_json():
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    image_path = here("../data/working_image.png")

    base64_image = encode_image(image_path)

    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": "{\n  \"places\": [\n    {\"id\": \"string\", \"tokens\": \"integer\"}\n  ],\n  \"transitions\": [\n    {\"id\": \"string\", \"delay\": \"number_or_string\"}\n  ],\n  \"arcs\": [\n    {\"source\": \"string\", \"target\": \"string\", \"weight\": \"integer\"}\n  ]\n}"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Take image of a Petri net as input and provide the textual representation of the graph in json format, according to this json template."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                ]
            }
        ],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    output_file_path = here("../data/output/output.json")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(completion.choices[0].message.content)



if __name__ == "__main__":
    # run_and_save_pipeline(config_path=here("../data/config.yaml"), image_path=here("../data/local/mid_petri_2.png"))

    fix_petri_net()
    ## the next steps should be done in parallel
    render_diagram_to("pnml")
    render_diagram_to("petriobj")
    render_to_graphviz()
    render_to_json()