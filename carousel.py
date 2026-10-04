import os
import streamlit.components.v1 as components

# Point to the dist folder where Vite builds the component
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "carousel_component/dist")

_carousel_component = components.declare_component(
    "minimal_carousel",
    path=_BUILD_DIR
)

def minimal_carousel(cards, key=None):
    """
    cards: list of dicts with id, title, value, color, icon
    """
    component_value = _carousel_component(
        cards=cards,
        key=key,
        default=None
    )
    return component_value
