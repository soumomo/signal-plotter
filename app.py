import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from signal_lib import SingularityFunction, TimeVariable, u, r, p, delta, rect, t

st.set_page_config(page_title="Signal Plotter", layout="wide")

st.title("Signal Graph Plotter")
st.markdown("Plot standard singularity functions like Step `u(t)`, Ramp `r(t)`, Parabola `p(t)`, and Impulse `delta(t)`.")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    t_start = st.number_input("Start Time", value=-5.0, step=1.0)
    t_end = st.number_input("End Time", value=10.0, step=1.0)
    resolution = st.slider("Resolution", min_value=100, max_value=5000, value=1000)
    
    st.markdown("---")
    st.markdown("### Help")
    st.markdown("""
    - `u(t)`: Unit Step
    - `r(t)`: Unit Ramp
    - `p(t)`: Unit Parabola
    - `delta(t)`: Unit Impulse
    - `rect(t)`: Rectangular Pulse (width 1)
    
    **Examples:**
    - `u(t)`
    - `rect(t)` (Pulse from -0.5 to 0.5)
    - `rect(t-2)` (Pulse from 1.5 to 2.5)
    - `3*r(t) - 3*r(t-2) - 6*u(t-2)`
    """)

# Main Area
expression = st.text_input("Enter Signal Expression (function of 't')", value="rect(t)")

if expression:
    try:
        # Safe evaluation environment
        safe_dict = {
            'u': u, 'r': r, 'p': p, 'delta': delta, 'rect': rect, 't': t,
            'SingularityFunction': SingularityFunction,
            'np': np
        }
        
        # Evaluate the expression
        func = eval(expression, {"__builtins__": None}, safe_dict)
        
        if isinstance(func, SingularityFunction):
            fig = func.plot(t_start=t_start, t_end=t_end, resolution=resolution, title=f"f(t) = {expression}")
            st.pyplot(fig)
        else:
            st.error("Expression must evaluate to a SingularityFunction object.")
            
    except Exception as e:
        st.error(f"Error evaluating expression: {e}")
