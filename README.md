# Signal Plotter 📈

A web-based tool for plotting standard singularity functions (`u(t)`, `r(t)`, `delta(t)`, etc.) and their combinations. Built with **Python** and **Streamlit**.

## Features

- **Standard Signals**:
  - `u(t)`: Unit Step
  - `r(t)`: Unit Ramp
  - `p(t)`: Unit Parabola
  - `delta(t)`: Unit Impulse
- **Complex Expressions**: Combine signals using standard math operators (e.g., `3*u(t) - 2*r(t-2)`).
- **Interactive Plotting**: Adjust time range and resolution dynamically.
- **Math-Correct**: Uses a custom `SingularityFunction` class for accurate evaluation and plotting.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/soumomo/signal-plotter.git
    cd signal-plotter
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser. Enter a signal expression in the text box (e.g., `u(t) - u(t-1)`) to see the plot.

## Examples

- **Rectangular Pulse**: `u(t) - u(t-1)`
- **Triangular Pulse**: `r(t) - 2*r(t-1) + r(t-2)`
- **Impulse Train**: `delta(t) + delta(t-1) + delta(t-2)`
