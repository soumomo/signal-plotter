import numpy as np
import matplotlib.pyplot as plt

# --- 1. Time Variable Helper ---
class TimeVariable:
    """
    Represents the linear time argument (k*t + c).
    Allows writing expressions like 't - 2', '2*t', '5 - t'.
    """
    def __init__(self, k=1.0, c=0.0):
        self.k = k
        self.c = c

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return TimeVariable(self.k, self.c + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return TimeVariable(self.k, self.c - other)
        return NotImplemented

    def __rsub__(self, other):
        # other - self = other - (k*t + c) = -k*t + (other - c)
        if isinstance(other, (int, float)):
            return TimeVariable(-self.k, other - self.c)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TimeVariable(self.k * other, self.c * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return TimeVariable(-self.k, -self.c)

# Create the global 't' variable instance
t = TimeVariable(k=1.0, c=0.0)


# --- 2. Main Function Class ---
class SingularityFunction:
    """
    Represents a composite singularity function.
    Supports addition, subtraction, scalar multiplication, AND function multiplication.
    
    Internal Structure:
    - continuous_terms: A list of 'Product Terms'. 
      Each Product Term is a dict: {'mag': float, 'components': [list of basic func dicts]}
      f(t) = Sum( mag_i * Product( component_func_j(t) ) )
      
    - impulses: A list of dicts {'mag', 't0', 'k'}. 
      Impulses are kept separate as they are singularities.
    """
    def __init__(self, terms=None, impulses=None):
        self.continuous_terms = terms if terms is not None else []
        self.impulses = impulses if impulses is not None else []

    def _evaluate_scalar(self, t_scalar):
        """ Helper to evaluate continuous part at a single point (for sampling impulses). """
        val = 0.0
        # t_scalar is a float, but our math uses numpy arrays usually. 
        # We wrap it in a 1-element array or just use float math.
        # Using simple float math for speed/safety here.
        
        for term in self.continuous_terms:
            term_val = term['mag']
            for comp in term['components']:
                k = comp['k']
                t0 = comp['t0']
                func_type = comp['type']
                
                arg = k * t_scalar - t0
                
                # Heaviside
                step = 1.0 if arg >= 0 else 0.0
                
                if func_type == 'step':
                    term_val *= step
                elif func_type == 'ramp':
                    term_val *= (arg * step)
                elif func_type == 'parabola':
                    term_val *= ((arg**2) * step)
            
            val += term_val
        return val

    def evaluate(self, t_vector):
        """ Evaluates continuous parts over a numpy time vector. """
        # y starts at 0
        y = np.zeros_like(t_vector, dtype=float)
        
        for term in self.continuous_terms:
            # specific term magnitude
            term_y = np.full_like(t_vector, term['mag'], dtype=float)
            
            # Multiply by each component in this product term
            for comp in term['components']:
                k = comp['k']
                t0 = comp['t0']
                func_type = comp['type']
                
                arg = k * t_vector - t0
                mask = (arg >= 0).astype(float)
                
                if func_type == 'step':
                    term_y *= mask
                elif func_type == 'ramp':
                    term_y *= (arg * mask)
                elif func_type == 'parabola':
                    term_y *= ((arg**2) * mask)
            
            y += term_y
                
        return y

    def plot(self, t_start=-2, t_end=10, resolution=1000, title="Singularity Function Plot"):
        t_vals = np.linspace(t_start, t_end, resolution)
        y = self.evaluate(t_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find discontinuities
        discontinuities = set()
        for term in self.continuous_terms:
            for comp in term['components']:
                t_disc = comp['t0'] / comp['k']
                if t_start <= t_disc <= t_end:
                    discontinuities.add(t_disc)
        
        # Sort discontinuities
        disc_points = sorted(list(discontinuities))
        
        # Plot segments between discontinuities
        if not disc_points:
            ax.plot(t_vals, y, label='Continuous', linewidth=2, color='royalblue')
        else:
            # Add start and end boundaries
            boundaries = [t_start] + disc_points + [t_end]
            
            for i in range(len(boundaries) - 1):
                # Create segment slightly inside the boundaries to avoid overlap
                eps = (t_end - t_start) * 1e-6
                seg_start = boundaries[i] + (eps if i > 0 else 0)
                seg_end = boundaries[i+1] - (eps if i < len(boundaries) - 2 else 0)
                
                mask = (t_vals >= seg_start) & (t_vals <= seg_end)
                if np.any(mask):
                    ax.plot(t_vals[mask], y[mask], linewidth=2, color='royalblue')
            
            # Draw vertical lines at discontinuities
            for t_disc in disc_points:
                # Evaluate just before and after
                y_before = self.evaluate(np.array([t_disc - 1e-9]))[0]
                y_after = self.evaluate(np.array([t_disc + 1e-9]))[0]
                
                if abs(y_before - y_after) > 1e-6:
                    ax.plot([t_disc, t_disc], [y_before, y_after], 
                           linewidth=2, color='royalblue', linestyle='-')
        
        # Add legend label manually
        ax.plot([], [], label='Continuous', linewidth=2, color='royalblue')
        
        # Plot Impulses
        for impulse in self.impulses:
            mag = impulse['mag']
            t0 = impulse['t0']
            k = impulse['k']
            
            t_loc = t0 / k
            eff_mag = mag / abs(k)
            
            if t_start <= t_loc <= t_end:
                base_y = self.evaluate(np.array([t_loc]))[0]
                
                # Visual scaling
                head_width = (t_end - t_start) * 0.02
                y_range = max(y.max(), 1) - min(y.min(), 0)
                if y_range == 0: y_range = 1
                head_length = y_range * 0.1
                
                color = 'red' if eff_mag > 0 else 'purple'
                ax.arrow(t_loc, base_y, 0, eff_mag, 
                         head_width=head_width, head_length=head_length, 
                         fc=color, ec=color, length_includes_head=True, label='Impulse')
                
                ax.text(t_loc + head_width, base_y + eff_mag, 
                        f'{eff_mag:.2g}$\delta$', 
                        verticalalignment='bottom' if eff_mag > 0 else 'top')

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)")
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        return fig

    # --- Operator Overloading ---

    def __add__(self, other):
        if isinstance(other, SingularityFunction):
            return SingularityFunction(
                terms=self.continuous_terms + other.continuous_terms,
                impulses=self.impulses + other.impulses
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SingularityFunction):
            return self + (other * -1)
        return NotImplemented

    def __mul__(self, other):
        # 1. Scalar Multiplication
        if isinstance(other, (int, float)):
            # Scale continuous terms
            new_terms = []
            for term in self.continuous_terms:
                new_term = {
                    'mag': term['mag'] * other,
                    'components': [c.copy() for c in term['components']]
                }
                new_terms.append(new_term)
            
            # Scale impulses
            new_impulses = []
            for imp in self.impulses:
                new_imp = imp.copy()
                new_imp['mag'] *= other
                new_impulses.append(new_imp)
                
            return SingularityFunction(terms=new_terms, impulses=new_impulses)

        # 2. Function Multiplication (Convolution of terms essentially, but purely algebraic here)
        elif isinstance(other, SingularityFunction):
            new_sf = SingularityFunction()
            
            # A. Continuous * Continuous
            # Distribute: (TermA1 + TermA2) * (TermB1 + TermB2)
            for term1 in self.continuous_terms:
                for term2 in other.continuous_terms:
                    new_mag = term1['mag'] * term2['mag']
                    # Combine the lists of basic components
                    new_components = term1['components'] + term2['components']
                    
                    new_sf.continuous_terms.append({
                        'mag': new_mag,
                        'components': new_components
                    })
            
            # B. Continuous (Self) * Impulses (Other)
            # Property: f(t) * delta(t-a) = f(a) * delta(t-a)
            for imp in other.impulses:
                t_loc = imp['t0'] / imp['k']
                # Sample 'self' at the impulse location
                sample_val = self._evaluate_scalar(t_loc)
                if sample_val != 0:
                    new_imp = imp.copy()
                    new_imp['mag'] *= sample_val
                    new_sf.impulses.append(new_imp)

            # C. Impulses (Self) * Continuous (Other)
            for imp in self.impulses:
                t_loc = imp['t0'] / imp['k']
                # Sample 'other' at the impulse location
                sample_val = other._evaluate_scalar(t_loc)
                if sample_val != 0:
                    new_imp = imp.copy()
                    new_imp['mag'] *= sample_val
                    new_sf.impulses.append(new_imp)
            
            # D. Impulse * Impulse is technically undefined (or infinity), ignored here.
            
            return new_sf

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self * -1


# --- 3. Shorthand Function Generators ---

def _extract_params(arg):
    if isinstance(arg, TimeVariable):
        return arg.k, -arg.c
    elif isinstance(arg, (int, float)):
        raise ValueError("Argument to u(), r(), p() must be a function of 't'.")
    else:
        raise ValueError("Invalid argument type.")

def u(arg):
    k, t0 = _extract_params(arg)
    sf = SingularityFunction()
    # Create a single product term with one component
    sf.continuous_terms.append({
        'mag': 1.0, 
        'components': [{'type': 'step', 't0': t0, 'k': k}]
    })
    return sf

def r(arg):
    k, t0 = _extract_params(arg)
    sf = SingularityFunction()
    sf.continuous_terms.append({
        'mag': 1.0, 
        'components': [{'type': 'ramp', 't0': t0, 'k': k}]
    })
    return sf

def p(arg):
    k, t0 = _extract_params(arg)
    sf = SingularityFunction()
    sf.continuous_terms.append({
        'mag': 1.0, 
        'components': [{'type': 'parabola', 't0': t0, 'k': k}]
    })
    return sf

def delta(arg):
    k, t0 = _extract_params(arg)
    sf = SingularityFunction()
    sf.impulses.append({'mag': 1.0, 't0': t0, 'k': k})
    return sf
