import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys, os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf
except ImportError as e:
    print(f"Error: Could not import 'xgc_ingen' from parent directory.\nDetails: {e}")
    sys.exit(1)

# --- Helper Functions ---
def write_prf(filename, psi, value):
    """ Writes data to a .prf file in TOMMS format """
    try:
        with open(filename, 'w') as f:
            f.write(f"{len(psi)}\n")
            for p, v in zip(psi, value):
                f.write(f"{p:.12E}  {v:.12E}\n")
            f.write("-1\n")
        print(f">> Written to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

# --- Main Logic Class ---
class ProfileModifier:
    def __init__(self, filename, exp_filename=None):
        self.filename = filename
        
        # Use the imported read_prf function
        try:
            self.psi, self.val = read_prf(filename)
        except Exception as e:
            print(f"Error reading profile: {e}")
            sys.exit(1)
            
        self.psi_orig = self.psi.copy()
        self.val_orig = self.val.copy()

        # Load experimental data if provided
        self.psi_exp = None
        self.val_exp = None
        if exp_filename:
            if os.path.exists(exp_filename):
                print(f"Loading experimental data from {exp_filename}...")
                try:
                    self.psi_exp, self.val_exp = read_prf(exp_filename)
                except Exception as e:
                    print(f"Warning: Failed to read experimental file: {e}")
            else:
                print(f"Warning: Experimental file {exp_filename} not found.")

    def extrapolate_sol(self, psi_sep=1.0, psi_max=1.3, target_floor=10.0):
        """ 
        Applies the exponential decay logic from IDL.
        """
        # 1. Find Separatrix Index (closest to psi_sep)
        idx_sep = np.argmin(np.abs(self.psi - psi_sep))
        
        # Ensure we are working with the "Core" part up to separatrix
        psi_core = self.psi[:idx_sep+1]
        val_core = self.val[:idx_sep+1]

        # 2. Calculate Derivative at Separatrix
        # np.gradient uses central diff in interior, one-sided at boundaries
        grad_arr = np.gradient(val_core, psi_core)
        dval_sep = grad_arr[-1] # Derivative at separatrix
        val_sep  = val_core[-1] # Value at separatrix

        print(f">> Extrapolating to SOL")
        print(f"  * Separatrix @ psi={psi_core[-1]:.4f}: Val={val_sep:.3e}, Grad={dval_sep:.3e}")
        print(f"  * Target Floor value: {target_floor}")

        # 3. Generate SOL Grid
        n_sol_points = 50
        psi_sol = np.linspace(psi_core[-1], psi_max, n_sol_points + 1)[1:] # exclude first point (sep)

        # 4. Calculate Exponential Decay
        decay_fraction = 1.0 - target_floor/val_sep
        
        k = dval_sep / (decay_fraction * val_sep)
        val_sol = val_sep * (decay_fraction * np.exp(k * (psi_sol - psi_core[-1])) + (1.0 - decay_fraction))

        # 5. Combine Core and SOL "Knots"
        psi_knots = np.concatenate([psi_core, psi_sol])
        val_knots = np.concatenate([val_core, val_sol])

        # 6. Interpolate back onto the original grid (or extended grid if needed)
        # Using Cubic Spline (kind='cubic') to match IDL 'spline' mapping sqrt(psi) -> val
        f_interp = interp1d(np.sqrt(psi_knots), val_knots, kind='cubic', 
                            bounds_error=False, fill_value="extrapolate")
        
        # Generate new values on the FULL original grid
        self.val = f_interp(np.sqrt(self.psi))

        # 7. Clip minimum value
        min_sol_val = np.min(val_sol)
        self.val = np.maximum(self.val, min_sol_val)

    def apply_ti_clamp(self, clamp_psi=1.15):
        """ 
        Clamps values beyond a certain psi to the value at that psi.
        """
        # Find index where psi >= clamp_psi
        indices = np.where(self.psi >= clamp_psi)[0]
        
        if len(indices) > 0:
            idx_clamp = indices[0]
            clamp_val = self.val[idx_clamp-1] # Use value just before
            
            print(f">> Clamping profile at psi={self.psi[idx_clamp]:.3f} to value {clamp_val:.3e}")
            self.val[indices] = clamp_val

    def plot_comparison(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Experimental Data if available
        if self.psi_exp is not None:
            ax.scatter(self.psi_exp, self.val_exp, c='blue', s=30, label='Experimental', alpha=0.4, zorder=1)

        ax.plot(self.psi_orig, self.val_orig, 'k--', marker='o', label='Original Input', alpha=0.6, zorder=2)
        ax.plot(self.psi,      self.val,      'r-',  marker='x', label='Modified', linewidth=2, zorder=3)
        
        # Guide line at separatrix
        ax.axvline(1.0, color='k', linestyle=':', lw=2, label='Separatrix', zorder=3)
        ax.axhline(0.0, color='k', linestyle=':', lw=2, zorder=3)
        
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel('Value')
        ax.set_title(f'Profile Modification: {os.path.basename(self.filename)}')
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
        plt.show()

# --- Main Script ---
if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python modify_profile.py <prf_file> [optional: <exp_prf_file>]")
        sys.exit(1)

    print("\n+++++++++++++++++++++++++++++++++++++++++++")
    print("+           Profile modifier              +")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    # profile to be modified from
    prf_file = sys.argv[1]

    # optional experimental data
    exp_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f" Target Profile: {prf_file}")
    print(f" Exp.   Profile: {exp_file} (optional)")
    
    # Initialize modifier with optional experimental file
    mod = ProfileModifier(prf_file, exp_filename=exp_file)
    
    # --- Apply Logic ---
    # 1. Extrapolate SOL (applies to Ne, Te, Ti)
    target_floor = 10.0
    mod.extrapolate_sol(psi_sep=1.0, psi_max=1.3, target_floor=target_floor)
    
    # 2. Apply Ti Clamp (Heuristic: if filename contains 't' or 'T')
    if 't' in os.path.basename(prf_file).lower():
        mod.apply_ti_clamp(clamp_psi=1.15)

    # --- Plot ---
    mod.plot_comparison()

    # --- Save ---
    base, ext = os.path.splitext(prf_file)
    new_filename = f"{base}_mod{ext}"
    write_prf(new_filename, mod.psi, mod.val)
