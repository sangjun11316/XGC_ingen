import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, filename):
        self.filename = filename
        
        # Use the imported read_prf function
        try:
            self.psi, self.val = read_prf(filename)
        except Exception as e:
            print(f"Error reading profile: {e}")
            sys.exit(1)
            
        self.psi_orig = self.psi.copy()
        self.val_orig = self.val.copy()

    def fix_hollow_core(self, cutoff_radius=0.20):
        """ 
        Applies a parabolic fit to the core to force monotonicity.
        """
        c_idx = np.argmin(np.abs(self.psi - cutoff_radius))
        rc = self.psi[c_idx]

        if rc <= 0:
            print(">> Cutoff radius is <= 0. Skipping core fix.")
            return

        # Calculate original gradient at the cutoff
        d1 = np.gradient(self.val, self.psi)
        m = d1[c_idx]

        # Extract core mask and apply the mathematical splice
        core_mask = self.psi < rc
        psi_core = self.psi[core_mask]

        self.val[core_mask] = self.val[c_idx] + m * (psi_core**2 - rc**2) / (2 * rc)
        print(f">> Fixed hollow core (Parabolic fit below psi={rc:.3f})")

    def plot_comparison(self):
        """ Plots original vs modified profile and their 1st derivatives """
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        
        # --- Plot 0: Main Value ---
        axs[0].plot(self.psi_orig, self.val_orig, 'k--', label='Original', alpha=0.5, zorder=1)
        axs[0].plot(self.psi, self.val, 'r-', marker='x', markersize=4, label='Fixed', linewidth=2, zorder=2)
        
        axs[0].set_ylabel('Value')
        axs[0].set_title(f'Core Profile Fix: {os.path.basename(self.filename)}')
        axs[0].axvline(0.20, color='k', linestyle=':', alpha=0.5) # Hardcoded reference line
        axs[0].legend(loc='best')
        
        # --- Plot 1: 1st Derivative ---
        d1_orig = np.gradient(self.val_orig, self.psi_orig)
        d1_fixed = np.gradient(self.val, self.psi)
        
        axs[1].plot(self.psi_orig, d1_orig, 'k--', alpha=0.5, zorder=1)
        axs[1].plot(self.psi, d1_fixed, 'r-', marker='x', markersize=4, linewidth=2, zorder=2)
        
        axs[1].set_ylabel(r'1st Deriv ($dVal/d\psi$)')
        axs[1].axhline(0.0, color='k', linestyle='-', lw=1, alpha=0.5, zorder=3)
        axs[1].axvline(0.20, color='k', linestyle=':', alpha=0.5)
        axs[1].set_xlabel(r'Normalized Poloidal Flux ($\psi_N$)')
        axs[1].set_xlim(0., 1.1)

        for ax in axs:
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

        plt.tight_layout()
        plt.show()

# --- Main Script ---
if __name__ == "__main__":
    # Strictly enforce a single input file
    if len(sys.argv) != 2:
        print("Usage: python fix_profile_core.py <prf_file>")
        sys.exit(1)

    print("\n+++++++++++++++++++++++++++++++++++++++++++")
    print("+           Core Profile Fixer            +")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    prf_file = sys.argv[1]
    print(f" Target Profile: {prf_file}")
    
    # Initialize modifier
    mod = ProfileModifier(prf_file)
    
    # --- Apply Logic ---
    cutoff = 0.10
    mod.fix_hollow_core(cutoff_radius=cutoff)

    # --- Plot ---
    mod.plot_comparison()

    # --- Save ---
    # Auto-generates the filename (e.g., 'te.prf' -> 'te_mod.prf')
    base, ext = os.path.splitext(prf_file)
    new_filename = f"{base}_fix_core{ext}"
    write_prf(new_filename, mod.psi, mod.val)
