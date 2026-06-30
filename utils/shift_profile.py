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
class ProfileShifter:
    def __init__(self, filename, exp_filename=None):
        self.filename = filename

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

    def scale_psi_axis(self, delta_psi=0.02, psi_ref=1.0):
        """
        Scale psi axis so that psi_ref moves outward by delta_psi.

        Example:
            psi_ref   = 1.0
            delta_psi = 0.02

        Then:
            scale = (psi_ref - delta_psi) / psi_ref = 0.98
            psi_new = scale * psi_old
        """
        scale = (psi_ref - delta_psi) / psi_ref

        print(">> Scaling psi axis")
        print(f"  * psi_ref   = {psi_ref}")
        print(f"  * delta_psi = {delta_psi}")
        print(f"  * scale     = {scale:.8f}")

        self.psi = scale * self.psi_orig
        self.val = self.val_orig.copy()

    def plot_comparison(self, delta_psi=0.02):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Experimental Data if available
        if self.psi_exp is not None:
            ax.scatter(self.psi_exp, self.val_exp, marker='s', c='blue', s=30,
                       label='Exp.', alpha=0.4, zorder=1)

            ax.scatter(self.psi_exp - delta_psi, self.val_exp, marker='s', facecolors='none', edgecolors='r', s=30,
                       label=f'Exp. (Δψ={delta_psi:.2f})', alpha=0.4, zorder=1)

        ax.plot(self.psi_orig, self.val_orig, 'k--', marker='o',
                label='Original Input', alpha=0.6, zorder=2)
        ax.plot(self.psi, self.val, 'r-', marker='x',
                label='Shifted', linewidth=2, zorder=3)

        ax.axvline(1.0, color='k', linestyle=':', lw=2, label='Separatrix', zorder=3)
        ax.axhline(0.0, color='k', linestyle=':', lw=2, zorder=3)

        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel('Value')
        ax.set_title(f'Profile Shift: {os.path.basename(self.filename)}')
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
        plt.show()

# --- Main Script ---
if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python shift_profile.py <prf_file> [optional: <exp_prf_file>]")
        sys.exit(1)

    print("\n+++++++++++++++++++++++++++++++++++++++++++")
    print("+          Profile radial shifter         +")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    # profile to be shifted
    prf_file = sys.argv[1]

    # optional experimental data
    exp_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f" Target Profile: {prf_file}")
    print(f" Exp.   Profile: {exp_file} (optional)")

    # Initialize shifter
    mod = ProfileShifter(prf_file, exp_filename=exp_file)

    # --- Apply Logic ---
    psi_ref = 1.0
    delta_psi = -0.01
    mod.scale_psi_axis(delta_psi=delta_psi, psi_ref=psi_ref)

    # --- Plot ---
    mod.plot_comparison(delta_psi=delta_psi)

    # --- Save ---
    base, ext = os.path.splitext(prf_file)
    new_filename = f"{base}_shift{ext}"
    write_prf(new_filename, mod.psi, mod.val)