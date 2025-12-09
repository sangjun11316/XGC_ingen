import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf
except ImportError as e:
    print(f"Error: Could not import 'xgc_ingen' from parent directory.\nDetails: {e}")
    sys.exit(1)

def plot_profiles(filenames):
    """
    Reads and plots multiple profile files on the same figure.
    """
    if not filenames:
        print("No files provided to plot.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    
    # Cycle through colors/markers
    colors     = ['tab:blue', 'r', 'b', 'g', 'm', 'k']
    markers    = ['o', 'x', 's', '^', 'v', '*']
    linestyles = ['-', '-', '--', '-.', ':', '-']

    for i, fname in enumerate(filenames):
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue

        try:
            psi, val = read_prf(fname)
            
            # --- Calculate Derivatives ---
            # np.gradient calculates central difference
            d1 = np.gradient(val, psi)
            d2 = np.gradient(d1, psi)

            # Label with the filename (basename only)
            label_str = os.path.basename(fname)
            
            # Select styles (use modulo to prevent index errors)
            c  = colors[i % len(colors)]
            m  = markers[i % len(markers)]
            ls = linestyles[i % len(linestyles)]
            
            # --- Plot 0: Main Value ---
            axs[0].plot(psi, val, label=label_str, color=c, marker=m, 
                        markersize=4, linestyle=ls, alpha=0.8)
            
            # --- Plot 1: 1st Derivative ---
            axs[1].plot(psi, d1, label=f"d/d$\psi$ {label_str}", color=c, marker=m, 
                        markersize=4, linestyle=ls, alpha=0.8)

            # --- Plot 2: 2nd Derivative ---
            axs[2].plot(psi, d2, label=f"d$^2$/d$\psi^2$ {label_str}", color=c, marker=m, 
                        markersize=4, linestyle=ls, alpha=0.8)
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # --- Formatting ---
    
    # Subplot 0 (Main)
    axs[0].set_ylabel('Value')
    axs[0].set_title('Profile Comparison')
    #axs[0].set_ylim(0, 1800) # Keep your specific limits for the main plot
    axs[0].legend(loc='best', fontsize='small')

    # Subplot 1 (Gradient)
    axs[1].set_ylabel(r'1st Deriv ($dVal/d\psi$)')
    axs[1].axhline(0, color='k', linestyle=':', alpha=0.3) # Zero line reference

    # Subplot 2 (Curvature)
    axs[2].set_ylabel(r'2nd Deriv ($d^2Val/d\psi^2$)')
    axs[2].axhline(0, color='k', linestyle=':', alpha=0.3) # Zero line reference
    
    # Common X-axis label (only on the bottom plot)
    axs[2].set_xlabel(r'Normalized Poloidal Flux ($\psi_N$)')
    axs[2].set_xlim(0.85, 1.1)

    # Grid for all
    for ax in axs:
        ax.grid(True, which='major', linestyle='-', alpha=0.6)
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get files from command line arguments
    files = sys.argv[1:]
    print("Usage: python plot_profile.py <file1.prf> [file2.prf ...]")
    print("      or hard-code the file names in the script")
    if not files:
        files = ['./inputs/te.prf', './inputs/ti.prf']
        plot_profiles(files)
    else:
        plot_profiles(files)
