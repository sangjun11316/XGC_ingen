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

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Cycle through colors/markers if you have many files
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(filenames)))
    markers = ['o', 'x', 's', '^', 'v', 'D']

    for i, fname in enumerate(filenames):
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
            
        try:
            # Usage of read_prf as requested
            psi, val = read_prf(fname)
            
            # Label with the filename (basename only)
            label_str = os.path.basename(fname)
            
            marker = markers[i % len(markers)]
            color = colors[i]
            
            # Plot
            ax.plot(psi, val, label=label_str, color=color, marker=marker, 
                    markersize=4, linestyle='-', alpha=0.8)
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # Formatting
    ax.set_xlabel(r'Normalized Poloidal Flux ($\psi_N$)')
    ax.set_ylabel('Value')
    ax.set_title('Profile Comparison')
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.legend(loc='best')
    
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
