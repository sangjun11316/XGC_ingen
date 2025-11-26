import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Import from parent directory ---
# This one-liner adds '../' to the system path so we can import xgc_ingen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from xgc_ingen import Eqdsk
except ImportError as e:
    print(f"Error: Could not import 'xgc_ingen' from parent directory.\nDetails: {e}")
    sys.exit(1)

PSIN_COMPARE = [0.25, 0.5, 0.75, 1.0]

def compare_eqdsks(file1, file2):
    """
    Loads two EQDSK files and plots their 2D psi contours side-by-side
    and overlaid using RAW Poloidal Flux (psirz).
    """
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        return
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        return

    # Load Data
    print(f"Loading {file1}...")
    eq1 = Eqdsk(file1)
    print(f"Loading {file2}...")
    eq2 = Eqdsk(file2)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    
    # --- Determine Common Contour Levels for Raw Flux ---
    # We want a unified scale to compare colors visually
    data1 = getattr(eq1, 'psirz', np.array([]))
    data2 = getattr(eq2, 'psirz', np.array([]))
    
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    # Create ~25 levels spanning the full range of both files
    levels = np.linspace(vmin, vmax, 25)

    # --- Plot 1: First EQDSK ---
    ax = axes[0]
    if hasattr(eq1, 'psirz'):
        cntr1 = ax.contour(eq1.r, eq1.z, eq1.psirz.T, levels=levels, cmap='viridis')
        ax.plot(eq1.rzsep[:, 0], eq1.rzsep[:, 1], 'r-', linewidth=2, label='Separatrix')
        ax.plot(eq1.rzlim[:, 0], eq1.rzlim[:, 1], 'k-', linewidth=1.5, label='Limiter')
        ax.plot(eq1.rmag, eq1.zmag, 'rx', markersize=8, mew=2)
        ax.set_title(f'File 1: {os.path.basename(file1)}\n{getattr(eq1, "header", "")}')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')

    # --- Plot 2: Second EQDSK ---
    ax = axes[1]
    if hasattr(eq2, 'psirz'):
        cntr2 = ax.contour(eq2.r, eq2.z, eq2.psirz.T, levels=levels, cmap='viridis')
        ax.plot(eq2.rzsep[:, 0], eq2.rzsep[:, 1], 'r-', linewidth=2, label='Separatrix')
        ax.plot(eq2.rzlim[:, 0], eq2.rzlim[:, 1], 'k-', linewidth=1.5, label='Limiter')
        ax.plot(eq2.rmag, eq2.zmag, 'rx', markersize=8, mew=2)
        ax.set_title(f'File 2: {os.path.basename(file2)}\n{getattr(eq2, "header", "")}')
        
        # Add colorbar for individual plots
        cbar = fig.colorbar(cntr2, ax=axes[:2], location='bottom', fraction=0.05, pad=0.05)
        cbar.set_label(r'Poloidal Flux $\psi$ [Wb]')

    ax.set_xlabel('R [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Overlay / Comparison ---
    ax = axes[2]
    
    if hasattr(eq1, 'psirz') and hasattr(eq2, 'psirz'):
        # Determine specific RAW flux levels to compare.
        # We calculate the raw flux values corresponding to psi_N=0.5 and psi_N=1.0 (boundary)
        # of File 1, and see where those values lie on File 2.
        overlay_levels = []
        overlay_levels = [eq1.smag + val * (eq1.sbdy - eq1.smag) for val in PSIN_COMPARE]

        level_labels = f"$\\psi_N$: [{', '.join(map(str, PSIN_COMPARE))}] of F1"

        # File 1 = Blue dashed
        ax.contour(eq1.r, eq1.z, eq1.psirz.T, levels=overlay_levels, colors='blue', linestyles='-')
        ax.scatter(eq1.rzsep[::5, 0], eq1.rzsep[::5, 1], c='tab:blue', zorder=3, s=10, label='File 1 Sep')
       
        # File 2 = Red solid
        ax.contour(eq2.r, eq2.z, eq2.psirz.T, levels=overlay_levels, colors='red', linestyles='-')
        ax.scatter(eq2.rzsep[::5, 0], eq2.rzsep[::5, 1], c='tab:orange', zorder=3, s=10, label='File 2 Sep')
        
        # Limiter
        ax.plot(eq1.rzlim[:, 0], eq1.rzlim[:, 1], 'k-', alpha=0.3, label='Limiter (F1)')

        # Print basic stats difference
        print("\n--- Quick Stats Comparison ---")
        print(f"{'Parameter':<15} | {'File 1':<12} | {'File 2':<12} | {'Diff':<12}")
        print("-" * 55)
        
        ip1, ip2 = getattr(eq1, 'ip', 0.0), getattr(eq2, 'ip', 0.0)
        b1, b2 = getattr(eq1, 'bcentr', 0.0), getattr(eq2, 'bcentr', 0.0)
        rm1, rm2 = getattr(eq1, 'rmag', 0.0), getattr(eq2, 'rmag', 0.0)
        zm1, zm2 = getattr(eq1, 'zmag', 0.0), getattr(eq2, 'zmag', 0.0)
        sm1, sm2 = getattr(eq1, 'smag', 0.0), getattr(eq2, 'smag', 0.0)
        sb1, sb2 = getattr(eq1, 'sbdy', 0.0), getattr(eq2, 'sbdy', 0.0)

        print(f"{'Ip [MA]':<15} | {ip1/1e6:<12.3f} | {ip2/1e6:<12.3f} | {(ip1-ip2)/1e6:<12.3f}")
        print(f"{'B0 [T]':<15} | {b1:<12.3f} | {b2:<12.3f} | {b1-b2:<12.3f}")
        print(f"{'R_mag [m]':<15} | {rm1:<12.3f} | {rm2:<12.3f} | {rm1-rm2:<12.3f}")
        print(f"{'Z_mag [m]':<15} | {zm1:<12.3f} | {zm2:<12.3f} | {zm1-zm2:<12.3f}")
        print(f"{'Psi_axis [Wb]':<15} | {sm1:<12.3f} | {sm2:<12.3f} | {sm1-sm2:<12.3f}")
        print(f"{'Psi_bdy [Wb]':<15} | {sb1:<12.3f} | {sb2:<12.3f} | {sb1-sb2:<12.3f}")

    ax.set_title(f'Overlay Comparison (Raw $\\psi$)\n(Blue=File1, Red=File2)\n{level_labels}')
    ax.set_xlabel('R [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    plt.show()

def compare_eqdsks_psin(file1, file2):
    """
    Loads two EQDSK files and plots their 2D psi contours side-by-side
    and overlaid.
    """
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        return
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        return

    # Load Data
    print(f"Loading {file1}...")
    eq1 = Eqdsk(file1)
    print(f"Loading {file2}...")
    eq2 = Eqdsk(file2)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    
    levels = np.linspace(0, 1.2, 21) # Contour levels for psi_N [0, 1.2]
    
    # --- Plot 1: First EQDSK ---
    ax = axes[0]
    if hasattr(eq1, 'psinrz'):
        cntr1 = ax.contour(eq1.r, eq1.z, eq1.psinrz.T, levels=levels, cmap='viridis')
        ax.plot(eq1.rzsep[:, 0], eq1.rzsep[:, 1], 'r-', linewidth=2, label='Separatrix')
        ax.plot(eq1.rzlim[:, 0], eq1.rzlim[:, 1], 'k-', linewidth=1.5, label='Limiter')
        ax.plot(eq1.rmag, eq1.zmag, 'rx', markersize=8, mew=2)
        ax.set_title(f'File 1: {os.path.basename(file1)}\n{getattr(eq1, "header", "")}')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')

    # --- Plot 2: Second EQDSK ---
    ax = axes[1]
    if hasattr(eq2, 'psinrz'):
        cntr2 = ax.contour(eq2.r, eq2.z, eq2.psinrz.T, levels=levels, cmap='viridis')
        ax.plot(eq2.rzsep[:, 0], eq2.rzsep[:, 1], 'r-', linewidth=2, label='Separatrix')
        ax.plot(eq2.rzlim[:, 0], eq2.rzlim[:, 1], 'k-', linewidth=1.5, label='Limiter')
        ax.plot(eq2.rmag, eq2.zmag, 'rx', markersize=8, mew=2)
        ax.set_title(f'File 2: {os.path.basename(file2)}\n{getattr(eq2, "header", "")}')
        
        # Add colorbar for individual plots
        cbar = fig.colorbar(cntr2, ax=axes[:2], location='bottom', fraction=0.05, pad=0.05)
        cbar.set_label(r'Normalized Poloidal Flux ($\psi_N$)')

    ax.set_xlabel('R [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Overlay / Comparison ---
    ax = axes[2]
    
    if hasattr(eq1, 'psinrz') and hasattr(eq2, 'psinrz'):
        # Contour lines only (no fill) for overlay
        overlay_levels = PSIN_COMPARE
        level_labels = f"$\\psi_N$: [{', '.join(map(str, PSIN_COMPARE))}]"

        # File 1 = Blue dashed
        ax.contour(eq1.r, eq1.z, eq1.psinrz.T, levels=overlay_levels, colors='blue', linestyles='-')
        #ax.plot(eq1.rzsep[:, 0], eq1.rzsep[:, 1], c='tab:blue', ls='--', zorder=3, linewidth=1.5, label='File 1 Sep')
        ax.scatter(eq1.rzsep[::5, 0], eq1.rzsep[::5, 1], c='tab:blue', zorder=3, s=10, label='File 1 Sep')
        
        # File 2 = Red solid
        ax.contour(eq2.r, eq2.z, eq2.psinrz.T, levels=overlay_levels, colors='red', linestyles='-')
        #ax.plot(eq2.rzsep[:, 0], eq2.rzsep[:, 1], c='tab:orange', ls='--', zorder=3, linewidth=1.5, label='File 2 Sep')
        ax.scatter(eq2.rzsep[::5, 0], eq2.rzsep[::5, 1], c='tab:orange', zorder=3, s=10, label='File 2 Sep')
        
        # Limiter
        ax.plot(eq1.rzlim[:, 0], eq1.rzlim[:, 1], 'k-', alpha=0.3, label='Limiter (F1)')

        # Print basic stats difference
        print("\n--- Quick Stats Comparison ---")
        print(f"{'Parameter':<15} | {'File 1':<12} | {'File 2':<12} | {'Diff':<12}")
        print("-" * 55)
        
        ip1, ip2 = getattr(eq1, 'ip', 0.0), getattr(eq2, 'ip', 0.0)
        b1, b2 = getattr(eq1, 'bcentr', 0.0), getattr(eq2, 'bcentr', 0.0)
        rm1, rm2 = getattr(eq1, 'rmag', 0.0), getattr(eq2, 'rmag', 0.0)
        zm1, zm2 = getattr(eq1, 'zmag', 0.0), getattr(eq2, 'zmag', 0.0)
        sm1, sm2 = getattr(eq1, 'smag', 0.0), getattr(eq2, 'smag', 0.0)
        sb1, sb2 = getattr(eq1, 'sbdy', 0.0), getattr(eq2, 'sbdy', 0.0)

        print(f"{'Ip [MA]':<15} | {ip1/1e6:<12.3f} | {ip2/1e6:<12.3f} | {(ip1-ip2)/1e6:<12.3f}")
        print(f"{'B0 [T]':<15} | {b1:<12.3f} | {b2:<12.3f} | {b1-b2:<12.3f}")
        print(f"{'R_mag [m]':<15} | {rm1:<12.3f} | {rm2:<12.3f} | {rm1-rm2:<12.3f}")
        print(f"{'Z_mag [m]':<15} | {zm1:<12.3f} | {zm2:<12.3f} | {zm1-zm2:<12.3f}")
        print(f"{'Psi_axis [Wb]':<15} | {sm1:<12.3f} | {sm2:<12.3f} | {sm1-sm2:<12.3f}")
        print(f"{'Psi_bdy [Wb]':<15} | {sb1:<12.3f} | {sb2:<12.3f} | {sb1-sb2:<12.3f}")

    ax.set_title(f'Overlay Comparison\n(Blue=File1, Red=File2)\n{level_labels}')
    ax.set_xlabel('R [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_eqdsk.py <gfile1> <gfile2>")
    else:
        compare_eqdsks_psin(sys.argv[1], sys.argv[2])
        #compare_eqdsks(sys.argv[1], sys.argv[2])
