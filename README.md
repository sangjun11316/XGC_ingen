# XGC_ingen
Python workflow for preparing TOMMS mesh generation

# Example usage
$ mkdir rundir
$ cd rundir

# After preparing 'inputs' and 'params.in'
$ python ../xgc_ingen.py

# Profile editor
The profile utilities have a GUI wrapper for the common modification workflow:

```sh
python utils/profile_gui.py path/to/profile.prf
```

Optional experimental overlays can be added at launch or from the GUI:

```sh
python utils/profile_gui.py path/to/profile.prf --overlay path/to/experiment.prf
```

The editor supports:
- pedestal-top smoothing with C1 cubic flattening and optional post-diffusion passes in boundary patches
- radial psi-axis shifting
- fixed-separatrix tanh or exponential connection with SOL floor
- original/previous/current comparison with value, first derivative, and second derivative views
- saving the modified profile back to TOMMS `.prf` format
