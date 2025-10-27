import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg') # to enable interactive backend
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline # RectBivariateSpline is often better than interp2d
from pathlib import Path

PROTON_MASS   = 1.6720e-27  # kg
ELECTRON_MASS = 9.1094e-31  # kg
UNIT_CHARGE   = 1.6022e-19  # C

class Eqdsk:
    def __init__(self, filename):
        self.filename = filename
        print(f">> loading g-file: {self.filename}")

        self._read_gfile()

    def _read_1d(self, f, num, width=16):
        """ Helper to read 1D array, correctly handling multiple lines """
        data = np.zeros(num)
        items_read = 0
        # Loop until the required number of items is read
        while items_read < num:
            line = f.readline()
            if not line: # Check for unexpected end of file
                raise EOFError(f"Expected {num} values, but reached end of file after reading {items_read}.")
            # remove newline characters
            line = line.rstrip('\n')
            # calculate how many full items are on this line
            items_on_line = len(line) // width
            # read items from the current line
            for i in range(items_on_line):
                if items_read < num:
                    start_index = i * width
                    end_index = start_index + width
                    try:
                        data[items_read] = float(line[start_index:end_index])
                    except ValueError:
                        raise ValueError(f"Could not convert '{line[start_index:end_index]}' to float. Line: '{line}'")
                    items_read += 1
                else:
                    break 
        return data

    def _read_2d(self, f, nrow, ncol, width=16):
        """
        Helper to read a 2D array (nrow x ncol) of fixed-width floats
        from a file object, correctly handling multiple lines.
        """
        total_num = nrow * ncol
        data = self._read_1d(f, total_num, width)
        data = data.reshape((nrow, ncol), order='F')
    
        return data

    def _read_gfile(self):
        with open(self.filename, 'r') as f:
            # line 1: header and dimensions
            line_str = f.readline()
            line_parts = line_str.split()
            self.header = line_str[:line_str.rfind(line_parts[-3])].strip() # Get text part
            self.idum, self.nw, self.nh = map(int, line_parts[-3:]) # mw, mh

            # line 2: grid geometry
            line_vals = self._read_1d(f, 5)
            self.rdim, self.zdim, self.rcentr, self.rleft, self.zmid = line_vals

            # line 3: magnetic axis, boundary psi, bcenter
            line_vals = self._read_1d(f, 5)
            self.rmag, self.zmag, self.smag, self.sbdy, self.bcentr = line_vals

            # line 4: plasma current, etc.
            line_vals = self._read_1d(f, 5)
            self.ip, self.smag_check, _, self.rmag_check, _ = line_vals 

            # line 5: Z-axis etc. (often duplicates)
            line_vals = self._read_1d(f, 5)
            self.zmag_check, _, self.sbdy_check, _, _ = line_vals

            # 1D profiles
            self.fpol  = self._read_1d(f, self.nw)
            self.pres  = self._read_1d(f, self.nw)
            self.ffp   = self._read_1d(f, self.nw) # F*F'
            self.pp    = self._read_1d(f, self.nw) # P' 
            self.psirz = self._read_2d(f, self.nw, self.nh) # [nw, nh]
            self.q     = self._read_1d(f, self.nw) # q profile

            # boundaries
            line_part = f.readline().split()
            self.nsep, self.nlim = map(int, line_part[:2])
            self.rzsep = self._read_2d(f, 2, self.nsep).T # [nsep, 2]
            self.rzlim = self._read_2d(f, 2, self.nlim).T # [nlim, 2]

        # basic grid setup
        self.r = np.linspace(self.rleft, self.rleft+self.rdim, self.nw)
        self.z = np.linspace(self.zmid-self.zdim/2,  self.zmid+self.zdim/2, self.nh)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        # normalized psi
        self.psinrz = (self.psirz - self.smag) / (self.sbdy - self.smag)
        self.psin = np.linspace(0, 1, self.nw)

        # etc
        self.bmag = self.fpol[0]/self.rmag

    def plot_overview(self):
        if not hasattr(self, 'psinrz'):
            print("g-file not read or processed.")
            return

        print("----------------------------------")
        buffer = f'Info:\n{self.header}'

        buffer += '\n\n'
        buffer += '{:25s} {:7.3f} [T] \n'.format('Center field',self.bcentr)
        buffer += '{:25s} {:7.3f} [MA]\n'.format('Plasma current',self.ip/1.e6)
        buffer += '{:25s} {:7.3f} [m] \n'.format('Magnetic axis R',self.rmag)
        buffer += '{:25s} {:7.3f} [m] \n'.format('Magnetic axis Z',self.zmag)
        buffer += '{:25s} {:7.3f} [T] \n'.format('Toroidal field',self.fpol[0]/self.rmag)
        buffer += '{:25s} {:7.3f} [Wb] \n'.format('Poloidal flux',self.sbdy - self.smag)
        buffer += '{:25s} {:7.3f} [Wb] \n'.format('psi center',self.smag)

        buffer += '{:25s} {:7.3f}     \n'.format('Q0',self.q[0])
        qf = interp1d(self.psin, self.q)
        buffer += '{:25s} {:7.3f}     \n'.format('Q95',qf(0.95))
        buffer += '{:25s} {:7.3f}     \n'.format('Qedge',self.q[-1])

        buffer += '\nCtr-clockwise direction is (+)'

        print(buffer)
        print("----------------------------------")

        fig,ax = plt.subplots(figsize=(8, 10))
        cntr = ax.contour(self.r, self.z, self.psinrz.T, levels=np.linspace(0, 1.2, 61)) # Transpose psinrz
        fig.colorbar(cntr, ax=ax, label='Normalized Poloidal Flux ($\psi_N$)')
        ax.plot(self.rzsep[:, 0], self.rzsep[:, 1], 'tab:orange', linewidth=2, label='Separatrix')
        ax.plot(self.rzlim[:, 0], self.rzlim[:, 1], 'k', linewidth=2, label='Limiter')
        ax.plot(self.rmag, self.zmag, 'kx', markersize=10, mew=2, label='Magnetic Axis')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title(f'Equilibrium Overview ({self.header})')
        ax.axis('equal')
        ax.legend()
        ax.grid(True)
        plt.show()

class TommsInputGenerator:
    def __init__(self):
        # setups
        self.params = self._default_parameters()

        self.eq = None
        self.midplane = {}
        self.prof = {}
        self.prof_interp = {}

        # flags to track progress
        self.equilibrium_loaded = False

        # interactive loop
        self._run_interface()

    def _default_parameters(self):
        params = {
            'g_file'  : './input/g184833.04800_new',
            'num_mid' : 1000,

            'te_file' : './input/te_d184833_4800_pfile_new.prf',
            'ti_file' : './input/ti_d184833_4800_pfile_adj_08.prf',
            'ne_file' : './input/ne_d184833_4800_pfile_new.prf',


        }
        return params

    def _read_equilibrium(self):
        try: 
            self.eq = Eqdsk(self.params['g_file'])
            self.equilibrium_loaded = True
        except Exception as e:
            print(f"Error loading equilibrium file: {e}")
            self.equilibrium_loaded = False

        self.eq.plot_overview()

    def _get_midplane_mapping(self):
        if not self.equilibrium_loaded:
            print("Warning: Load equilibrium first.")
            return

        print(">> get midplane mapping")

        rmid = np.linspace(self.eq.rmag, np.amax(self.eq.r), self.params['num_mid'])
        zmid = np.full_like(rmid, self.eq.zmag)

        psi_interpolator = RectBivariateSpline(self.eq.z, self.eq.r, self.eq.psirz.T, kx=3, ky=3)
        psimid = psi_interpolator.ev(zmid, rmid)
        psinmid = (psimid - self.eq.smag) / (self.eq.sbdy - self.eq.smag)

        self.midplane['r']    = rmid
        self.midplane['z']    = zmid
        self.midplane['psi']  = psimid
        self.midplane['psin'] = psinmid

    def _read_prf(self, filename):
        with open(filename, 'r') as file:
            #read dimensions
            [n] = map(int, file.readline().strip().split())
            #allocate array
            psi=np.zeros(n)
            var=np.zeros(n)

            for l in range(n):
                [psi[l], var[l]]=map(float, file.readline().strip().split() )

            #read end flag
            [end_flag]=map(int, file.readline().strip().split())
            if(end_flag!=-1):
                print('Error: end flag is not -1. end_flag= %d'%end_flag)

        return psi, var

    def _read_profiles(self):
        if not self.equilibrium_loaded:
            print("Warning: Load equilibrium first.")
            return
        try:
            print(">> loading te, ti, ne profiles")

            psi_te, te = self._read_prf(self.params['te_file'])
            psi_ti, ti = self._read_prf(self.params['ti_file'])
            psi_ne, ne = self._read_prf(self.params['ne_file'])
            self.prof = {'psi_te': psi_te, 'te': te,
                         'psi_ti': psi_ti, 'ti': ti,
                         'psi_ne': psi_ne, 'ne': ne}
            self.profiles_loaded = True
        except Exception as e:
            print(f"Error loading profile files: {e}")
            self.profiles_loaded = False        

    def _interpolate_profiles(self):
        if not self.midplane or not self.prof:
            print("Warning: Midplane mapping and raw profiles must be loaded first.")
            return

        print(">> interpolating profiles onto midplane grid")

        psi_target = self.midplane['psin']

        # interp1d
        val_first = self.prof['te'][0]
        val_last = self.prof['te'][-1]
        f_te = interp1d(self.prof['psi_te'], self.prof['te'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        val_first = self.prof['ti'][0]
        val_last = self.prof['ti'][-1]
        f_ti = interp1d(self.prof['psi_ti'], self.prof['ti'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        val_first = self.prof['ne'][0]
        val_last = self.prof['ne'][-1]
        f_ne = interp1d(self.prof['psi_ne'], self.prof['ne'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        self.prof_interp['psin'] = psi_target
        self.prof_interp['te'] = f_te(psi_target) # eV
        self.prof_interp['ti'] = f_ti(psi_target) # eV
        self.prof_interp['ne'] = f_ne(psi_target) # m^-3
        
        # ensure profiles don't go unphysically negative after extrapolation
        #self.prof_interp['te'][self.prof_interp['te'] < 0] = 1e-3
        #self.prof_interp['ti'][self.prof_interp['ti'] < 0] = 1e-3
        #self.prof_interp['ne'][self.prof_interp['ne'] < 0] = 1e-3

    def plot_profiles(self):
        fig,ax = plt.subplots(figsize=(5,4))

        if self.prof:
            ax.plot(self.prof['psi_te'], self.prof['te']/1E3, label='raw te [keV]')
            ax.plot(self.prof['psi_ti'], self.prof['ti']/1E3, label='raw ti [keV]')
            ax.plot(self.prof['psi_ne'], self.prof['ne']/1E19, label='raw ne [1E19]')

        if self.prof_interp:
            ax.plot(self.prof_interp['psin'], self.prof_interp['te']/1E3, c='k', ls='--', label='__interp te [keV]')
            ax.plot(self.prof_interp['psin'], self.prof_interp['ti']/1E3, c='k', ls='--', label='__interp ti [keV]')
            ax.plot(self.prof_interp['psin'], self.prof_interp['ne']/1E19, c='k', ls='--', label='__interp ne [1E19]')

        ax.set_xlabel('Normalized Poloidal Flux ($\psi_N$)')
        ax.set_ylabel('a.u.')
        ax.set_title('Initial Raw Profiles')
        ax.grid(True, alpha=0.5)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _run_interface(self):
        while True:
            status_eq = 'yes' if self.equilibrium_loaded else 'no'
            print("##################################")
            print("#      TOMMS Input Generator     #")
            print("##################################")
            print("\n### Status ###")
            print(f" Eq: [{status_eq}]")

            # TODO: separate by 'choices'
            print("\n### Main ###")
            self._read_equilibrium()
            self._get_midplane_mapping()
            self._read_profiles()
            self._interpolate_profiles() 
            self.plot_profiles()

            print("----------------------------------")

            # exiting
            print(">> Exiting")
            break

if __name__=='__main__':
    tomms_ingen = TommsInputGenerator()
