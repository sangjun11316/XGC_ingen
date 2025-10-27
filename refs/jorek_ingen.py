import os,sys
import numpy as np
import lmfit
from shutil import move,copyfile,copytree
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
import time
import jorek_ingen as jin
from skimage import measure

class injorek:

    if __name__ == '__main__':
        injorek = jin.injorek()

    def __init__(self):

        self._declare_variables()
        self._initialise_input_variables()
        if not os.path.isfile('injorek.in'): 
            print('>>> Initial run, generate input')
            self._generate_input()
            exit()
        self._read_namelist()
        self._check_input_variables()
        self._show_input()
        self._get_inputs()
        self._run_interface()

    def _run_interface(self):
        isrun = 0
        while True:
            isrun = self._get_interaction('int','Choose the commands\n>>> 1) Make equil[%s] 2) Make profile[%s] 3) Make input[%s] 4) Exit'%(self.iseq,self.isprof,self.isready))
            if (isrun<=3 and isrun>0): self._read_namelist(); self._check_input_variables(); self._get_pfval();
            if isrun==1:   self._make_equil_infos()
            elif isrun==2: self._make_profiles()
            elif isrun==3: self._write_input_main()
            elif isrun==4: exit()

    def _make_profiles(self):
        if not self.iseq: print('>>> Do the equil step first!\n')
        else:
            self.isprof = False
            self._get_kinprof()
            self._fit_profiles()
            self._draw_profiles()

    def _make_equil_infos(self):

        self.iseq = False
        self._get_psival()
        self._draw_equils()

    def _get_interaction(self,type,msg):

        out = None
        while out==None:
            ans = input('>>> %s \n>>> '%msg)
            ans = ans.lower()
            if type=='logic':
                if ans=='y': out = True
                elif ans=='n': out = False
            elif type=='int':
                try: out = int(ans)
                except: pass
            elif type=='float':
                try: out = float(ans)
                except: pass
        return out

    def _write_input_main(self):
        if not self.iseq: print('>>> Do equil first'); return
        if not self.isprof: print('>>> Do profile first'); return
        self.isready = False
        if not os.path.isdir('PROFILES'): os.mkdir('PROFILES')
        self._jorek_variables()
        self._write_input_injorek()
        self._write_input_profiles()
        self.isready = True

    def _write_input_injorek(self):

        qf = interp1d(self.eq.psin,self.eq.q);
        f = open('inkstar_jorek','w');
        f.write('&in1\n\n');
        f.write(' ! ===============================\n');
        f.write(' ! KSTAR \n');
        f.write(' ! #%i \n'%self.eq.shotn);
        f.write(' ! q_95 = %2.2f, IP= %2.2fMA\n'%(qf(0.95),self.eq.ip/1.e6));
        f.write(' ! =============================== \n\n');
        f.write(' restart = .f.\n');
        f.write(' regrid  = .f.\n');
        f.write(' nstep_n = 50, 50 \n');
        f.write(' tstep_n = 1.,1.\n');        
        f.write(' nout = 2\n');
        f.write(' \n');
        f.write(' output_bnd_elements = .t.\n');
        f.write(' ! electron_density_eq = .t.\n');
        f.write(' ! bc_natural_open = .t.\n');
        f.write(" ! time_evol_scheme = 'Gears'\n");
        f.write(' ! gmres_max_iter = 700\n');
        f.write(' \n');
        f.write(' fbnd(1)   = 2.\n');
        f.write(' fbnd(2:4) = 0.\n');
        f.write(' mf        = 0\n');
        f.write(' \n');
        f.write(' n_boundary = %i\n'%self.in_param['ntht'])
        for i in range(self.in_param['ntht']):
            f.write(' R_boundary(%4i) = %+11.9f, Z_boundary(%4i) = %+11.9f, psi_boundary(%4i) = %+11.9f\n'%(i+1,self.inR[i],i+1,self.inZ[i],i+1,-self.inpsi[i]))
        f.write(' \n');

        f.write(' ellip       = %+f \n'%self.in_param['ellip'])
        f.write(' tria_u      = %+f \n'%self.in_param['tria_u'])
        f.write(' tria_l      = %+f \n'%self.in_param['tria_l'])
        f.write(' quad_u      = %+f \n'%self.in_param['quad_u'])
        f.write(' quad_l      = %+f \n'%self.in_param['quad_l'])
        f.write(' R_geo       = %+f \n'%self.in_param['rg0'])
        f.write(' Z_geo       = %+f \n'%self.in_param['zg0'])
        f.write(' F0          = %+f \n'%-self.eq.fpol[-1])
        f.write(' \n');

        f.write(' xtheta    = 0.1\n');
        f.write(' xleft     = 0.0\n');
        f.write(' xshift    = 0.0\n');
        f.write(' xpoint    = .t.\n');
        if self.in_param['double_null']: f.write(' xcase     = 3  \n');
        else:    
            if self.xpnt_P[0]>self.xpnt_P[1]: f.write(' xcase     = 1  \n');
            else:   f.write(' xcase     = 2  \n');
        f.write(' \n');
        f.write(' psi_axis_init = %2.1f\n'%-self.eq.smag);
        f.write(' amix = 0.5\n');
        f.write(' \n');
        if not self.in_param['fitted_ffprime']:
            f.write(' ffprime_file="PROFILES/jorek_ffprime.dat"\n');

        if not self.in_param['fitted_temperature']:
            if not self.in_param['two_temperature']:
                f.write(' T_file = "PROFILES/jorek_temperature.dat"\n');
            else:
                f.write(' Te_file = "PROFILES/jorek_temperature_e.dat"\n');
                f.write(' Ti_file = "PROFILES/jorek_temperature_i.dat"\n');
        
        if not self.in_param['fitted_density']:
            f.write(' rho_file = "PROFILES/jorek_density.dat"\n');
        f.write(' \n');

        if not self.in_param['two_temperature']:
            val1 = self.fit_coef['t'][1]*self.norm_t; val2 = self.fit_coef['t'][0]*self.norm_t;
            f.write(' ! Core temperature is %e Joules = %f keV \n'%(self.t0*self.norm_t,self.t0));
            f.write(' T_0             =%f \n'%val1);
            f.write(' T_1             =%f \n'%val2);
            f.write(' T_coef(1)       =%f \n'%self.fit_coef['t'][2]);
            f.write(' T_coef(2)       =%f \n'%self.fit_coef['t'][3]);
            f.write(' T_coef(3)       =%f \n'%self.fit_coef['t'][4]);
            f.write(' T_coef(4)       =%f \n'%self.fit_coef['t'][5]);
            f.write(' T_coef(5)       =%f \n'%self.fit_coef['t'][6]);
            f.write(' \n');
        else:
            val1 = self.fit_coef['te'][1]*self.norm_t; val2 = self.fit_coef['te'][0]*self.norm_t;
            f.write(' ! Core E temperature is %e Joules = %f keV \n'%(self.te0*self.norm_t,self.te0));
            f.write(' Te_0            =%f \n'%val1);
            f.write(' Te_1            =%f \n'%val2);
            f.write(' Te_coef(1)      =%f \n'%self.fit_coef['te'][2]);
            f.write(' Te_coef(2)      =%f \n'%self.fit_coef['te'][3]);
            f.write(' Te_coef(3)      =%f \n'%self.fit_coef['te'][4]);
            f.write(' Te_coef(4)      =%f \n'%self.fit_coef['te'][5]);
            f.write(' Te_coef(5)      =%f \n'%self.fit_coef['te'][6]);
            f.write(' \n');
            val1 = self.fit_coef['ti'][1]*self.norm_t; val2 = self.fit_coef['ti'][0]*self.norm_t;
            f.write(' ! Core I temperature is %e Joules = %f keV \n'%(self.ti0*self.norm_t,self.ti0));
            f.write(' Ti_0            =%f \n'%val1);
            f.write(' Ti_1            =%f \n'%val2);
            f.write(' Ti_coef(1)      =%f \n'%self.fit_coef['ti'][2]);
            f.write(' Ti_coef(2)      =%f \n'%self.fit_coef['ti'][3]);
            f.write(' Ti_coef(3)      =%f \n'%self.fit_coef['ti'][4]);
            f.write(' Ti_coef(4)      =%f \n'%self.fit_coef['ti'][5]);
            f.write(' Ti_coef(5)      =%f \n'%self.fit_coef['ti'][6]);

        f.write(' central_density = %e \n'%(self.ne0*1.e-1));
        val1 = self.fit_coef['ne'][0]/self.fit_coef['ne'][1];
        f.write(' rho_0           =%f \n'%1.0);
        f.write(' rho_1           =%f \n'%val1);
        f.write(' rho_coef(1)     =%f \n'%self.fit_coef['ne'][2]);
        f.write(' rho_coef(2)     =%f \n'%self.fit_coef['ne'][3]);
        f.write(' rho_coef(3)     =%f \n'%self.fit_coef['ne'][4]);
        f.write(' rho_coef(4)     =%f \n'%self.fit_coef['ne'][5]);
        f.write(' rho_coef(5)     =%f \n'%self.fit_coef['ne'][6]);
        f.write(' \n');

        f.write(' \n');
        if self.in_param['normalized_vtor']: f.write(' normalized_velocity_profile = .t.\n');
        else:                                f.write(' normalized_velocity_profile = .f.\n');
        f.write(' rot_file = "PROFILES/jorek_rot.dat"\n');
        f.write(' \n');

        f.write(' !===============================\n');
        f.write(' ! Grid parameters\n');
        f.write(' !===============================\n');
        f.write(' n_R = 0\n');
        f.write(' n_Z = 0\n');
        f.write(' n_radial  =  150 !85  !\n');
        f.write(' n_pol     =  250 !85  !\n');
        f.write(' \n');
        f.write(' n_flux    = 90! 90!  ! increase for radial resoluion\n');
        f.write(' n_tht     =100! 150  !\n');
        f.write(' n_open    = 13! 35!  ! number of surfaces on outer side (    used for single-null!)\n');
        f.write(' n_outer   = 10! 15!  ! number of surfaces on outer side (  used for single-null!)\n');
        f.write(' n_inner   = 08! 08!  ! number of surfaces on inner side (not used for single-null!)\n');
        f.write(' n_leg     = 13! 13!  ! number of surfaces in lower leg  (   used for single-null!)\n');
        f.write(' n_private = 13! 13!  ! number of surfaces in lower priv (   used for single-null!)\n');
        f.write(' n_up_leg  = 15! 15!  ! number of surfaces in upper leg  (not used for single-null!)\n');
        f.write(' n_up_priv = 08! 08!  ! number of surfaces in upper priv (not used for single-null!)\n');
        f.write(' \n');
        f.write(' !-------------------- Control the localisation of grid resolution\n');
        f.write(' SIG_closed  = 0.03d0\n');
        f.write(' SIG_open    = 0.15d0\n');
        f.write(' SIG_outer   = 0.09d0 !increase to minimize localization\n');
        f.write(' SIG_inner   = 0.06d0\n');
        f.write(' SIG_private = 0.15d0\n');
        f.write(' SIG_up_priv = 0.15d0\n');
        f.write(' SIG_theta   = 0.02d0\n');
        f.write(' SIG_leg_0   = 0.1d0\n');
        f.write(' SIG_leg_1   = 0.2d0\n');
        f.write(' SIG_up_leg_0= 0.05d0\n');
        f.write(' SIG_up_leg_1= 0.1d0\n');
        f.write(' \n');
        f.write(' !-------------------- Control distance between plasma and open flux surface boundaries\n');
        f.write(' dPSI_open   = %6.4f    ! Used for songle-Null plasmas\n'%self.in_param['psi_open']);
        f.write(' dPSI_outer  = %6.4f    ! Used for songle-Null plasmas\n'%self.in_param['psi_outer']);
        f.write(' dPSI_inner  = %6.4f    ! Not used for songle-Null plasmas\n'%self.in_param['psi_inner']);
        f.write(' dPSI_private= %6.4f    ! Used for songle-Null plasmas\n'%self.in_param['psi_private_low']);
        f.write(' dPSI_up_priv= %6.4f    ! Not used for songle-Null plasmas\n'%self.in_param['psi_private_up']);
        f.write(' \n');
        f.write(' !--------------------------------- RMP_input ---\n');
        f.write(' RMP_on = .f.\n');
        f.write(' RMP_psi_cos_file = ""\n');
        f.write(' RMP_psi_sin_file = ""\n');
        f.write(' RMP_growth_rate    = 0.01 ! RMP_growth_rate * RMP_ramp_up_time must be ~cst\n');
        f.write(' RMP_ramp_up_time   = 600  ! in JOREK times\n');
        f.write(' \n');
        f.write(' !===============================\n');
        f.write(' ! MHD parameters (SI units)\n');
        f.write(' !===============================\n');
        f.write(' \n');
        f.write(' eta            = %5.3e\n'%self.eta);
        f.write(' visco          = %2.1e\n'%self.norm1);
        f.write(' visco_par      = %2.1e\n'%(self.norm1*(self.eq.bcentr**2)));
        f.write(' !visco_sol         = 20.\n');
        f.write(' !visco_sol_psi     = 1.01\n');
        f.write(' !visco_sol_sig     = 0.01\n');
        f.write(' visco_T_dependent = .true.\n')
        f.write(' \n');

        f.write(' D_par          = 0.d-1\n');
        f.write(' D_perp(1)      = %2.1e\n'%(2.*self.norm1));
        f.write(' D_perp(10)     = 1.d0\n');
        f.write(' \n');
        if not self.in_param['two_temperature']:
            f.write(' ZK_par         = %5.3e\n'%(self.kappa / 80.));
            f.write(' ZK_perp(1)     = %2.1e\n'%self.norm1);
            f.write(' ZK_perp(10)    = 1.d0\n');
        else:
            f.write(' ZK_i_par         = %5.3e\n'%(self.kappa / 80.));
            f.write(' ZK_i_perp(1)     = %2.1e\n'%self.norm1);
            f.write(' ZK_i_perp(10)    = 1.d0\n');
            f.write(' ZK_e_par         = %5.3e\n'%(self.kappa));
            f.write(' ZK_e_perp(1)     = %2.1e\n'%self.norm1);
            f.write(' ZK_e_perp(10)    = 1.d0\n');                       
        f.write(' \n');
        f.write(' eta_num        = 1.d-13\n');
        f.write(' visco_num      = 1.d-13\n');
        f.write(' D_perp_num     = 1.d-13\n');
        f.write(' ZK_perp_num    = 0.d-14\n');
        f.write(' \n');
        f.write(' heatsource     = 1.d-7 \n');
        f.write(' particlesource = 4.d-6 \n');
        f.write(' !source_file    = "PROFILES/jorek_source_prof.dat"\n');
        f.write(' \n');
        if self.in_param['two_temperature']:
            f.write(' tauIC  = %5.4f\n'%self.tauic);
        else:
            f.write(' tauIC  = %5.4f\n'%(self.tauic/2.));
        f.write(' Wdia   = .f.\n');
        f.write(' \n');
        f.write(' bootstrap = .f.\n');
        f.write(' \n');
        f.write(' NEO           = .f.\n');
        f.write(' neo_file = "PROFILES/neoclass_coef.dat"\n');
        f.write(' !amu_neo_const =  2.d-5 \n');
        f.write(' !aki_neo_const = -1.d0  \n');
        f.write(' \n');

        f.close();

    def _write_input_profiles(self):

        f = open('PROFILES/jorek_ffprime.dat','w')
        for i in range(len(self.ffp_val)):
            f.write('%+16.13e\t%+16.13e\n'%(self.ffp_psi[i],self.ffp_val[i]))
        f.close()

        f = open('PROFILES/jorek_density.dat','w')
        for i in range(len(self.psi_norm)):
            f.write('%+16.13e\t%+16.13e\n'%(self.psi_norm[i],self.p['nfit'][i]/self.p['nfit'][0]))
        f.close()

        f = open('PROFILES/jorek_temperature.dat','w')
        for i in range(len(self.psi_norm)):
            f.write('%+16.13e\t%+16.13e\n'%(self.psi_norm[i],self.p['tfit'][i]*self.norm_t))
        f.close()        

        f = open('PROFILES/jorek_temperature_e.dat','w')
        for i in range(len(self.psi_norm)):
            f.write('%+16.13e\t%+16.13e\n'%(self.psi_norm[i],self.p['tefit'][i]*self.norm_t))
        f.close()   

        f = open('PROFILES/jorek_temperature_i.dat','w')
        for i in range(len(self.psi_norm)):
            f.write('%+16.13e\t%+16.13e\n'%(self.psi_norm[i],self.p['tifit'][i]*self.norm_t))
        f.close()         

        if self.vt_in_pfile: rrf = interp1d(self.eq.prhoR[:,0],self.eq.prhoR[:,2],'cubic')
        else: rrf = interp1d(self.rot['psi_norm'],self.rot['rr'],'cubic')

        rr = np.copy(self.psi_norm)
        for i in range(len(rr)):
            if rr[i]>1.: rr[i] = 2.*rr[i-1] - rr[i-2]
            else: rr[i] = rrf(rr[i])

        f = open('PROFILES/jorek_rot.dat','w')
        for i in range(len(self.psi_norm)):
            if not self.in_param['normalized_vtor']: factor = 1./2./np.pi/rr[i]
            else: factor = 1./abs(self.eq.fpol[-1])*rr[i]
            factor = factor * 1.e3 * self.norm1
            f.write('%+16.13e\t%+16.13e\t%+16.13e\n'%(self.psi_norm[i],self.p['vtfit'][i]*factor,0.))
        f.close()                            

    def _jorek_variables(self):

        self.ne0    = self.p['nfit'][0]
        self.te0    = self.p['tefit'][0]
        self.ti0    = self.p['tifit'][0]
        self.t0     = self.p['tfit'][0]

        self.norm_t = 1.e3*self.e0*self.ne0*1.e19*self.mu0
        self.rho0   = self.ne0*1.e19 * self.mass_number * self.proton_mass
        self.norm1  = np.sqrt(self.rho0* self.mu0) 
        self.norm2  = np.sqrt(self.rho0/ self.mu0)
        self.norm3  = np.sqrt(self.mu0 / self.rho0)

        self.tauic  = self.mass_number * self.proton_mass / self.e0 / abs(self.eq.fpol[0]) / self.norm1
        if self.in_param['two_temperature']:
            self.eta    = 2*1.e-8 * self.zeff / ((1.-np.sqrt(0.258))**2) / (self.te0) ** 1.5
        else:
            self.eta    = 2*1.e-8 * self.zeff / ((1.-np.sqrt(0.258))**2) / (self.t0/2.) ** 1.5
        self.eta  = self.eta * self.norm2

        if self.in_param['two_temperature']:
            self.kappa    = 3.6 * 1.e29 * (self.te0)  ** 2.5 / self.ne0 / 1.e19
        else:
            self.kappa    = 3.6 * 1.e29 * (self.t0/2.)** 2.5 / self.ne0 / 1.e19

        self.kappa = self.kappa * self.norm1 * 2./3.

    def _draw_profiles(self):

        fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=(15,9))

        ax1.scatter(self.p['psi_norm'],self.p['nref'],marker='x',s=30,color='r')
        ax1.plot(self.psi_norm,self.p['nfit'])

        if self.in_param['two_temperature']:
            ax2.scatter(self.p['psi_norm'],self.p['tiref'],marker='x',s=30,color='r')
            ax2.plot(self.psi_norm,self.p['tifit'])
        else:
            ax2.scatter(self.p['psi_norm'],self.p['tref'],marker='x',s=30,color='r')
            ax2.plot(self.psi_norm,self.p['tfit'])

        ax3.scatter(self.p['psi_norm'],self.p['teref'],marker='x',s=30,color='r')
        ax3.plot(self.psi_norm,self.p['tefit'])        

        ax4.scatter(self.p['psi_norm'],self.p['vtref'],marker='x',s=30,color='r')
        ax4.plot(self.psi_norm,self.p['vtfit'])   

        ax1.set_title('Density [10(19)/m3]')
        ax1.set_xlabel('$\\psi_N$ [a.u]')
        if self.in_param['two_temperature']:
            ax2.set_title('$T_e$ [kev]')
            ax2.set_xlabel('$\\psi_N$ [a.u]')
        else:
            ax2.set_title('$T_{tot}$ [kev]')
            ax2.set_xlabel('$\\psi_N$ [a.u]')

        ax3.set_title('$T_e$ [kev]')
        ax3.set_xlabel('$\\psi_N$ [a.u]')
        ax4.set_title('Vtor(carbon) [km/s]')
        ax4.set_xlabel('$\\psi_N$ [a.u]')                        

        ax1.legend(['FIT','RAW'])
        ax2.legend(['FIT','RAW'])
        ax3.legend(['FIT','RAW'])
        ax4.legend(['FIT','RAW'])

        fig.tight_layout()
        plt.show(block=False)
        self.isprof = self._get_interaction('logic','Use these profiles [y/n]?')

    def _draw_equils(self):

        fig=plt.figure('EFIT overview',figsize=(15,8))
        ax1=plt.subplot2grid((2,3),(0,0),rowspan=2)
        ax2=plt.subplot2grid((2,3),(0,1),colspan=2)
        ax3=plt.subplot2grid((2,3),(1,1),colspan=2)

        ax1.contourf(self.eq.R,self.eq.Z,self.eq.psirz,50)
        ax1.plot(self.eq.rzbdy[:,0],self.eq.rzbdy[:,1],c='r')
        ax1.plot(self.eq.rzlim[:,0],self.eq.rzlim[:,1],c='blue')
        ax1.scatter(self.inR,self.inZ,marker='x',s=30,color='magenta')
        ax1.scatter(self.eq.rmag,self.eq.zmag,marker='x',s=50,color='green')
        ax1.scatter(self.xpnt_R[0],self.xpnt_Z[0],marker='x',s=50,color='orange')
        ax1.scatter(self.xpnt_R[1],self.xpnt_Z[1],marker='x',s=50,color='orange')

        lower_singlenull =False
        if self.xpnt_P[0]>self.xpnt_P[1]: lower_singlenull = True
        psi_bnd1 = min(self.xpnt_P[0],self.xpnt_P[1])
        psi_bnd2 = max(self.xpnt_P[0],self.xpnt_P[1])
        psi_open = self.in_param['psi_open'] + psi_bnd1
        psi_inner= self.in_param['psi_inner'] + psi_bnd2
        psi_outer= self.in_param['psi_outer'] + psi_bnd2
        psi_private_up= -self.in_param['psi_private_up'] + self.xpnt_P[0]
        psi_private_low= -self.in_param['psi_private_low'] + self.xpnt_P[1]
        psi_private   = -self.in_param['psi_private_low'] + psi_bnd1

        cs = self.eq._contour(psi_bnd1); 
        ax1.plot(cs[0][:,1],cs[0][:,0],c='red')

        cs = self.eq._contour(psi_bnd2); 
        for cc in cs: ax1.plot(cc[:,1],cc[:,0],c='red')

        if self.in_param['double_null']: 

            cs = self.eq._contour(psi_inner); tarv = 100.; tarc = np.zeros((1,2));
            for cc in cs:
                if np.max(cc[:,1])<tarv: tarc = cc; tarv = np.max(cc[:,1]);
            ax1.plot(tarc[:,1],tarc[:,0],c='yellow')        

            cs = self.eq._contour(psi_outer); tarv = 0.; tarc = np.zeros((1,2));
            for cc in cs:
                if np.min(cc[:,1])>tarv: tarc = cc; tarv = np.min(cc[:,1]);
            ax1.plot(tarc[:,1],tarc[:,0],c='yellow')        

            cs = self.eq._contour(psi_private_up); tarv = 0.; tarc = np.zeros((1,2));
            for cc in cs:
                if np.max(cc[:,0])>tarv: tarc = cc; tarv = np.max(cc[:,0]);
            ax1.plot(tarc[:,1],tarc[:,0],c='yellow')

            cs = self.eq._contour(psi_private_low); tarv = 0.; ttarc = np.zeros((1,2));
            for cc in cs:
                if np.min(cc[:,0])<tarv: tarc = cc; tarv = np.min(cc[:,0]);
            ax1.plot(tarc[:,1],tarc[:,0],c='yellow')        
        else:
            cs = self.eq._contour(psi_open); 
            for cc in cs: ax1.plot(cc[:,1],cc[:,0],c='yellow')

            cs = self.eq._contour(psi_private); tarv = 0.; ttarc = np.zeros((1,2));
            for cc in cs:
                if lower_singlenull:
                    if np.min(cc[:,0])<tarv: tarc = cc; tarv = np.min(cc[:,0]);
                else:
                    if np.max(cc[:,0])>tarv: tarc = cc; tarv = np.max(cc[:,0]);
            ax1.plot(tarc[:,1],tarc[:,0],c='yellow')                 

        ax1.set_title('Poloidal flux function')
        ax1.set_xlabel('R [m]')
        ax1.set_ylabel('Z [m]')
        ax1.axis('equal')
        ax1.set_xlim((min(self.eq.R),max(self.eq.R)))
        ax1.set_ylim((min(self.eq.Z),max(self.eq.Z)))

        ax2.scatter(self.eq.psin,self.eq.ffp,marker='x',s=30,color='r')
        ax2.plot(self.ffp_psi,self.ffp_val)        
        ax2.set_title('FFprime')
        ax2.set_xlabel('$\\psi_N$ [a.u]')

        factor = self.e0*1.e19;
        pres_th= self.p['ne']*self.p['te'] + self.p['ni']*self.p['ti']
        ax3.plot(self.eq.psin,self.eq.pres/1.e3,c='red')
        ax3.plot(self.p['psi_norm'],pres_th*factor,c='blue')
        ax3.set_title('Pressure [kPa]')
        ax3.set_xlabel('$\\psi_N$ [a.u]')
        ax3.legend(['Eq.','Thermal.'])

        fig.tight_layout()
        plt.show(block=False)
        self.iseq = self._get_interaction('logic','Use these equils [y/n]?')        

    def _fit_profiles(self):

        self.fit_coef = dict()
        x = self.p['psi_norm']; s = 0.2+ 5.*np.tanh((x-0.7)/0.1)

        #density
        print('>>> Density FIT')
        y = self.p['nref']; fun_flag = self.in_param['func_ne']        
        coef,self.p['nfit'] = self._lmfit_fit(x,y,s,fun_flag,'ne');
        if not self.in_param['func_ne']=='ptanh': coef, _ = self._lmfit_fit(x,y,s,'ptanh','ne');
        self.fit_coef['ne'] = coef

        #total_temperature
        print('>>> Ttot FIT')
        y = self.p['tref']; fun_flag = self.in_param['func_ti']; self.in_param['bnd_ti'] = 2.0*self.in_param['bnd_ti'];       
        coef,self.p['tfit'] = self._lmfit_fit(x,y,s,fun_flag,'ti');
        if not self.in_param['func_ti']=='ptanh': coef, _ = self._lmfit_fit(x,y,s,'ptanh','ti');
        self.fit_coef['t']  = coef; self.in_param['bnd_ti'] = 0.5*self.in_param['bnd_ti'];

        #electron_temperature
        print('>>> Te FIT')
        y = self.p['teref']; fun_flag = self.in_param['func_ti'];        
        coef,self.p['tefit'] = self._lmfit_fit(x,y,s,fun_flag,'te');
        if not self.in_param['func_te']=='ptanh': coef, _ = self._lmfit_fit(x,y,s,'ptanh','te');
        self.fit_coef['te'] = coef

        #ion_temperature
        print('>>> Ti FIT')
        y = self.p['tiref']; fun_flag = self.in_param['func_ti']; 
        coef,self.p['tifit'] = self._lmfit_fit(x,y,s,fun_flag,'ti');     
        if not self.in_param['func_ti']=='ptanh': coef, _ = self._lmfit_fit(x,y,s,'ptanh','ti');
        self.fit_coef['ti'] = coef                 

        #ion_temperature
        print('>>> Vtor FIT')
        s = 0.2+ 5.*np.tanh((x-0.7)/0.1)*(1-np.tanh((x-0.95)/0.02))
        y=self.p['vtref']; fun_flag = self.in_param['func_vt']; 
        coef,self.p['vtfit'] = self._lmfit_fit(x,y,s,fun_flag,'vt');                

        self.p['nfit'] =self._damp_sol_profiles(self.psi_norm,self.p['nfit'], self.in_param['func_ne'],self.in_param['bnd_ne'])
        self.p['tfit'] =self._damp_sol_profiles(self.psi_norm,self.p['tfit'], self.in_param['func_ti'],2.*self.in_param['bnd_ti'])
        self.p['tefit']=self._damp_sol_profiles(self.psi_norm,self.p['tefit'],self.in_param['func_te'],self.in_param['bnd_te'])
        self.p['tifit']=self._damp_sol_profiles(self.psi_norm,self.p['tifit'],self.in_param['func_ti'],self.in_param['bnd_ti'])
        self.p['vtfit']=self._damp_sol_profiles(self.psi_norm,self.p['vtfit'],self.in_param['func_vt'],self.in_param['bnd_vt'])

    def _damp_sol_profiles(self,x,y,fun_flag,bnd_val):

        ybnd = y[-1]
        if (bnd_val>=0 and (not fun_flag =='ptanh') and ybnd > bnd_val):
            y = y + 0.5*(1.+np.tanh((x-1.1)/0.1)) * (bnd_val-ybnd)
        return y

    def _lmfit_fit(self,x,y,s,fun_flag,flag):

        init_time = time.time()
        self._lmfit_param(flag,fun_flag)
        if fun_flag == 'ptanh': func = self._tanh_prof
        else: func = self._eped_prof

        result = lmfit.minimize(self._lmfit_residual,self.lmfit_p,args=(x,y,s,func))
        if self.in_param['fit_debug']:
            print('=----------------------------------------------------------------------------=')
            print('>>> Fitting result for %s, FIT_FUNC = %s'%(flag.upper(),self.in_param['func_'+flag].upper()))
            print('>>> XI2 = %6.3f, RED_XI2 = %6.4e'%(result.chisqr,result.redchi))
            print('>>> Elapsed time %6.3f(s) '%(time.time()-init_time))
            print('=----------------------------------------------------------------------------=')
            self._lmfit_print(result.params)
            print('=----------------------------------------------------------------------------=')

        coef = self._lmfit_params2array(result.params)
        return coef, func(self.psi_norm,**result.params)

    def _lmfit_params2array(self,p):

        varn = 7
        dat = np.zeros(varn)
        for i in range(varn):
            s = 'a%i'%(i+1)
            dat[i] = float(p[s].value)
        return dat

    def _lmfit_print(self,lm,colwidth=8, precision=4, fmt='g',
                     columns=['value', 'min', 'max', 'stderr', 'vary', 'expr',
                              'brute_step']):

        name_len = max(len(s) for s in lm)
        allcols = ['name'] + columns
        title = '{:{name_len}} ' + len(columns) * ' {:>{n}}'
        print(title.format(*allcols, name_len=name_len, n=colwidth).title())
        numstyle = '{%s:>{n}.{p}{f}}'  # format for numeric columns
        otherstyles = dict(name='{name:<{name_len}} ', stderr='{stderr!s:>{n}}',
                           vary='{vary!s:>{n}}', expr='{expr!s:>{n}}',
                           brute_step='{brute_step!s:>{n}}')
        line = ' '.join([otherstyles.get(k, numstyle % k) for k in allcols])

        count = 0
        for name, values in sorted(lm.items()):
            pvalues = {k: getattr(values, k) for k in columns}
            pvalues['name'] = name
            # stderr is a special case: it is either numeric or None (i.e. str)
            if 'stderr' in columns and pvalues['stderr'] is not None:
                pvalues['stderr'] = (numstyle % '').format(
                    pvalues['stderr'], n=colwidth, p=precision, f=fmt)
            elif 'brute_step' in columns and pvalues['brute_step'] is not None:
                pvalues['brute_step'] = (numstyle % '').format(
                    pvalues['brute_step'], n=colwidth, p=precision, f=fmt)
            print(line.format(name_len=name_len, n=colwidth, p=precision,
                              f=fmt, **pvalues))

    def _lmfit_residual(self,p,x,y,s,fun):

        yf = fun(x,**p)
        yt = (yf-y)**2 * (s+0.1)**2
        return yt

    def _lmfit_param(self,flag,fun_flag):

        self.lmfit_p = lmfit.Parameters()
        bndv = self.in_param['bnd_'+flag]
        if fun_flag == 'ptanh':
            if  (flag == 'te' or flag=='ti'): initv = 0.15; minv = 0.05; maxv = 0.2   ;   fixv = False
            elif(flag == 'ne'):               initv = 0.50; minv = 0.20; maxv = 1.0   ;   fixv = False
            else:                             initv = 50. ; minv = 0.;   maxv = 100.  ;   fixv = False
            if bndv>=0.:                      initv = bndv; minv = 0.;   maxv = 2.*bndv+1;fixv = True
            self._lmfit_put_param(1,initv,minv,maxv,fixv)

            if  (flag == 'te' or flag=='ti'): initv = 1.  ; minv = 0.3;  maxv = 20. ;   fixv = False
            elif(flag == 'ne'):               initv = 1.  ; minv = 0.5;  maxv = 10. ;   fixv = False
            else:                             initv = 100.; minv = 0. ;  maxv = 800.;   fixv = False            
            self._lmfit_put_param(2,initv,minv,maxv,fixv)

            self._lmfit_put_param(3,-0.1   ,-np.inf,0.     ,False)
            self._lmfit_put_param(4,+1.0   ,-np.inf,+np.inf,False)
            self._lmfit_put_param(5,+1.0   ,-np.inf,+np.inf,False)
            self._lmfit_put_param(6,+0.04  ,+0.02  ,+0.15  ,False)
            self._lmfit_put_param(7,+0.98  ,+0.88  ,+1.02  ,False)

        else:
            
            if  (flag == 'te' or flag=='ti'): initv = 0.15; minv = 0.05; maxv = 0.3   ; fixv = False
            elif(flag == 'ne'):               initv = 0.50; minv = 0.30; maxv = 1.0   ; fixv = False
            else:                             initv = 50. ; minv = 0.;   maxv = 100.  ; fixv = False
            self._lmfit_put_param(1,initv,minv,maxv,fixv)

            if  (flag == 'te' or flag=='ti'): initv = 0.5 ; minv = 0.15; maxv = 5.0 ;   fixv = False
            elif(flag == 'ne'):               initv = 1.  ; minv = 0.15; maxv = 5.0 ;   fixv = False
            else:                             initv = 100.; minv = 0. ;  maxv = +np.inf;fixv = False            
            self._lmfit_put_param(2,initv,minv,maxv,fixv)

            self._lmfit_put_param(3,+0.04  ,+0.02  ,0.15   ,False)

            if  (flag == 'vt'): initv = 200.; minv = 0. ;  maxv = +np.inf;   fixv = False
            else:               initv = 3.0 ; minv = 0.2;  maxv = 10.0   ;   fixv = False            
            self._lmfit_put_param(4,initv,minv,maxv,fixv)

            self._lmfit_put_param(5,+1.2   ,+1.05  ,+4.0   ,False)
            self._lmfit_put_param(6,+1.2   ,+1.05  ,+4.0   ,False)
            self._lmfit_put_param(7,+0.00  ,-0.02  ,+0.02  ,True)      

    def _lmfit_put_param(self,param_ind,initv,minv,maxv,fixv):
        args = dict()
        var_name      = 'a%i'%param_ind
        args['value'] = initv
        args['max']   = minv
        args['min']   = maxv
        args['vary']  = not fixv
        self.lmfit_p.add(var_name,**args)

    def _tanh_prof(self,x,a1,a2,a3,a4,a5,a6,a7):

        y = (a2 -  a1) * (1.0 + a3*x + a4 * x**2 + a5 * x**3) * 0.5 * (1.0 - np.tanh((x-a7)/a6*2.)) + a1
        return y

    def _eped_prof(self,x,a1,a2,a3,a4,a5,a6,a7):

        y = a1
        y = y + a2*(np.tanh((1. - (1-0.5*a3+a7))/(a3)*2.0) - np.tanh((x - (1-0.5*a3+a7))/(a3)*2.0))
        yt = (x/((1-0.5*a3+a7)-0.5*a3)) ** (a5)
        y = y + a4 * ((abs(1-yt)) **(a6)) * 0.5 * (1.0 + np.sign((1-0.5*a3+a7)-0.5*a3-x))
        return y

    def _get_kinprof(self):

        self.p['nref'] = np.copy(self.p['ne'])
        if self.in_param['use_ion_density']: self.p['nref'] = np.copy(self.p['ni'])

        self.p['tiref'] = np.copy(self.p['ti'])
        self.p['teref'] = np.copy(self.p['te'])
        self.p['tref']  = self.p['tiref'] + self.p['teref']
        self.p['nref'] = np.copy(self.p['ne'])
        self.p['vtref'] = np.copy(self.p['vt'])

        if self.in_param['use_gfile_p']:
            ind  = np.where(self.p['psi_norm']<=1.)
            pres = self.presf(self.p['psi_norm'][ind])
            t    = pres/self.p['nref'][ind]/1.e19/self.e0/1.e3

            self.p['tref'] = self.p['tref'] * 0.
            self.p['tref'][0:len(pres)] = t

            if self.in_param['two_temperature']: self.p['tiref'] = self.p['tref'] - self.p['teref']

            for i in range(len(self.p['tiref'])):
                if self.p['tiref'][i]<0.: self.p['tiref'][i] = 0.;

    def _get_pfval(self):

        ffpsig = self.in_param['ffpsig']

        psi_norm = np.linspace(0,1.,self.eq.nw);
        self.ffp_psi = np.zeros(self.eq.nw+41);
        self.ffp_val = np.zeros(self.eq.nw+41);
        for i in range(self.eq.nw):
            self.ffp_psi[i] = psi_norm[i]
            self.ffp_val[i] = self.eq.ffp[i]

        for i in range(38):
            self.ffp_psi[i+self.eq.nw-1] = 1. + 0.004*i
            self.ffp_val[i+self.eq.nw-1] = self.ffp_val[self.eq.nw-1] * (1.-np.tanh((0.004*i)/ffpsig));

        self.ffp_psi[self.eq.nw+37] = 1.15;  self.ffp_val[self.eq.nw+10] = self.ffp_val[self.eq.nw-1] * (1.-np.tanh((0.15*i)/ffpsig));
        self.ffp_psi[self.eq.nw+38] = 1.2;   self.ffp_val[self.eq.nw+11] = self.ffp_val[self.eq.nw-1] * (1.-np.tanh((0.20*i)/ffpsig));
        self.ffp_psi[self.eq.nw+39] = 1.25;  self.ffp_val[self.eq.nw+12] = self.ffp_val[self.eq.nw-1] * (1.-np.tanh((0.25*i)/ffpsig));
        self.ffp_psi[self.eq.nw+40] = 1.3;   self.ffp_val[self.eq.nw+13] = self.ffp_val[self.eq.nw-1] * (1.-np.tanh((0.30*i)/ffpsig));

        self.presf = interp1d(psi_norm,self.eq.pres,'cubic')

    def _get_xpoint(self):

        dr = 1.e-4; dz = 1.e-4; eps = 1.e-4;
        self.xpnt_R = np.zeros(2); self.xpnt_Z = np.zeros(2); self.xpnt_P = np.zeros(2);

        zmax = np.max(self.eq.rzbdy[:,1]); zmaxloc = np.argmax(self.eq.rzbdy[:,1]); rmax = self.eq.rzbdy[zmaxloc,0]
        zmin = np.min(self.eq.rzbdy[:,1]); zminloc = np.argmin(self.eq.rzbdy[:,1]); rmin = self.eq.rzbdy[zminloc,0]
        
        bnf  = interp2d(self.eq.r,self.eq.z,self.eq.br**2+self.eq.bz**2)

        #find upper-xpnt
        epx = 1.e0; epz = 1.e0; xx = rmax; zz= zmax; count1 = 0; count2 = 0;
        while ((epx > eps or epz > eps) and count1 < 100):
            zz2 = np.linspace(zmax,0.95*self.eq.z[-1],int((0.95*self.eq.z[-1]-zmax)/dz));
            bnt = bnf(xx,zz2);
            minb= np.min(bnt); minbloc = np.argmin(bnt)
            epz = (zz2[minbloc]-zz)/zz; zz = zz2[minbloc];

            xx2 = np.linspace(1.05*self.eq.r[0],0.95*self.eq.r[-1],int((0.95*self.eq.r[-1]-1.05*self.eq.r[0])/dr));
            bnt = bnf(xx2,zz);
            minb= np.min(bnt); minbloc = np.argmin(bnt);
            epx = (xx2[minbloc]-xx)/xx; xx = xx2[minbloc];
            count1 += 1

        self.xpnt_R[0] = xx; self.xpnt_Z[0] = zz; self.xpnt_P[0] = self.eq.psif(self.xpnt_R[0],self.xpnt_Z[0])

        #find lower-xpnt
        epx = 1.e0; epz = 1.e0; xx = rmin; zz= zmin;
        while ((epx > eps or epz > eps) and count2 < 100):
            
            zz2 = np.linspace(1.05*self.eq.z[0],zmin,int((zmin-1.05*self.eq.z[0])/dz));
            bnt = bnf(xx,zz2);
            minb= np.min(bnt); minbloc = np.argmin(bnt)
            epz = (zz2[minbloc]-zz)/zz; zz = zz2[minbloc];
            
            xx2 = np.linspace(1.05*self.eq.r[0],0.95*self.eq.r[-1],int((0.95*self.eq.r[-1]-1.05*self.eq.r[0])/dr));
            bnt = bnf(xx2,zz);
            minb= np.min(bnt); minbloc = np.argmin(bnt);
            epx = (xx2[minbloc]-xx)/xx; xx = xx2[minbloc];
            count2 += 1

        if count1==100: print('>>> Failed to find upper xpoint')
        if count2==100: print('>>> Failed to find lower xpoint')
        self.xpnt_R[1] = xx; self.xpnt_Z[1] = zz; self.xpnt_P[1] = self.eq.psif(self.xpnt_R[1],self.xpnt_Z[1])
        self.xpnt_P    = (self.xpnt_P - self.eq.smag) / (self.eq.sbdy - self.eq.smag)

        print('>>> RZ-Xpoint upper [%f,%f]/ lower [%f,%f] in [m]'%(self.xpnt_R[0],self.xpnt_Z[0],self.xpnt_R[1],self.xpnt_Z[1]))
        if self.xpnt_P[0] > self.xpnt_P[1]: print('>>> Active xpoint = lower')
        if self.xpnt_P[0] < self.xpnt_P[1]: print('>>> Active xpoint = upper')

    def _get_psival(self):
        ntht = self.in_param['ntht']; rg0 = self.in_param['rg0']; zg0 = self.in_param['zg0']; ellip = self.in_param['ellip']; amin = self.in_param['amin'];
        tria_l = self.in_param['tria_l']; tria_u = self.in_param['tria_u']; quad_l = self.in_param['quad_l']; quad_u = self.in_param['quad_u']
        if np.mod(ntht,2) == 0: ntht += 1
        ntht2 = int((ntht+1)/2)
        self.in_param['ntht'] = ntht;
        self.inR=np.zeros(ntht); self.inZ=np.zeros(ntht);

        for i in range(ntht2):
            angle = i*2.*np.pi/(ntht-1)
            self.inR[i] = rg0 + amin*np.cos(angle+tria_u*np.sin(angle)-quad_u*np.sin(2*angle));
            self.inZ[i] = zg0 + amin*ellip*np.sin(angle);

        for i in range(ntht2,ntht):
            angle = i*2.*np.pi/(ntht-1)
            self.inR[i] = rg0 + amin*np.cos(angle+tria_l*np.sin(angle)-quad_l*np.sin(2*angle));
            self.inZ[i] = zg0 + amin*ellip*np.sin(angle);

        if self.in_param['use_extbnd']:
            self._read_ext_bnd_file()
            extR = self.bndR; extZ = self.bndZ;
            maxv = np.max(extR); maxloc = np.argmax(extR)
            a0   = 0.5*(np.max(extR)-np.min(extR))
            cent = 0.5*(np.max(extR)+np.min(extR))-0.3*a0;
            ext_theta = np.arctan2(extZ-extZ[maxloc],extR-cent);
            ext_rr = np.sqrt((extZ-extZ[maxloc])**2+(extR-cent)**2);
            for i in range(len(ext_theta)):
                if (extZ[i]-extZ[maxloc]) <0.: ext_theta[i] = ext_theta[i] + 2*np.pi;
            ext_theta[-1] = ext_theta[-1] + 2.*np.pi;

            rrf = interp1d(ext_theta,ext_rr,'cubic')

            for i in range(ntht):
                angle = i*2.*np.pi/(ntht-1)
                self.inR[i] = cent + rrf(angle)*np.cos(angle);
                self.inZ[i] = extZ[maxloc] + rrf(angle)*np.sin(angle);
        self.inpsi      = np.copy(self.inR)
        for i in range(len(self.inpsi)): self.inpsi[i]      = self.eq.psif(self.inR[i],self.inZ[i])

    def _get_inputs(self):

        if not os.path.isfile(self.in_param['gfile']): print('>>> No g-file'); exit()
        self._read_gfile()
        if not os.path.isfile(self.in_param['pfile']): print('>>> No p-file'); exit()
        self._read_pfile()

        if not self.vt_in_pfile: 
            print('>>> No Vtor in p-file')
            if not os.path.isfile(self.in_param['rot_file']): print('>>> No rot-file'); exit()
            self._read_rot_file()

        if self.in_param['use_extbnd']:
            if not os.path.isfile(self.in_param['extbnd_file']): print('>>> No bnd-file'); exit()

    def _read_gfile(self):
        print('>>> Read g-file')
        self.eq = jin.eqdsk(self.in_param['gfile'])
        self._get_xpoint()

    def _read_pfile(self):
        print('>>> Read p-file')
        self.p  = dict();
        f = open(self.in_param['pfile'],'r')
        nline = int(f.readline())
        line  = f.readline().split()
        self.zeff  = float(line[0])
        self.mass_number = float(line[2])
        line  = f.readline().split()
        if len(line)<=5: self.vt_in_pfile = False

        self.p['psi_norm'] = np.zeros(nline)
        self.p['te'] = np.zeros(nline)
        self.p['ne'] = np.zeros(nline)
        self.p['ti'] = np.zeros(nline)
        self.p['ni'] = np.zeros(nline)
        if self.vt_in_pfile: 
            self.p['vt'] = np.zeros(nline)
            [self.p['psi_norm'][0],self.p['te'][0],self.p['ne'][0],self.p['ti'][0],self.p['ni'][0],self.p['vt'][0]] = np.array(line,dtype='float')
        else:
            [self.p['psi_norm'][0],self.p['te'][0],self.p['ne'][0],self.p['ti'][0],self.p['ni'][0]] = np.array(line,dtype='float')

        for i in range(1,nline):
            line  = f.readline().split()
            if self.vt_in_pfile: 
                [self.p['psi_norm'][i],self.p['te'][i],self.p['ne'][i],self.p['ti'][i],self.p['ni'][i],self.p['vt'][i]] = np.array(line,dtype='float')
            else:
                [self.p['psi_norm'][i],self.p['te'][i],self.p['ne'][i],self.p['ti'][i],self.p['ni'][i]] = np.array(line,dtype='float')
        f.close()

    def _read_rot_file(self):
        print('>>> Read rot-file')
        self.rot = dict()
        f = open(self.in_param['rot_file'],'r')
        nline = int(f.readline())
        self.rot['psi_norm'] = np.zeros(nline)
        self.rot['rr']       = np.zeros(nline)
        self.rot['vt']       = np.zeros(nline)
        line  = f.readline()
        for i in range(nline):
            line = f.readline.split()
            [self.rot['psi_norm'][i],self.rot['rr'][i],self.rot['vt'][i]] = np.array(line,dtype='float')
        f.close()
        vtf = interp1d(self.rot['psi_norm'],self.vt['vt'],'cubic')
        self.p['vt'] = vtf(self.p['psi_norm'])

    def _read_ext_bnd_file(self):
        print('>>> Read bnd-file')
        f = open(self.in_param['extbnd_file'],'r')
        bndR = []; bndZ = [];
        while True:
            line = f.readline()
            if not line: break
            bndR.append(line.split()[0])
            bndZ.append(line.split()[1])
        f.close()
        self.bndR = np.array(bndR,dtype='float')
        self.bndZ = np.array(bndZ,dtype='float')

    def _declare_variables(self):

        self.in_param    = dict()
        self.e0          = 1.60217662*1.e-19; 
        self.mu0         = 4.*np.pi*1.e-7;   
        self.proton_mass = 1.673 * 1.e-27;
        self.mass_number = 2.;
        self.zeff        = 2.;
        self.vt_in_pfile = True;
        self.psi_norm    = np.linspace(0,1.3,301)

        self.iseq        = False;
        self.isprof      = False;
        self.isready     = False;

    def _initialise_input_variables(self):

        #Input files
        self.in_param['gfile']              = 'g025607.006600_kin_3';
        self.in_param['pfile']              = 'p025607.006600_kin_3';
        self.in_param['rot_file']           = 'chease_vtor.txt';

        #Plasma boundary input
        self.in_param['use_extbnd']         = True
        self.in_param['double_null']        = False
        self.in_param['extbnd_file']        = 'bnd_file.txt'
        self.in_param['ntht']               = 501;
        self.in_param['ellip']              = 2.2;
        self.in_param['tria_u']             = 0.6;
        self.in_param['tria_l']             = 0.3;
        self.in_param['quad_u']             = 0.2;
        self.in_param['quad_l']             = 0.3;
        self.in_param['rg0']                = 1.7;
        self.in_param['zg0']                = 0.0;
        self.in_param['amin']               = 0.62;

        self.in_param['psi_open']           = 0.15;
        self.in_param['psi_inner']          = 0.05;
        self.in_param['psi_outer']          = 0.08;
        self.in_param['psi_private_up']     = 0.004;
        self.in_param['psi_private_low']    = 0.05;

        #FFprime extension
        self.in_param['ffpsig']             = 0.01;

        #Kinetic Profile extension
        self.in_param['use_gfile_p']        = True;
        self.in_param['use_ion_density']    = False;
        self.in_param['fit_debug']          = False;
        self.in_param['bnd_te']             = 0.01;
        self.in_param['bnd_ne']             = 0.2;
        self.in_param['bnd_ti']             = 0.01;
        self.in_param['bnd_vt']             = 2.;
        self.in_param['func_te']            = 'ptanh';
        self.in_param['func_ne']            = 'ptanh';
        self.in_param['func_ti']            = 'ptanh';
        self.in_param['func_vt']            = 'ptanh';

        #JOREK input option
        self.in_param['normalized_vtor']    = False;
        self.in_param['two_temperature']    = False;
        self.in_param['fitted_density']     = True;
        self.in_param['fitted_temperature'] = True;
        self.in_param['fitted_ffprime']     = True; # Always True

    def _read_namelist(self):
        if not os.path.isfile('injorek.in'): print('>>> No input file'); exit()
        f = open('injorek.in','r')
        while True:
            line = f.readline()
            if not line: break
            linet = line.split('!')[0]
            linet = linet.split('#')[0]
            line_n= linet.split('=')[0].split()[0].lower()
            line_v= linet.split('=')[1].split()[0].lower()
            if line_n in self.in_param.keys():
                self._put_namelist(line_n,line_v)
        f.close()

    def _put_namelist(self,var_key,value):
        var = self.in_param[var_key]; var2 = None;
        if isinstance(var,float): 
            try: var2 = float(value)
            except: pass
        if isinstance(var,bool):
            var2 = value.lower() in ("true")
        if isinstance(var,str): 
            var2 = value
        if not var2==None:
            if not var==var2: print('>>> %s changed from %s to %s'%(var_key,var,var2))
            self.in_param[var_key] = var2

    def _str2bool(self,v):

        return v.lower() in ("true")            

    def _check_input_variables(self):

        if self.in_param['fitted_density']: self.in_param['func_ne'] = 'ptanh'
        if self.in_param['fitted_temperature']: self.in_param['func_te'] = 'ptanh'; self.in_param['func_ti'] = 'ptanh'

    def _generate_input(self):

        f = open('injorek.in','w')
        for var in self.in_param.keys():
            f.write('%-20s= %-s\n'%(var,self.in_param[var]))
        f.close()

    def _show_input(self):
        print('>>> Input lists')
        print('--------------------------')
        for var in self.in_param.keys():
            print('>>> %-20s= %-s'%(var,self.in_param[var]))
        print('--------------------------')

class eqdsk:

    def __init__(self,filename):
        self.filename = filename;
        self._read_eqdsk(self.filename)
        self._make_grid()
        self._make_rho_R_psin()
        self._construct_volume()   
        print('>>> Construct Poloidal fields')
        self._construct_2d_field()        

    def _read_1d(self,file,num):
    
        dat = np.zeros(num)
        
        linen = int(np.floor(num/5))
        
        ii = 0
        
        for i in range(linen):
        
            line = file.readline()
            
            for j in range(5):
                
                dat[ii] = float(line[16*j:16*(j+1)])
                ii = ii + 1
        
        if not (num == 5*linen):
            line = file.readline()
        
        for i in range(num - 5*linen):
        
            dat[ii] = float(line[16*i:16*(i+1)])
            ii = ii + 1
            
        return (dat)

    def _read_colum(self,file,num):
    
        dat = np.zeros(shape=(num,2))
        
        dat1 = self._read_1d(file,num*2)
        
        for i in range(num):
        
            dat[i,0] = dat1[2*i]
            dat[i,1] = dat1[2*i+1]
            
        return (dat)
        
    def _read_2d(self,file,num1,num2):
    
        dat = np.zeros(shape=(num1,num2))
        
        dat1 = self._read_1d(file,num1*num2)
        
        ii = 0
        
        for i in range(num1):
        
            for j in range(num2):
            
                dat[i,j] = dat1[ii]
                ii = ii + 1
                
        return (dat)

    def _read_eqdsk(self,filename):
    
        file = open(filename,'r')
        
        line = file.readline().split()#
        linen = len(line)

        self.id = []
    
        if linen > 4:
            for i in range(linen-4):
                self.id.append(line[i])
        self.shotn = 0
        try:
            if (self.id[0].find('EFIT')>-1):
                self.shotn = int(self.id[3])
        except:
            pass

        self.idum = int(line[linen-3])  
        self.nw = int(line[linen-2])
        self.nh = int(line[linen-1])
        
        line = file.readline()#
        self.rdim = float(line[0:16])
        self.zdim = float(line[16:32])
        self.rcentr = float(line[32:48])
        self.rleft = float(line[48:64])
        self.zmid = float(line[64:80])
        
        line = file.readline()#
        self.rmag = float(line[0:16])
        self.zmag = float(line[16:32])
        self.smag = float(line[32:48])
        self.sbdy = float(line[48:64])
        self.bcentr = float(line[64:80])
        
        line = file.readline()#
        self.ip = float(line[0:16])
        self.xdum = float(line[32:48])
        
        line = file.readline()#
        
        self.fpol = self._read_1d(file,self.nw)
        self.pres = self._read_1d(file,self.nw)
        self.ffp = self._read_1d(file,self.nw)
        self.pp = self._read_1d(file,self.nw)
        self.psirz = self._read_2d(file,self.nh,self.nw)
        self.q = self._read_1d(file,self.nw)
        
        line = file.readline().split()
        self.nbbbs = int(line[0])
        self.limitr = int(line[1])
        
        self.rzbdy = self._read_colum(file,self.nbbbs)
        self.rzlim = self._read_colum(file,self.limitr)
        
        self.rzbdy[:,1] = self.rzbdy[:,1] 
        self.rzlim[:,1] = self.rzlim[:,1] 
        
        return
        
    def _make_grid(self):
    
        self.R = np.zeros(self.nw)
        self.Z = np.zeros(self.nh)
        self.psin = np.linspace(0,1,self.nw)
    
        for i in range(self.nw):
            self.R[i] = self.rleft + float(i)*self.rdim/(self.nw-1)
        for i in range(self.nh):
            self.Z[i] = self.zmid + float(i)*self.zdim/(self.nh-1) - self.zdim / 2.
            self.Z[i] = self.Z[i]
            
        self.RR, self.ZZ = np.meshgrid(self.R,self.Z)
            
        return
    
    def _contour(self,psi_norm):
        psi = psi_norm * (self.sbdy-self.smag) + self.smag
        cs = measure.find_contours(self.psirz,psi)
        for cc in cs:
            cc[:,1] = cc[:,1] * (max(self.R)-min(self.R))/(len(self.R)-1.) + min(self.R)
            cc[:,0] = cc[:,0] * (max(self.Z)-min(self.Z))/(len(self.Z)-1.) + min(self.Z)
        rz = np.copy(cs[0])
        rz[:,0] = cs[0][:,1]
        rz[:,1] = cs[0][:,0]
        
        return cs

    def _make_rho_R_psin(self):
    
        self.prhoR = np.zeros(shape=(201,4))
        
        self.prhoR[:,0] = np.linspace(0,1.0,201)
        
        RR = np.linspace(self.rmag,max(self.R)*0.999,301)
        RR2 = np.linspace(min(self.R)*1.001,self.rmag,301)
        
        psif = interp2d(self.R,self.Z,self.psirz)
        
        psir = psif(RR,self.zmag)
        psir2 = psif(RR2,self.zmag)

        psirn = np.zeros(301)
        psirn2 = np.zeros(301)
        
        for i in range(301):
            psirn[i] = (psir[i]-self.smag)/(self.sbdy - self.smag)
            psirn2[i] = (psir2[i]-self.smag)/(self.sbdy - self.smag)

        psirn[0] = 0.0;
        psirn2[-1] = 0.0;
        
        prf = interp1d(psirn,RR,'cubic')
        prf2 = interp1d(psirn2,RR2,'cubic')
        
        self.prhoR[:,2] = prf(self.prhoR[:,0])
        self.prhoR[0,2] = self.rmag

        self.prhoR[:,3] = prf2(self.prhoR[:,0])
        self.prhoR[0,3] = self.rmag
        
        qf = interp1d(self.psin,self.q,'cubic')
        q = qf(self.prhoR[:,0])
    
        for i in range(200):
            self.prhoR[i+1,1] = np.trapz(q[0:i+2],x=self.prhoR[0:i+2,0])
        for i in range(201):
            self.prhoR[i,1] = np.sqrt(self.prhoR[i,1] / self.prhoR[-1,1])
            
    
        rhof = interp1d(self.prhoR[:,0],self.prhoR[:,1],'slinear')
        self.rho = rhof(self.psin)

        return

    def _construct_volume(self):

        len1 = len(self.rzbdy)
        len2 = 101
        psin = np.linspace(0,1.,len2)
        self.avolp = np.zeros(shape=(len2,3))
        
        self.psif = interp2d(self.R,self.Z,self.psirz,'cubic')
        r = np.zeros(shape=(len1,len2))
        z = np.zeros(shape=(len1,len2))
        for i in range(len1):
            rr = np.linspace(self.rmag,self.rzbdy[i,0],len2)
            zz = (self.rzbdy[i,1]-self.zmag)/(self.rzbdy[i,0]-self.rmag)*(rr-self.rmag) + self.zmag
            psi = np.zeros(len2)
            for j in range(len2):
                psi[j] = (self.psif(rr[j],zz[j])-self.smag)/(self.sbdy-self.smag)

            psi[0] = 0.0;
            psi[-1] = 1.0;
            psifr = interp1d(psi,rr,'cubic')
            r[i,:] = psifr(psin)
            z[i,:] = (self.rzbdy[i,1]-self.zmag)/(self.rzbdy[i,0]-self.rmag)*(r[i,:]-self.rmag) + self.zmag

        sum1 = 0.
        sum2 = 0.
        sum3 = 0.
        for i in range(len2-1):
            for j in range(len1):
                i1 = i + 1
                j1 = j + 1
                if (j1 == len1):
                    j1 = 0

                dx1 = r[j1,i1] - r[j,i]
                dz1 = z[j1,i1] - z[j,i]
                dx2 = r[j1,i]  - r[j,i1]
                dz2 = z[j1,i]  - z[j,i1]
                dx3 = r[j1,i1]  - r[j,i1]
                dz3 = r[j1,i1]  - r[j,i1]

                dl1 = np.sqrt(dx1**2 + dz1**2)
                dl2 = np.sqrt(dx2**2 + dz2**2)
                cos = (dx1*dx2 + dz1*dz2)/dl1/dl2

                if (abs(cos) > 1):  cos = 1.
                sin = np.sqrt(1.0 - cos**2)
                dA = 0.5*dl1*dl2*sin
                Rc = 0.25*(r[j,i]+r[j1,i]+r[j,i1]+r[j1,i1])
                Zc = 0.25*(z[j,i]+z[j1,i]+z[j,i1]+z[j1,i1])
                sum1 = sum1 + dA
                sum2 = sum2 + 2.*np.pi*Rc*dA

            self.avolp[i+1,0] = psin[i+1]
            self.avolp[i+1,1] = sum1
            self.avolp[i+1,2] = sum2

        pref = interp1d(self.psin,self.pres,'cubic')
        pres = pref(psin)

        self.wmhd = np.trapz(pres,x=self.avolp[:,2]) * 1.5;
        self.area = self.avolp[-1,1]
        self.vol  = self.avolp[-1,2]
        self.pva = self.wmhd / 1.5 / self.avolp[-1,2]

        return
        
    def _construct_2d_field(self):

        self.br = np.zeros(shape=(self.nh-2,self.nw-2))
        self.bz = np.copy(self.br)
        self.bt = np.copy(self.br)
        self.r = np.zeros(self.nw-2)
        self.z = np.zeros(self.nh-2)

        for i in range(self.nh-2):
            self.z[i] = self.Z[i+1]
        for i in range(self.nw-2):
            self.r[i] = self.R[i+1]

        fpf = interp1d(self.psin,self.fpol,'cubic')

        for i in range(self.nh-2):
            Z = self.Z[i+1]
            for j in range(self.nw-2):
                R = self.R[j+1]
                psi = (self.psirz[i+1,j+1]-self.smag) / (self.sbdy - self.smag)
                if (psi < 0.0): psi = 0.0

                self.br[i,j] = - (self.psirz[i+2,j+1] - self.psirz[i,j+1]) / (self.Z[i+2] - self.Z[i]) / R
                self.bz[i,j] = + (self.psirz[i+1,j+2] - self.psirz[i+1,j]) / (self.R[j+2] - self.R[j]) / R

                if (psi < 1.0): self.bt[i,j] = fpf(psi) / R
                else: self.bt[i,j] = self.fpol[-1] / R
        return  
        
