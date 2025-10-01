import os
import glob
from functools import partial
import numpy as np
from mpmath import polylog
from numba import njit
from scipy import integrate
from astropy import units as ut
from astropy import constants as ct

###############################################################################

@njit
def TDMA_solver(aa,ba,ca,da_in,nf):
    """
    Tridiagonal matrix algorithm.
    """
    dc = -da_in # d needs to be on other side of hydrostatic equation for TDMA
    for it in range(1, nf):
        wa = aa[it-1]/ba[it-1]
        ba[it] = ba[it] - wa*ca[it-1]
        dc[it] = dc[it] - wa*dc[it-1]
    xa = ba
    xa[-1] = dc[-1]/ba[-1]
    for il in range(nf-2, -1, -1):
        xa[il] = (dc[il]-ca[il]*xa[il+1])/ba[il]
    return xa

@njit
def function_YukawaBornViscosityApprox_order1(s,par0,par1,par2,par3,t_channel_only):
    """ Function used for average_YukawaBornViscosityApprox(). """
    s2 = s*s + 1e-4
    s4 = s*s*s*s + 1.e-8
    sn = s2 if t_channel_only else s4
    # gives correct large s limit = (np.log(s**4)+np.log(par1))/(par0*s**4)
    # gives correct large s limit = (np.log(s**2)+np.log(par1))/(par0*s**4) if t_channel_only
    g = 1.5*np.log( 1. + (sn * par1)**par2 ) / (par0 * par2 * s4 )
    return (1. + 1./g**par3)**(-1./par3)

def average_YukawaBornViscosityApprox(index,order,t_channel_only):
    """ Approximate form of Kp (Eq. 5 of 2204.06568). """
    if t_channel_only:
        params_dict = {3: [ 8.,0.303941,0.74,0.68],
                       5: [24.,0.826198,0.82,0.76],
                       7: [48.,1.36217, 0.83,0.79],
                       9: [80.,1.90106, 0.84,0.81]}
        params = params_dict[index]
    else:
        params_dict = {3: [ 8.,0.0339848,0.37,0.63],
                       5: [24.,0.251115, 0.41,0.71],
                       7: [48.,0.682602, 0.42,0.74],
                       9: [80.,1.32953,  0.43,0.76]}
        params = params_dict[index]

    if order==1 and index in params_dict.keys():
        output = partial(function_YukawaBornViscosityApprox_order1,
                         par0=params[0],par1=params[1],par2=params[2],par3=params[3],
                         t_channel_only=t_channel_only)
    elif order==2 and index==5:
        params = params_dict[5]
        K5 = partial(function_YukawaBornViscosityApprox_order1,
                     par0=params[0],par1=params[1],par2=params[2],par3=params[3],
                     t_channel_only=t_channel_only)
        params = params_dict[7]
        K7 = partial(function_YukawaBornViscosityApprox_order1,
                     par0=params[0],par1=params[1],par2=params[2],par3=params[3],
                     t_channel_only=t_channel_only)
        params = params_dict[9]
        K9 = partial(function_YukawaBornViscosityApprox_order1,
                     par0=params[0],par1=params[1],par2=params[2],par3=params[3],
                     t_channel_only=t_channel_only)
        output = lambda x: (28.*K5(x)*K5(x) + 80.*K5(x)*K9(x) - 64.*K7(x)*K7(x)) / (77.*K5(x) - 112.*K7(x) + 80.*K9(x))
    else:
        raise IOError('No approximation for Yukawa Born model with index {} and order {}'.format(index,order))
    return output

###############################################################################
###############################################################################
###############################################################################

class Halo:
    """
    Main class to define a dark matter halo with self-interacting dark matter
    and calculate its gravothermal evolution.
    """
    def __init__(self,record,**kwargs):
        """
        Initialize dark matter halo to run gravothermal evolution.

        Parameters
        ----------
        record: `HaloRecord` instance
            Handles file I/O for this instance of `Halo`.

        **kwargs: dictionary, optional
            Dictionary of inputs to override default halo values.
            Possible inputs are listed below.

        Inputs for halo profile
        -----------------------
        profile: 'NFW', 'Hernquist', or 'Isothermal', default: 'NFW'
            Specify halo density profile.

        r_s: float, default: 2.586
            Scale radius of halo in units of [kpc].

        rho_s: float, default: 0.0194
            Scale density in units of [M_sun/pc^3].

        Inputs for fluid properties
        ---------------------------
        a: float, default: 4/sqrt(pi)
            Coefficient for hard-sphere scattering.

        b: float, default: 25*sqrt(pi)/32
            Effective impact parameter.

        C: float, default: 0.753
            Calibration parameter for LMFP thermal conductivity kappa. Must be calibrated to simulation.

        gamma: float, default: 5/3
            Gas constant for monatomic gas.

        Inputs for particle properties
        ------------------------------
        model_elastic_scattering_lmfp: string, default: 'constant'
            Particle physics model for the scattering cross section in the LMFP
            regime. See `initialize_elastic_scattering()` for details.

        model_elastic_scattering_smfp: string, default: 'constant'
            Particle physics model for the scattering cross section in the SMFP
            regime. See `initialize_elastic_scattering()` for details.

        sigma_m_with_units: float, default: 5 [cm^2/g]
            Cross section prefactor for elastic scattering.

        w_units: float, default: 1 [km/s]
            Velocity scale for elastic scattering.

        Inputs for numerical solving
        ----------------------------
        n_shells: integer, default: 400
            Number of shells to divide the halo into.

        r_min: float, default: 0.01
            Minimum dimensionless radius of radial bins.

        r_max: float, default: 100
            Maximum dimensionless radius of radial bins.

        p_rmax_factor: float, default: 10
            Factor to multiply r_max to obtain upper limit of integral when
            numerically integrating to find initial halo pressure.

        n_adjustment_fixed: integer, default: -1
            Number of forced hydrostatic adjustment steps per heat conduction
            steps. A negative value indicates a fixed number should not be used
            and this parameter is ignored. The default behavior is to determine
            the number of steps dynamically to satisfy a convergence criterion
            set at run time; see implementation of `hydrostatic_adjustment()`.

        n_adjustment_max: integer, default: -1
            Number of maximum hydrostatic adjustment steps per heat conduction
            steps. A negative value indicates a maximum should not be enforced
            and this parameter is ignored. The default behavior is to determine
            the number of steps dynamically to satisfy a convergence criterion
            set at run time; see implementation of `hydrostatic_adjustment()`.

        flag_timestep_use_relaxation: bool, default: True
            If True, incorporate the minimum local relaxation time of the halo
            to help set the heat conduction time step.

        flag_timestep_use_energy: bool, default: True
            If True, incorporate the relative specific energy change
            to help set the heat conduction time step.

        flag_hydrostatic_initial: bool, default: False
            If True, perform hydrostatic adjustment to initialized halo
            at t=0 before beginning halo evolution process.
        """
        ##### save HaloRecord class
        self.record = record

        ##### quantities to initialize halo
        # these values are inherent to a given halo calculation and
        # should not be altered at any point during halo evolution
        self.halo_ini = {
            'profile': 'NFW',
            'r_s': 2.586, # [kpc]
            'rho_s': 0.0194, # [M_sun/pc^3]
            'a': 4./np.sqrt(np.pi),
            'b': 25.*np.sqrt(np.pi)/32.,
            'C': 0.753,
            'gamma': 5./3.,
            'model_elastic_scattering_lmfp': 'constant',
            'model_elastic_scattering_smfp': 'constant',
            'sigma_m_with_units': 5., # [cm^2/g]
            'w_units': 1., # [km/s]
            'n_shells': 400,
            'r_min': 0.01,
            'r_max': 100.,
            'p_rmax_factor': 10.,
            'n_adjustment_fixed': -1,
            'n_adjustment_max': -1,
            'flag_timestep_use_relaxation': True,
            'flag_timestep_use_energy': True,
            'flag_hydrostatic_initial': False
        }

        ##### overwrite halo initialization defaults
        if self.record.has_record: # saved initialization information exists
            halo_ini_input, original_state = self.record.get_halo_initialization()
            if kwargs!={}:
                print('Warning: all user input is ignored for initialization (using settings from existing file).')
        else: # no previously saved information exists
            halo_ini_input = kwargs
            original_state = {}

        # handle bad user input or backward compatibility issues
        bad_keys = np.setdiff1d(list(halo_ini_input.keys()),list(self.halo_ini.keys()))
        if len(bad_keys) > 0:
            if self.record.has_record:
                print('Warning: ignoring the following obsolete inputs in the saved initialization file: '+', '.join(bad_keys))
            else:
                print('Warning: ignoring the following unknown inputs: '+', '.join(bad_keys))
        for key in bad_keys:
            halo_ini_input.pop(key)

        # update parameters from either saved file or user input
        self.halo_ini.update(halo_ini_input)

        # verify user input and perform checks for new halo
        if not self.record.has_record:
            # ensure certain quantities are integers
            self.halo_ini['n_adjustment_fixed'] = round(self.halo_ini['n_adjustment_fixed'])
            self.halo_ini['n_adjustment_max']   = round(self.halo_ini['n_adjustment_max'])

            # ensure at least one criterion for defining the time step
            if (self.halo_ini['flag_timestep_use_relaxation'] is False) and (self.halo_ini['flag_timestep_use_energy'] is False):
                raise IOError('At least one time step criterion must be used.')

            # ensure criteria for number of adjustment steps do not conflict
            if (self.halo_ini['n_adjustment_fixed'] > -1) and (self.halo_ini['n_adjustment_max'] > -1):
                raise IOError('Cannot set both a fixed and max number of hydrostatic adjustments. Set only one or none.')

        # set class attributions
        for key, value in self.halo_ini.items():
            setattr(self, key, value)

        ##### handle units (dimensionful inputs: r_s, rho_s, sigma_m_with_units)
        # dimensionful halo scales
        self.scale_r   = self.r_s * ut.kpc # scale radius
        self.scale_rho = self.rho_s * ut.M_sun/ut.pc**3 # scale density
        self.scale_m   = 4.*np.pi * self.scale_rho * self.scale_r**3 # mass scale
        self.scale_u   = ct.G * self.scale_m / self.scale_r # specific energy scale
        self.scale_p   = self.scale_u * self.scale_rho # pressure scale
        self.scale_v   = np.sqrt(self.scale_u) # velocity dispersion scale
        self.scale_t   = 1./np.sqrt(4.*np.pi*self.scale_rho * ct.G) # dynamical time scale
        self.scale_L   = ct.G * self.scale_m**2 / (self.scale_r * self.scale_t) # luminosity scale
        self.scale_sigma_m = 1./(self.scale_r * self.scale_rho) # cross section per mass scale

        # convert dimensionful quantities to be dimensionless
        self.sigma_m = (self.sigma_m_with_units*ut.cm**2/ut.g).to_value(self.scale_sigma_m)
        self.w = (self.w_units*ut.km/ut.s).to_value(self.scale_v)

        ##### initialize quantities for evolution, saved in snapshot files
        # this list should coincide with that in save_halo()
        self.rho_center = 0    # initial density of innermost shell
        self.n_adjustment = 0  # counter for hydrostatic adjustment steps
        self.n_conduction = 0  # counter for heat conduction steps
        self.n_save = 0        # counter for total saved files
        self.t_epsilon = None  # tolerance level for heat conduction time step
        self.r_epsilon = None  # tolerance level for satisfying hydrostatic adjustment

        self.t = 0        # current dimensionless time
        self.t_before = 0 # dimensionless time from previous time step

        self.r = None   # array of outer radii for each shell
        self.m = None   # total mass enclosed within radii, given by r
        self.rho = None # average density for each shell
        self.p = None   # average pressure for each shell

        ##### initialize derived quantities (not saved to file, set in update_derived_parameters)
        # generic quantities needed for evolution
        self.u = None # average specific energy for each shell
        self.v = None # average 1d velocity dispersion for each shell
        self.L = np.empty(self.n_shells,dtype=np.float64) # luminosity at radii, given by r

        # quantities for elastic scattering
        # define variables that represent velocity-dependent functions
        self.F_elastic_lmfp = self.initialize_elastic_scattering(self.model_elastic_scattering_lmfp)
        self.F_elastic_smfp = self.initialize_elastic_scattering(self.model_elastic_scattering_smfp)

        # other useful quantities
        self.Kn = None # Knudsen number
        self.Kinv_lmfp = None # lmfp conductivity (inverse)
        self.Kinv_smfp = None # smfp conductivity (inverse)

        ##### handle runtime quantities (not saved or used for evolution)
        # relaxation time scale (in units of scale_t), as defined in arXiv 2204.06568
        scale_t_relax = 2./(3. * self.a * self.C * self.sigma_m_with_units*ut.cm**2/ut.g * self.F_elastic_lmfp(1./self.w) * self.scale_v * self.scale_rho)
        self.t_relax = (scale_t_relax).to_value(self.scale_t)

        ##### either load existing halo save state or create a new one
        load_success = False

        # attempt to load most recently saved halo state
        if self.record.has_record:
            load_success = self.load_halo(None,time=None,verbose=True)

        # otherwise, check for unmodified initial state
        if not load_success and original_state != {}:
            for key,value in original_state.items():
                setattr(self,key,value)

            # store initial density of innermost shell
            self.rho_center = self.get_central_quantity(self.rho)

            # with necessary quantities initialized, set derived quantities
            self.update_derived_parameters()

            # print information
            print('~~~~~ Loaded original initial halo state for {}'.format(self.record.basename))
            load_success = True

        #  otherwise, set initial halo radius, mass, density, and pressure profiles
        if not load_success:
            # initial radius, location of outermost edge of each shell
            self.r = np.logspace(np.log10(self.r_min),np.log10(self.r_max),
                                 num=self.n_shells,endpoint=True,base=10)

            # location of midpoints of shells
            r_mid = self.get_shell_midpoints()

            # set mass
            self.m = self.get_initial_mass(self.r)

            # set rho
            self.rho = np.empty(self.n_shells,dtype=np.float64)
            self.rho[0] = 3.*self.m[0]/self.r[0]**3
            self.rho[1:] = self.get_initial_rho(r_mid[1:])

            # set pressure
            self.p = self.get_initial_pressure(r_mid,numeric=False)
            self.p[0] = self.get_initial_pressure(self.r[0],numeric=False)

            # store initial density of innermost shell
            self.rho_center = self.get_central_quantity(self.rho)

            # with necessary quantities initialized, set derived quantities
            self.update_derived_parameters()

        ##### initialize arrays needed for evolution (not saved to file)
        self.r_ext = np.empty(self.n_shells+1, dtype=np.float64) # radius array, extended by 1
        self.m_ext = np.empty(self.n_shells+1, dtype=np.float64) # mass array, extended by 1
        self.a_arr = np.empty(self.n_shells-2, dtype=np.float64) # tridiagonal matrix element array
        self.b_arr = np.empty(self.n_shells-1, dtype=np.float64) # tridiagonal matrix element array
        self.c_arr = np.empty(self.n_shells-2, dtype=np.float64) # tridiagonal matrix element array
        self.d_arr = np.empty(self.n_shells-1, dtype=np.float64) # tridiagonal matrix element array
        self.delta_r   = np.empty(self.n_shells, dtype=np.float64) # shell location change for adjustment
        self.delta_uc  = np.empty(self.n_shells, dtype=np.float64) # specific energy change for adjustment
        self.delta_rho = np.empty(self.n_shells, dtype=np.float64) # density change for adjustment
        self.delta_ph  = np.empty(self.n_shells, dtype=np.float64) # pressure change for adjustment

        ##### save initialization file, if needed
        if not self.record.has_record:
            save_names = ['m','r','rho','p']
            original_state = {n: getattr(self,n) for n in save_names}

            print('~~~~~ Creating new halo {}'.format(self.record.basename))
            self.record.save_halo_initialization(self.halo_ini,original_state)

        return

    def load_halo(self,file_halo,time=None,verbose=True):
        """
        Load pickled halo information saved by `save_halo()`.

        If loading fails (due to no pickle files existing) but an archive file
        exists, obtain halo state from archived data.

        Parameters
        ----------
        file_halo: string
            Name of pickle file to load in order recover saved halo state.
            Provide the file name only, not the full path location.
            Input None to use input time instead.

        time: float, default: None
            If None, recover most recently saved halo state. Otherwise,
            recover halo state that corresponds most closely to input time.

        verbose: bool, default: True
            If True, print information about status of loading halo state file.

        Returns
        -------
        Bool to indicate if loading data was successful.
        """

        # retrive information from pickled data
        data = self.record.get_halo_state_pickled(file_halo=file_halo,time=time)
        source_str = 'pickled'

        # if there are no pickle files, try obtaining archive data
        if data=={}:
            # try obtaining pickle file
            if file_halo is not None:
                file_time = self.record.extract_filename_time(file_halo)
            else:
                file_time = time
            data = self.record.get_halo_state_archived(time=file_time)
            source_str = 'archived'

        # if there are no files of saved halo states, return failure to load
        if data=={}:
            if verbose:
                print('~~~~~ No halo state was loaded')
            return False

        # print information
        if verbose:
            print('~~~~~ Recovered {} halo state for {}, with t={}'.format(source_str,self.record.basename,data['t']))

        # set class attributes
        for key,value in data.items():
            setattr(self,key,value)
        # recover derived parameters if halo save state is loaded
        self.update_derived_parameters()

        return True

    def save_halo(self,prefix=None):
        """
        Save halo information to pickle file.
        See `save_halo_state_pickled()` of `HaloRecord` for details.

        Parameters
        ----------
        prefix: string, default: None
            File name prefix for save state of the halo.
            If None, use default prefix defined in `HaloRecord`.
        """
        # names of variables to save (derived parameters can be recovered)
        save_names = ['rho_center','n_adjustment','n_conduction','n_save',
                      't_epsilon','r_epsilon','t','t_before','m','r','rho','p']

        # if prefix is None, set it to default
        if prefix is None:
            prefix = self.record.prefix_default

        # update the counter if prefix is default
        if prefix==self.record.prefix_default:
            self.n_save += 1

        # save data
        data = {n: getattr(self,n) for n in save_names}
        self.record.save_halo_state_pickled(prefix,self.t,data)

        return

    def get_central_quantity(self,x):
        """
        Obtain input quantity (with length of 'n_shells') at halo center. This
        function defines what is meant by 'center' in order to maintain
        consistency throughout the code.
        """
        return x[3]

    def get_shell_midpoints(self):
        """
        Obtain radii of central location of shells. Midpoints of shells are
        taken to be the linear average of the outer and inner shell radii.
        """
        r_mid = np.empty(self.n_shells,dtype=np.float64)
        r_mid[0] = self.r[0]/2.
        r_mid[1:] = (self.r[:-1]+self.r[1:])/2.
        return r_mid

    def get_initial_mass(self,x):
        """
        Obtain halo mass profile.
        """
        if self.profile == 'NFW':
            mass = -x/(1.+x) + np.log(1.+x)
        elif self.profile == 'Hernquist':
            mass = x**2/(2.*(1.+x)**2)
        elif self.profile == 'Isothermal':
            mass = x - np.arctan(x)
        else:
            raise IOError('Profile {} is not recognized for mass calculation'.format(self.profile))
        return mass

    def get_initial_rho(self,x):
        """
        Obtain halo density profile.
        """
        if self.profile == 'NFW':
            rho = 1. / (x * (1.+x)**2)
        elif self.profile == 'Hernquist':
            rho = 1. / (x * (1.+x)**3)
        elif self.profile == 'Isothermal':
            rho = 1. / (1. + x**2)
        else:
            raise IOError('Profile {} is not recognized for density calculation'.format(self.profile))
        return rho

    def get_initial_pressure(self,x,numeric=False,xmax=None):
        """
        Obtain halo pressure profile.
        If numeric is True, calculate by integrating rather than using the
        analytic expression. Numeric integration extends to xmax.
        """
        if numeric:
            if xmax is None:
                xmax = self.p_rmax_factor*self.r_max
            p_integrand = lambda r: self.get_initial_mass(r)*self.get_initial_rho(r)/(r*r)
            it = np.nditer([x,None],flags=['buffered'],op_dtypes=np.float64)
            for (xi,y) in it:
                y[...] = integrate.quad(p_integrand,xi,xmax)[0]
            return it.operands[1]

        if self.profile == 'NFW':
            it = np.nditer([x,None],op_dtypes=np.float64)
            for (xi,y) in it:
                y[...] = polylog(2.,-xi)
            plog = it.operands[1]
            p = (np.pi**2.-(1.+x*(9.+7.*x))/(x*(1.+x)**2.)-np.log(x)+np.log(1.+x)*(1.+x**(-2.)-4./x-2./(1.+x)+3.*np.log(1.+x))+6.*plog)/2.
        elif self.profile == 'Hernquist':
            p = ( np.log(1.+1./x) - (25.+2.*x*(26.+3.*x*(7.+2.*x)))/(12.*(1.+x)**4) ) / 2.
        elif self.profile == 'Isothermal':
            p = np.pi**2 / 8. - np.arctan(x)*(2.+x*np.arctan(x))/(2.*x)
        else:
            raise IOError('Profile {} is not recognized for pressure calculation'.format(self.profile))
        return p

    def initialize_elastic_scattering(self,model_name):
        """
        Set appropriate velocity-dependent function F(v) for LMFP or SMFP
        regime to represent impact of particle physics model for elastic
        scattering on thermal conductivities:
            kappa_LMFP = 3aC/(8\pi G) * sigma0*F(v)/m_{DM}^2 * rho * v^3
            kappa_SMFP = (3/2) * b*v / (sigma0 * F(v))
        The functions F(v) for LMFP and SMFP are assigned to the variables
        F_elastic_lmfp and F_elastic_smfp.

        Model options:
        'constant': velocity-independent cross section with F(v)=1
        'powerlaw_n': power-law behavior F(v) = (v/w)^n for float n
        'YukawaBornViscosityApprox_Kp_orderN': Yukawa scattering
            under Born approximation, incorporating t- and u-channel
            F(v) = average_YukawaBornViscosityApprox(p,N,False)
            p = power-law index for velocity weighting
            N = order of calculation in standard SMFP case
        'YukawaBornViscosityApproxTchannel_Kp_orderN': Yukawa scattering
            under Born approximation, incorporating t-channel only
            F(v) = average_YukawaBornViscosityApprox(p,N,True)

        Parameters
        ----------
        model_name: string
            Model name for either LMFP or SMFP regime. Expect class parameter
            'model_elastic_scattering_lmfp' or 'model_elastic_scattering_smfp'
            as input here.

        Returns
        -------
        Function F(v).
        """
        if model_name == 'constant':
            output = lambda x: 1.
        elif 'powerlaw' in model_name:
            index = float(model_name.split('_')[1])
            output = lambda x: np.pow(x,index)
        elif 'YukawaBornViscosityApprox' in model_name:
            index = int(model_name.split('_')[1].lstrip('K'))
            order = int(model_name.split('_')[2].lstrip('order'))
            flag_t_channel = False
            if 'Tchannel' in model_name:
                flag_t_channel = True
            output = average_YukawaBornViscosityApprox(index,order,flag_t_channel)
        else:
            raise IOError('Elastic scattering model {} unknown'.format(model_name))

        return output

    def update_derived_parameters(self):
        """
        Given r, mass, rho, and p, set derived halo quantities.
        """
        # specific energy (energy per unit mass)
        self.u = (3./2.) * self.p/self.rho

        # 1d velocity dispersion
        self.v = np.sqrt(self.p/self.rho)

        # effective thermal conductivity, Keff = (2/3) \kappa m_{DM}
        # heat flux equation: L/(4\pi r^2) = -kappa \frac{\partial T}{\partial r}
        #                                  = - Keff \frac{\partial u}{\partial r}
        self.Kinv_smfp = self.sigma_m * self.F_elastic_smfp(self.v/self.w) / (self.b * self.v)
        self.Kinv_lmfp = 1. / (self.a * self.C * self.v * self.p * self.sigma_m * self.F_elastic_lmfp(self.v/self.w))
        Keff = 1./(self.Kinv_smfp + self.Kinv_lmfp)

        # luminosity
        self.L[1:-1] = -self.r[1:-1]*self.r[1:-1] * (Keff[1:-1]+Keff[2:])/2. * (self.u[2:]-self.u[1:-1])/((self.r[2:]-self.r[:-2])/2.)
        self.L[0] = -self.r[0]*self.r[0] * (Keff[0]+Keff[1])/2. * (self.u[1]-self.u[0])/(self.r[1]/2.)
        self.L[-1] = 0

        # Knudsen number
        self.Kn = 1. / (np.sqrt(self.p) * self.sigma_m)

        return

    def get_timestep(self):
        """
        Determine heat conduction time step.
        """
        delta_t1 = min(abs(self.u[0]/(self.L[0]/self.m[0])),
                       min(abs(self.u[1:] / ((self.L[1:]-self.L[:-1])/(self.m[1:]-self.m[:-1])))))
        delta_t2 = min(1. / (self.rho * self.v) )

        # determine minimum time step
        if self.flag_timestep_use_relaxation and self.flag_timestep_use_energy:
            delta_t = min(delta_t1, delta_t2)
        elif self.flag_timestep_use_relaxation:
            delta_t = delta_t2
        elif self.flag_timestep_use_energy:
            delta_t = delta_t1
        else:
            raise IOError('Something went wrong. At least one time step criterion must be used.')

        return self.t_epsilon*delta_t

    def conduct_heat(self):
        """
        Perform heat conduction step.
        """
        # update conduction counter
        self.n_conduction += 1

        # determine minimum time step
        delta_t = self.get_timestep()

        # calculate delta_uc
        self.delta_uc[1:] = -delta_t*((self.L[1:]-self.L[:-1])/(self.m[1:]-self.m[:-1]))
        self.delta_uc[0]  = -delta_t*(self.L[0]/self.m[0])

        # calculate delta_pc
        delta_pc = self.p * self.delta_uc / self.u

        # update variables
        self.p += delta_pc
        self.u += self.delta_uc

        # update the current dimensionless time
        self.t_before = self.t
        self.t += delta_t

        return

    def hydrostatic_adjustment_step(self):
        """
        Perform one hydrostatic adjustment process. Adiabaticity is satisified
        in the hydrostatic process after each conduction step. The tridiagonal
        matrix is derived by keeping the central and outermost shells fixed.
        """
        # update the counter
        self.n_adjustment += 1

        self.set_hydrostatic_coefficients()

        delta_r = TDMA_solver(self.a_arr,self.b_arr,self.c_arr,self.d_arr,self.n_shells-1)
        self.delta_r[:-1] = delta_r
        self.delta_r[-1]  = 0

        r2_times_dr = self.r*self.r * self.delta_r
        r3 = self.r*self.r*self.r

        # update delta_rho
        self.delta_rho[1:] = -self.rho[1:] * (r2_times_dr[1:]-r2_times_dr[:-1]) / ((r3[1:]-r3[:-1])/3.)
        self.delta_rho[0] = -self.rho[0] * r2_times_dr[0] / (r3[0]/3.)

        # update delta_ph
        self.delta_ph[1:] = -self.gamma * self.p[1:] * (r2_times_dr[1:]-r2_times_dr[:-1]) / ((r3[1:]-r3[:-1])/3.)
        self.delta_ph[0] = -self.gamma * self.p[0] * r2_times_dr[0] / (r3[0]/3.)

        # update r, rho, and p
        self.r += self.delta_r
        self.rho += self.delta_rho
        self.p += self.delta_ph

        return

    def hydrostatic_adjustment(self):
        """
        Perform series of hydrostatic adjustment steps to achieve hydrostatic
        condition for halo to specified tolerance. If the parameter
        n_adjustment_fixed>=0, use a fixed number of adjustment steps.
        If the parameter n_adjustment_max>=0, limit the max number of
        adjustment steps the code should perform.
        """
        # initialize number of adjustment steps taken
        self.n_adjustment = 0

        if self.n_adjustment_fixed==0 or self.n_adjustment_max==0:
            # perform no adjustments
            pass
        elif self.n_adjustment_fixed > 0:
            # perform fixed number of adjustments
            for idx in range(self.n_adjustment_fixed):
                self.hydrostatic_adjustment_step()
        else:
            self.hydrostatic_adjustment_step()
            # adjust until convergence is met
            while max(abs(self.delta_r/self.r)) > self.r_epsilon:
                if self.n_adjustment == self.n_adjustment_max:
                    # break if max adjustments is reached
                    break
                self.hydrostatic_adjustment_step()

        self.update_derived_parameters()
        return

    def set_hydrostatic_coefficients(self):
        """
        Set the elements of the tridiagonal matrix for solving the linearized
        hydrostatic equation.
        """
        # fill extended r and m arrays
        self.r_ext[1:] = self.r
        self.m_ext[1:] = self.m
        self.r_ext[0] = 0
        self.m_ext[0] = 0

        r2 = self.r_ext * self.r_ext
        r3 = r2 * self.r_ext

        self.a_arr[:] = (
            (12.*self.gamma*self.p[1:-1]*r2[1:-2]*r2[2:-1]
             + self.m_ext[2:-1] * (-3.*r2[1:-2]*self.r_ext[3:]*self.rho[1:-1]
                                   + r3[1:-2]*(2.*self.rho[1:-1]-self.rho[2:])
                                   + r3[2:-1]*(self.rho[1:-1]+self.rho[2:])))
            / (4.*(r3[1:-2]-r3[2:-1])))

        self.b_arr[:] = (
            self.r_ext[1:-1]
             * (-4.*self.p[:-1] * (2.*r3[:-2]+(-2.+3.*self.gamma)*r3[1:-1]) * (r3[1:-1]-r3[2:])
                - 4.*self.p[1:]*(r3[:-2]-r3[1:-1]) * ((-2.+3.*self.gamma)*r3[1:-1]+2.*r3[2:])
                + 3.*self.m_ext[1:-1]*self.r_ext[1:-1]*(self.r_ext[:-2]-self.r_ext[2:])
                * (r3[2:]*self.rho[:-1]+r3[:-2]*self.rho[1:] - r3[1:-1]*(self.rho[:-1]+self.rho[1:])))
            / (4.*(r3[:-2]-r3[1:-1]) * (r3[1:-1]-r3[2:])))

        self.c_arr[:] = (
            (12.*self.gamma*self.p[1:-1]*r2[1:-2]*r2[2:-1] + self.m_ext[1:-2]*(
                r3[1:-2]*(self.rho[:-2]+self.rho[1:-1])
                - r2[2:-1] * (self.r_ext[2:-1]*self.rho[:-2]+3.*self.r_ext[:-3]*self.rho[1:-1]
                              - 2.*self.r_ext[2:-1]*self.rho[1:-1])))
            / (4.*(r3[1:-2]-r3[2:-1])))

        self.d_arr[:] = (
            -self.p[:-1]*r2[1:-1]+self.p[1:]*r2[1:-1]
            + 1./4.*self.m_ext[1:-1]*(-self.r_ext[:-2]+self.r_ext[2:])
            * (self.rho[:-1]+self.rho[1:]))

        return

    def evolve_halo(self,t_end=np.inf,rho_factor_end=np.inf,Kn_end=0,
                    save_frequency_rate = None,
                    save_frequency_timing = None,
                    save_frequency_density = None,
                    t_epsilon=1.e-4,r_epsilon=1.e-14):
        """
        Run the halo evolution.

        Parameters
        ----------
        t_end: float, default: np.inf
            Maximum dimensionless time to run the halo evolution.
            (Set to np.inf to render this criterion ineffectual.)

        rho_factor_end: float, default: np.inf
            Factor to determine halo evolution termination, based on central
            density. If the central density reaches this factor times the
            initial central density, evolution stops.
            (Set to np.inf to render this criterion ineffectual.)

        Kn_end: float, default: 0
            Minimum Knudsen number to determine halo evolution termination.
            If the Knudsen number anywhere in the halo drops to this value
            or lower, evolution stops.
            (Set to 0 to render this criterion ineffectual.)

        save_frequency_rate: float, default: None
            Frequency in which to save halo files, in terms of (dimensionless)
            time. An input of N means that N save files will be created for
            each elapsed unit of time. If None, this criteria will not be used
            in saving files. Note: dimensionless time used is defined by the
            relaxation time scale, not the default time scale.

        save_frequency_timing: float, default: None
            Frequency in which to save halo files, in terms of the fractional
            change in elapsed time. An input of N means that a new save file
            will be created if the time changes by a fractional amount N from
            the previous save. If None, this criteria will not be used in
            saving files.

        save_frequency_density: float, default: None
            Frequency in which to save halo files, in terms of the fractional
            change in the central energy density. An input of N means that a
            new save file will be created if the energy density of the
            innermost shell changes by a fractional amount N from the previous
            save. If None, this criteria will not be used in saving files.

        t_epsilon: float, default: 1e-4
            Tolerance level for heat conduction time step.

        r_epsilon: float, default: 1e-14
            Tolerance level for hydrostatic adjustments.
        """
        if (save_frequency_rate is None) and (save_frequency_timing is None) and (save_frequency_density is None):
            print('WARNING: No intermediate halo files will be saved')

        # set tolerance for each evolutionary step
        self.t_epsilon = t_epsilon
        self.r_epsilon = r_epsilon

        # initialization for new evolution calculation
        if self.t == 0:
            # enforce hydrostatic condition initially
            if self.flag_hydrostatic_initial:
                self.hydrostatic_adjustment()
            # save initial state
            self.save_halo()

        # initialize reference values for determining save frequency
        last_save_time = self.t
        last_save_rho  = self.get_central_quantity(self.rho)

        # evolve the halo
        while 1:
            # conduct heat
            try:
                self.conduct_heat()
            except:
                print('!!! Error occurred while conducting heat at time {}'.format(self.t))
                self.save_halo(prefix=self.record.prefix_debug)
                raise
            # make hydrostatic adjustment
            try:
                self.hydrostatic_adjustment()
            except:
                print('!!! Error occurred while making hydrostatic adjustment at time {}'.format(self.t))
                self.save_halo(prefix=self.record.prefix_debug)
                raise

            # determine if halo has met conditions for terminating evolution
            current_rho = self.get_central_quantity(self.rho)
            if self.t >= t_end:
                self.save_halo()
                print('Success: Evolution has reached dimensionless time {}'.format(self.t))
                break
            elif current_rho > rho_factor_end*self.rho_center:
                self.save_halo()
                print('Success: Central density reached its maximum requested value of {} times its initial value'.format(rho_factor_end))
                break
            elif np.any(self.Kn < Kn_end):
                self.save_halo()
                print('Success: Smallest Knudsen number in halo is {}'.format(np.min(self.Kn)))
                break

            # save halo state at intermediate stages of evolution
            save_bool = False
            if save_frequency_rate is not None:
                if self.t/self.t_relax >= self.n_save/save_frequency_rate:
                    save_bool = True
            elif save_frequency_timing is not None:
                delta_t = (self.t - last_save_time)/last_save_time
                if delta_t >= save_frequency_timing:
                    save_bool = True
            elif save_frequency_density is not None:
                delta_rho = np.abs((current_rho-last_save_rho)/last_save_rho)
                if delta_rho >= save_frequency_density:
                    save_bool = True

            if save_bool:
                self.save_halo()
                last_save_time = self.t
                last_save_rho  = current_rho

        return

    def get_dimensionful_quantity(self,quantity,units,value=None):
        """
        Obtain dimensionful radius.

        Parameters
        ----------
        quantity: string
            'r', 'rho', 'm', 'u', 'p', 'v', 't', 'L', 'sigma_m'
        units: Astropy unit
            Units in which quantity will be output.
        value: array-like object, default: None
            Dimensionless quantity to be converted.
            If None, use the relevant quantity of the current halo state.

        Returns
        -------
        Array-like object.
        """
        scale = getattr(self,'scale_'+quantity)
        if value is None:
            value = getattr(self,quantity)
        return (value * scale).to_value(units)
