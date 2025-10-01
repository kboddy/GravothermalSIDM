import os
import string
import glob
import pickle
import h5py
import numpy as np

class HaloRecord:
    """
    Class to handle the file I/O for running the gravothermal evolution.
    """
    def __init__(self,dir_data):
        """
        Initialize record keeping for halo evolution.

        Parameters
        ----------
        dir_data: string
            Directory to store information for halo evolution.
            Either an absolute or a relative path may be provided.
        """
        ##### data directories and files
        self.dir_data = os.path.abspath(dir_data.strip())
        self.basename = os.path.basename(self.dir_data)
        self.path_ini = os.path.join(self.dir_data,'halo_ini.h5')
        self.path_archive = os.path.join(self.dir_data,self.basename+'.h5')

        ##### flags
        self.has_record = os.path.isfile(self.path_ini)

        ##### file name prefixes for halo state pickle files
        # use only ASCII letters and 'time' is reservered for default prefix
        self.prefix_default = 'time'
        self.prefix_debug   = 'broken'

        return

    def get_halo_initialization(self):
        """
        Obtain saved halo initialization information and original halo state.
        Information about original halo state is limited to profile quantities,
        so array-like objects are expected.

        Returns
        -------
        Tuple containing dictionary of saved halo information
        and dictionary of original halo state.
        """
        if not self.has_record:
            raise IOError('No halo initialization file exists.')

        with h5py.File(self.path_ini,'r') as hf:
            data1 = {key: hf.attrs[key] for key in hf.attrs.keys()}
            if len(hf.keys()) > 0:
                data2 = {key: hf[key][:] for key in hf.keys()}
            else:
                data2 = {} # for backward compatibility
        return data1, data2

    def save_halo_initialization(self,halo_ini,original_state):
        """
        Save halo initialization information and original halo state to file.
        Information about original halo state is limited to profile quantities,
        so array-like objects are expected in `get_halo_initialization()`.

        Parameters
        ----------
        halo_ini: dictionary
            Halo initialization dictionary associated with Halo class object.

        original_state: dictionary
            Original halo state information.
        """
        # create data directory
        os.makedirs(self.dir_data,exist_ok=True)

        # save halo initialization file
        with h5py.File(self.path_ini,'w') as hf:
            # save initialization dictionary
            hf.attrs.update(halo_ini)
            # save original halo profile parameters
            for key in original_state.keys():
                hf.create_dataset(key,data=original_state[key])

        # update flag
        self.has_record = True

        return

    def extract_filename_time(self,file_halo):
        """
        Obtain time associated with halo state pickle.

        Parameters
        ----------
        file_halo: string
            Name of pickle file.

        Returns
        -------
        Float for time associated with `file_halo` name.
        """
        time_str = file_halo.lstrip(string.ascii_letters).removesuffix('.pickle').replace('d','.')
        return float(time_str)

    def glob_pickle_files(self):
        """
        Obtain all halo state pickle files with default prefix.

        Parameters
        ----------
        prefix: string
            Prefix of pickle files to glob.

        Returns
        -------
        List of data files and list of associated times (both sorted by time).
        """
        file_name  = self.prefix_default+'*.pickle'
        glob_files = glob.glob(os.path.join(self.dir_data,file_name))
        list_files = [os.path.basename(f) for f in glob_files]
        list_times = [self.extract_filename_time(f) for f in list_files]

        # check if there are no pickle files
        if len(list_files)==0:
            return [],[]

        # sort lists by increasing time
        sorted_times, sorted_files = zip(*sorted(zip(list_times,list_files)))
        return sorted_files, np.array(sorted_times)

    def get_halo_state_pickled(self,file_halo=None,time=None):
        """
        Obtain saved halo information from pickle file.

        Parameters
        ----------
        file_halo: string
            Name of pickle file produced by `save_halo`.
            Provide the file name only, not the full path location.
            If not None, ignore all other inputs.

        time: float, default: None
            If None, recover most recently saved halo state with default prefix.
            If not None, find halo state pickle file that corresponds
            most closely to input time.

        Returns
        -------
        Dictionary of halo information (empty if data is not found).
        """
        if file_halo is None:
            list_files, list_times = self.glob_pickle_files()
            if len(list_files)==0:
                return {}
            if time is None:
                index = -1
            else:
                index = np.argmin(np.abs(time-list_times))
            path_halo = os.path.join(self.dir_data,list_files[index])
        else:
            path_halo = os.path.join(self.dir_data,file_halo)

        # determine availability of data
        if not os.path.isfile(path_halo):
            return {}

        with open(path_halo,'rb') as fopen:
            data = pickle.load(fopen)
        return data

    def save_halo_state_pickled(self,prefix,time,halo_state):
        """
        Save halo information to pickle file. File names have the format
        '(prefix)(time).pickle', without the parentheses.

        Exisiting files of the same name are overwritten.

        Parameters
        ----------
        prefix: string
            File name prefix for the save state of the halo.

        time: float
            Time associated with the halo state.

        halo_state: dictionary
            Dictionary of halo quantities to be saved.
        """
        # set the name of the output file
        time_str = '{}'.format(time).replace('.','d')
        fname = os.path.join(self.dir_data,prefix+time_str+'.pickle')

        # save data
        with open(fname,'wb') as fopen:
            pickle.dump(halo_state,fopen)
        return

    def get_halo_state_archived(self,time=None):
        """
        Load halo information from archive.

        Parameters
        ----------
        time: float, default: None
            If None, recover most recently saved halo state.
            If not None, obtain archived halo state that corresponds
            most closely to input time.

        Returns
        -------
        Dictionary of halo information (empty if data is not found).
        """
        # determine availability of data
        if not os.path.isfile(self.path_archive):
            return {}

        index = -1 # initialize index to correspond to latest time
        with h5py.File(self.path_archive,'r') as hf:
            if time is not None:
                index = np.argmin(np.abs(time-hf['t'][:]))
            data = {}
            for key in hf.keys():
                data_array = hf[key][:]
                if data_array.ndim==1:
                    data[key] = data_array[index]
                else:
                    data[key] = data_array[index,:]
        return data

    def create_archive(self):
        """
        Create HDF5 archive file from halo state pickle files.
        Any existing archive file is overwritten, and only the existing
        pickle files are used to create the new archive.
        """
        # obtain all halo state pickle files
        list_files, list_times = self.glob_pickle_files()

        # extract names of saved variables
        data = self.get_halo_state_pickled(file_halo=list_files[-1],time=None)
        save_names = data.keys()

        # separate names based on np.ndarray
        array_names = [n for n in save_names if     isinstance(data[n],np.ndarray)]
        other_names = [n for n in save_names if not isinstance(data[n],np.ndarray)]

        # specify types
        array_types = [(n,data[n].dtype) for n in array_names]
        other_types = [(n,np.dtype(type(data[n]))) for n in other_names]

        # initialize and fill tables and types
        n_shells = len(data[array_names[0]])
        array_table = np.empty((len(list_files),n_shells),dtype=array_types)
        other_table = np.empty(len(list_files),dtype=other_types)
        for idx,f in enumerate(list_files):
            data = self.get_halo_state_pickled(file_halo=f,time=None)
            for n in array_names:
                array_table[n][idx,:] = data[n]
            for n in other_names:
                other_table[n][idx] = data[n]

        # organize data structures and save into archive
        with h5py.File(self.path_archive,'w') as hf:
            for n in array_names:
                hf.create_dataset(n,data=array_table[n])
            for n in other_names:
                hf.create_dataset(n,data=other_table[n])
        return
