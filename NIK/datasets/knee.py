from torch.utils.data import Dataset
import torch
import numpy as np



class KneeDataset(Dataset):
    def __init__(
            self,
            # The `path_ktraj` and `path_kdata` parameters in the `KneeDataset` class constructor are
            # expected to be file paths to NumPy arrays containing k-space trajectory data and k-space
            # data respectively.
            path_ktraj: np.ndarray, # [xyz, nSpokePoints, nSpokes]
            path_kdata: np.ndarray, # [nSpokes, nSpokePoints, nCoils]
            coils: list = None,
        ):
        super().__init__()
        
        
        ktraj = np.load(path_ktraj) # 
        ktraj = np.transpose(ktraj, (0, 2, 1)) # [xyz, nSpokes, nSpokePoints]
        #normalize each xyz coordinate to [-1, 1]
        #independently for each xyz axis
        for i in range(3):
            ktraj[i,:] = (ktraj[i,:] - np.min(ktraj[i,:])) / (np.max(ktraj[i,:]) - np.min(ktraj[i,:])) * 2 - 1
        kdata = np.load(path_kdata)
        kdata = np.transpose(kdata, (2, 1, 0)) # [nCoils, nSpokes, nSpokePoints]
        if coils is not None:
            kdata = kdata[coils, :, :]
        
        assert ktraj.shape[1] == kdata.shape[1]
        assert ktraj.shape[2] == kdata.shape[2]
        
        # kspace data starts as [nCoils, nSpokes, nSpokePoints]
        # need to flatten this to [nCoils * nSpokes * nSpokePoints, 1]
        nc, n_spokes, n_spoke_points = kdata.shape
        kdata_flat = kdata.astype(np.complex64).reshape(-1, 1)
        self.n_kspace_points = kdata_flat.shape[0]

        assert self.n_kspace_points == nc * n_spokes * n_spoke_points

        
        # now, need to do perpare the kspace trajectory / coorinates data
        # there are 5 inputs (t, c, kx, ky, kz) and for each combination
        # of these, we should get back a single k-space value/datapoint (complex)
        
        kcoords = np.zeros((nc, n_spokes, n_spoke_points, 5))
        time_ = np.linspace(-1, 1, n_spokes) # create normalized time values to span [-1, 1]
        kcoords[:, :, :, 0] = np.reshape(time_, [1, n_spokes, 1])
        k_coil = np.linspace(-1, 1, nc) # create normalized coil values to span [-1, 1]
        kcoords[:, :, :, 1] = np.reshape(k_coil, [nc, 1, 1])
        kcoords[:, :, :, 2] = ktraj[0, :][None]
        kcoords[:, :, :, 3] = ktraj[1, :][None]
        kcoords[:, :, :, 4] = ktraj[2, :][None]

        kcoords_flat = np.reshape(kcoords.astype(np.float32), (-1, 5))

        assert kcoords_flat.shape[0] == kdata_flat.shape[0]
        
        self.kcoords_flat = torch.from_numpy(kcoords_flat)
        self.kdata_flat = torch.from_numpy(kdata_flat)

    def __len__(self):
        return self.n_kspace_points

    def __getitem__(self, index):
        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'targets': self.kdata_flat[index]
        }
        return sample
