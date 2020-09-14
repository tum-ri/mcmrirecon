import pathlib
import random
import numpy as np
import h5py
import itertools as it

from torch.utils.data import Dataset


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, data_set, dim, domain, transform=None,
                 challenge='multicoil', sample_rate=1, slice_cut=(50, 50), num_edge_slices=0, edge_model=False):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.data_set = data_set
        self.dim = dim
        try:
            self.transform = getattr(transform, domain)
        except:
            self.transform = None 
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = list()
        files = list(pathlib.Path(root).iterdir())

        if not files:  # If the list is empty for any reason
            raise FileNotFoundError('Empty directory')

        print(f'Initializing {root}.')

        if sample_rate < 1:
            random.shuffle(files)
            num_files = np.ceil(len(files) * sample_rate)
            files = files[:num_files]

        for file_name in sorted(files):
            try:
                kspace = h5py.File(file_name, mode='r')['kspace']
            except:
                continue

            num_slices = kspace.shape[0]
            if num_edge_slices == 0:
                self.examples += [(file_name, slice_num) for slice_num in range(slice_cut[0], num_slices - slice_cut[1])]
            elif num_edge_slices > 0 and edge_model:
                self.examples += [(file_name, slice_num) for slice_num in
                                  it.chain(range(slice_cut[0], slice_cut[0]+num_edge_slices),
                                           range(num_slices-slice_cut[1]-num_edge_slices, num_slices-slice_cut[1]))]
            elif num_edge_slices > 0 and not edge_model:
                self.examples += [(file_name, slice_num) for slice_num in range(slice_cut[0]+num_edge_slices, num_slices-slice_cut[1]-num_edge_slices)]
            else:
                raise ValueError('num_edge_slices is negative')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        file_path, slice_num = self.examples[idx]
        with h5py.File(file_path, mode='r') as data:
            attrs = dict(data.attrs)
            k_slice = data['kspace'][slice_num]
            # Most volumes have 170 slices, but some have more. For these cases we crop back to 170 during training.
            # Could be made more generic.
            if k_slice.shape[1] != self.dim[1] and self.data_set == 'train':
                idx = int((k_slice.shape[1] - self.dim[1]) / 2)
                k_slice = k_slice[:, idx:-idx, :]

            # return data for test data generator
            if self.transform is None:
                #print("Returning Test Data")
                return k_slice, file_path.name, slice_num
            
            # Explicit zero-filling after (sr) 85% in the slice-encoded direction
            z = k_slice.shape[1]
            z_sampled = int(np.ceil(z*0.85))
            k_slice[:, z_sampled:, :] = 0
            return self.transform(k_slice, attrs, file_path.name, slice_num)
