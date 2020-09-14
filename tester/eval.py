import numpy as np
import os
import glob
import h5py
import pickle
from utils.metrics import metrics

results_path =".\data"
team_name = "QuadImageR5BestVal=True"
recs = [os.path.join(results_path,"Track01","12-channel-R=5","QuadImageR5BestVal=True","*.h5")]

refs = [os.path.join(results_path,"Track01","12-channel-R=5","ref")]

nfiles = 20
nslices = 156
res_dic = {}

dic_keys = ["track01_12_channel_R=5"]

for i in range(len(recs)):
    print(i)
    rec_path = glob.glob(recs[i])
    rec_path.sort()
    print(rec_path[0])
    if len(rec_path) != 20:
        print(recs[i])
        print("Number of files incompaible with number of files in the test set!")
        continue

    m = np.zeros((nfiles * nslices, 3))
    for (jj, ii) in enumerate(rec_path):
        print(ii)
        with h5py.File(ii, 'r') as f:
            rec = f['reconstruction'][()]

        name = ii.split("\\")[-1]
        with h5py.File(os.path.join(refs[i], name), 'r') as f:
            ref = f['reconstruction'][()]

        ref_max = ref.max(axis=(1, 2), keepdims=True)
        ref = ref / ref_max

        rec = rec / ref_max

        ssim, psnr, vif = metrics(rec, ref)
        m[jj * nslices:(jj + 1) * nslices, 0] = ssim
        m[jj * nslices:(jj + 1) * nslices, 1] = psnr
        m[jj * nslices:(jj + 1) * nslices, 2] = vif

    res_dic[dic_keys[i]] = m

with open(os.path.join(results_path, 'Metrics', team_name + '.pickle'), 'wb') as handle:
    pickle.dump(res_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

