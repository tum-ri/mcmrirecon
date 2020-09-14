import glob
import pickle

results = glob.glob(
    "C:/Users/Fabi/PycharmProjects/git/mri-reconstruction/tester/data/Metrics/*.pickle")
crop = 30, 30
nslices = 156
nfiles = 20

print("Track 01 - 12-channel")
print("R = 5")
t1_r5 = 'track01_12_channel_R=5'
for ii in results:
    name = ii.split("/")[-1].split(".pickle")[0]

    res_file = open(ii, 'rb')
    res = pickle.load(res_file)
    res_file.close()

    if t1_r5 in res:
        res = res[t1_r5]
        ssim = res[:, 0].reshape(nslices, nfiles)[crop[0]:-crop[1], :]
        psnr = res[:, 1].reshape(nslices, nfiles)[crop[0]:-crop[1], :]
        vif = res[:, 2].reshape(nslices, nfiles)[crop[0]:-crop[1], :]
        print("%s,  %.4f +/- %.4f, %.4f +/- %.4f, %.4f +/- %.4f" % (
        name, ssim.mean(), ssim.std(), psnr.mean(), psnr.std(), \
        vif.mean(), vif.std()))
