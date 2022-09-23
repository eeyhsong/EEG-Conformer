# plot EEG topograpy with mne 
# https://mne.tools/stable/index.html

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import mlab as mlab
# from preprocess import import_data

data = 0
label = 0
# get the data and label 
# data - (samples, channels, trials)
# label -  (label, 1)

data = np.transpose(data, (2, 1, 0))
label = np.squeeze(np.transpose(label))
idx = np.where(label == 1)
data_draw = data[idx]

mean_trial = np.mean(data_draw, axis=0)  # mean trial
# use standardization or normalization to adjust
mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)


mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left


# Draw topography
biosemi_montage = mne.channels.make_standard_montage('biosemi64')  # set a montage, see mne document
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]  # correspond channel
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')  # sample rate

evoked1 = mne.EvokedArray(mean_trial, info)
evoked1.set_montage(biosemi_montage)
plt.figure(1)
# im, cn = mne.viz.plot_topomap(np.mean(mean_trial, axis=1), evoked1.info, show=False)
im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False)
plt.colorbar(im)

plt.savefig('./topo/test.png')
print('the end')

