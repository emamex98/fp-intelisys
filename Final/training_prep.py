import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import data_construction as dc
import classifiers as classifiers

def training_preparation(filename):
    # Read data file
    np.seterr(over='raise')
    data = np.loadtxt(filename) 
    samp_rate = 256
    samps = data.shape[0]
    n_channels = data.shape[1]

    time = data[:, 0]

    # Data channels
    chann1 = data[:, 1]
    chann2 = data[:, 3]

    # Mark data
    mark = data[:, 6]

    training_samples = {}
    for i in range(0, samps):       
        if mark[i] > 0: 
            #print("Marca", mark[i], 'Muestra', i, 'Tiempo', time[i]) 

            if  (mark[i] > 100) and (mark[i] < 200):
                iniSamp = i
                condition_id = mark[i]
            elif mark[i] == 200:
                if not str(condition_id) in training_samples.keys():
                    training_samples[str(condition_id)] = []
                training_samples[str(condition_id)].append([iniSamp, i])

    #print('Rango de muestras con datos de entrenamiento:', training_samples)

    emg_data = {}
    psd_data_ch1 = {}
    psd_data_ch2 = {}
    window_size = 256
    for posture in training_samples:
        #emg_data[posture] = []
        print(posture)
        #input()
        
        for interval in training_samples[posture]:
            start_samp = interval[0]
            end_samp = interval[1]

            n_win = (end_samp - start_samp)//window_size
            for i in range(n_win):
                next_samp = start_samp + window_size
                emg_data = time[start_samp:next_samp]

                x = chann1[start_samp:next_samp]
                x2 = chann2[start_samp:next_samp]
                
                power, freq = psd(x, NFFT = window_size, Fs = samp_rate)
                start_freq = next(j for j, val in enumerate(freq) if val >= 4.0)
                end_freq = next(j for j, val in enumerate(freq) if val >= 60.0)
                start_index = np.where(freq >= 4.0)[0][0]
                end_index = np.where(freq >= 60.0)[0][0]
                if not posture in psd_data_ch1:
                    psd_data_ch1[posture] = np.empty((0,end_index-start_index+1))
                psd_data_ch1[posture] = np.append(psd_data_ch1[posture],[power[start_index:end_index+1]], axis=0)

                power2, freq2 = psd(x2, NFFT = window_size, Fs = samp_rate)
                start_freq = next(j for j, val in enumerate(freq2) if val >= 4.0)
                end_freq = next(j for j, val in enumerate(freq2) if val >= 60.0)
                start_index = np.where(freq2 >= 4.0)[0][0]
                end_index = np.where(freq2 >= 60.0)[0][0]
                if not posture in psd_data_ch2:
                    psd_data_ch2[posture] = np.empty((0,end_index-start_index+1))

                psd_data_ch2[posture] = np.append(psd_data_ch2[posture],[power2[start_index:end_index+1]], axis=0)           
                
                start_samp += window_size

    freq = [x for x in range(4,61)]

    for posture in psd_data_ch1:
        psd_data_ch1[posture] = psd_data_ch1[posture].tolist()
        psd_data_ch2[posture] = psd_data_ch2[posture].tolist()

    x, y = dc.create_matrix(psd_data_ch1, psd_data_ch2)

    return x, y