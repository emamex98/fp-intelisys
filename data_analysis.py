#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd


# Read data file
np.seterr(over='raise')
data = np.loadtxt("Izquierda, derecha, cerrado.txt") 
samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]

print('Número de muestras: ', data.shape[0])
print('Número de canales: ', data.shape[1])
print('Duración del registro: ', samps / samp_rate, 'segundos')
#print(data)

# Time channel
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
    psd_data_ch1[posture] = np.empty((1,57))
    psd_data_ch2[posture] = np.empty((1,57))
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
            psd_data_ch1[posture] = np.append(psd_data_ch1[posture],[power[start_index:end_index+1]], axis=0)

            power2, freq2 = psd(x2, NFFT = window_size, Fs = samp_rate)
            start_freq = next(j for j, val in enumerate(freq2) if val >= 4.0)
            end_freq = next(j for j, val in enumerate(freq2) if val >= 60.0)
            start_index = np.where(freq2 >= 4.0)[0][0]
            end_index = np.where(freq2 >= 60.0)[0][0]

            psd_data_ch2[posture] = np.append(psd_data_ch2[posture],[power2[start_index:end_index+1]], axis=0)
           
            start_samp += window_size
print(psd_data_ch1['101.0'][0])
avg_psd_ch1 = {}
avg_psd_ch2 = {}
for posture in training_samples:
    avg_psd_ch1[posture] = []
    avg_psd_ch2[posture] = []
    sum1 = psd_data_ch1[posture].sum(axis=0)
    sum2 = psd_data_ch2[posture].sum(axis=0)
    div1 = np.true_divide(sum1,57)
    div2 = np.true_divide(sum2,57)
    avg_psd_ch1[posture].append(list(div1))
    avg_psd_ch2[posture].append(list(div2))
#print(avg_psd_ch2)
freq = [x for x in range(4,61)]

def plotWindow(posture, window, figure):

    start_samp = training_samples[posture][window][0]
    end_samp = training_samples[posture][window][1]
    plt.figure(figure)

    plt.subplot(2,2,1)
    plt.plot(time[start_samp:end_samp], chann1[start_samp:end_samp], label = 'Canal 1')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('micro V')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(time[start_samp:end_samp], chann2[start_samp:end_samp], color = 'red', label = 'Canal 2')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('micro V')
    plt.legend()

    # Power Spectral Density (PSD) (1 second of training data)
    win_size = 256
    ini_samp = training_samples[posture][window][0]
    end_samp = ini_samp + win_size
    x = chann1[ini_samp : end_samp]
    x2 = chann2[ini_samp : end_samp] 
    t = time[ini_samp : end_samp]

    power, freq = psd(x, NFFT = win_size, Fs = samp_rate)

    start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
    end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
    #print(start_freq, end_freq)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    plt.subplot(2,2,3)
    plt.plot(freq[start_index:end_index], power[start_index:end_index], label = 'Canal 1')
    plt.xlabel('Hz')
    plt.ylabel('Power')
    plt.legend()

    power2, freq2 = psd(x2, NFFT = win_size, Fs = samp_rate)

    start_freq = next(x for x, val in enumerate(freq2) if val >= 4.0)
    end_freq = next(x for x, val in enumerate(freq2) if val >= 60.0)

    start_index = np.where(freq2 >= 4.0)[0][0]
    end_index = np.where(freq2 >= 60.0)[0][0]

    plt.subplot(2,2,4)
    plt.plot(freq2[start_index:end_index], power2[start_index:end_index], color = 'red', label = 'Canal 2')
    plt.xlabel('Hz')
    plt.ylabel('Power')
    plt.legend()

    print("********************************")

    print("Average PSD of ", posture, " - CH1")
    print(np.average(power))

    print("Average PSD of ", posture, " - CH2")
    print(np.average(power2))

    return [np.average(power), np.average(power2)]


# Plot data
#window = 3

#avg101 = plotWindow(posture='101.0', window=window, figure=101)
#avg102 = plotWindow(posture='102.0', window=window, figure=102)
#avg103 = plotWindow(posture='103.0', window=window, figure=103)

#print(avg101)
plt.figure(101)
#plt.subplot(1,2,1)
plt.plot(freq, avg_psd_ch1['101.0'][0], label = 'Canal 1', color="blue")
#plt.plot(freq, avg_psd_ch2['101.0'][0] , label = 'Canal 2', color="red")
plt.xticks(np.arange(4, 60))
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()

plt.figure(1012)
#plt.subplot(1,2,2)
plt.plot(freq, avg_psd_ch2['101.0'][0] , label = 'Canal 2', color="red")
plt.xticks(np.arange(4, 60))
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()

plt.show()
