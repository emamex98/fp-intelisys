#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.mlab import psd

from training_prep import training_preparation as tp
from classifiers import svd_lineal as sl

# Training
x,y = tp("Izquierda, derecha, cerrado.txt")

clf = sl(x,y)

# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
samp_count = 0

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time.time()

# Create figure for plotting
xs = []
ys = []
powers = []

xp = []
yp = []
powers2 = []

# PSD CALC
def psdCalc(_channel, _win_size, _start_samp, _samp_rate, accumPSDPower, _posturas, _mark):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp: end_samp]

    power, freq = psd(x, NFFT=_win_size, Fs=_samp_rate)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    if _mark is not None:
        _posturas.append(int(_mark))

    return power[start_index:end_index]

def calculate(samp_count=0, start_time=start_time):

    # Read data from socket
    try:
        data, addr = sock.recvfrom(1024*1024)                        
            
        values = np.frombuffer(data)       
        ns = int(len(values)/n_channels)
        samp_count+=ns        

        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])
            
        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            print ("Muestras: ", ns)
            print ("Cuenta: ", samp_count)
            print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
            print("")
            
    except socket.timeout:
        pass  


    # power = [row[samp_count-1] for row in emg_data][channel-1]

    # Take average of window
    power = [np.average(row) for row in emg_data][0] 
    power2 = [np.average(row) for row in emg_data][1]

    powers.append(power)
    powers2.append(power2)

    psd_pow, freq = psd(powers, NFFT = 256, Fs = 256)
    psd_pow_avg = []

    psd_pow2, freq2 = psd(powers2, NFFT = 256, Fs = 256)
    psd_pow_avg2 = []

    # power2, freq2 = psd(x2, NFFT = window_size, Fs = samp_rate)
    start_freq = next(j for j, val in enumerate(freq) if val >= 4.0)
    end_freq = next(j for j, val in enumerate(freq) if val >= 60.0)
    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    start_freq2 = next(j for j, val in enumerate(freq2) if val >= 4.0)
    end_freq2 = next(j for j, val in enumerate(freq2) if val >= 60.0)
    start_index2 = np.where(freq2 >= 4.0)[0][0]
    end_index2 = np.where(freq2 >= 60.0)[0][0]

    psd_pow_avg.append(psd_pow[start_index:end_index+1])

    print(psd_pow_avg)
    
    psd1 = psdCalc()
    x = [[]]
    for i in psd_pow[start_index:end_index+1]:
        x[0].append(i)

    for i in psd_pow2[start_index2:end_index2+1]:
        x[0].append(i)
    print(x)
    input()
    y_pred = clf.predict(x)
    print(y_pred)

    # Add x and y to lists
    #xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    #ys.append(power)
    #yp.append(power2)

while True:
    calculate()
    input()
