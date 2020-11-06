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
from classifiers import sl

# Training
x,y = tp("Izquierda, derecha, cerrado.txt")

clf = sl(x,y)

# Data configuration
n_channels = 5
samp_rate = 256
win_size = 256
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
def calcPSD(channel, win_size, samp_rate):

    power, freq = psd(channel, NFFT=win_size, Fs=samp_rate)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    return power[start_index:end_index], freq[start_index:end_index]

def calculate():

    global samp_count

    # Read data from socket
    try:
        data, addr = sock.recvfrom(1024*1024)                        
            
        values = np.frombuffer(data)       
        ns = int(len(values)/n_channels)
        samp_count+=ns

        for i in range(ns):            
            for j in range(n_channels):                
                emg_data[j].append(values[n_channels*i + j])

        #elapsed_time = time.time() - start_time

        #if (elapsed_time > 2 and samp_count >= win_size):

         #   start_time = time.time()
            
        chann1 = emg_data[0][-win_size:]
        chann2 = emg_data[2][-win_size:]

        psd_chann1, freq_chann1 = calcPSD(chann1, win_size, samp_rate)
        psd_chann2, freq_chann2 = calcPSD(chann2, win_size, samp_rate)

        """[time_axis = []
        inc = 2 / len(chann1)
        for i in range(len(chann1)):
            time_axis.append(elapsed_time)
            elapsed_time += inc]"""

        x = [[]]
        for i in psd_chann1:
            x[0].append(i)

        for i in psd_chann2:
            x[0].append(i)
        y_pred = clf.predict(x)
        print(y_pred)
        input()
    except socket.timeout:
        pass
    # Add x and y to lists
    #xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    #ys.append(power)
    #yp.append(power2)

while True:
    calculate()
