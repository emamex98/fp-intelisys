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

# Data configuration
n_channels = 6
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

# Global variables
psd_chann1 = []
psd_chann2 = []
freq_chann1 = []
freq_chann2 = []

# Set plot figures
fig = plt.figure()

ax_emg_chann1 = fig.add_subplot(2,2,1)
ax_emg_chann1.title.set_text('Loading Data...')
ax_emg_chann1.set_xlabel('Time')
ax_emg_chann1.set_ylabel('micro V')

ax_emg_chann2 = fig.add_subplot(2,2,2)
ax_emg_chann2.title.set_text('Loading Data...')
ax_emg_chann2.set_xlabel('Time')
ax_emg_chann2.set_ylabel('micro V')

ax_psd_chann1 = fig.add_subplot(2,2,3)
ax_psd_chann1.title.set_text('Loading Data...')
ax_psd_chann1.set_xlabel('Hz')
ax_psd_chann1.set_ylabel('Power')

ax_psd_chann2 = fig.add_subplot(2,2,4)
ax_psd_chann2.title.set_text('Loading Data...')
ax_psd_chann2.set_xlabel('Hz')
ax_psd_chann2.set_ylabel('Power')

# Data acquisition
start_time = time.time()

# Calculate PSD
def calcPSD(channel, win_size, samp_rate):

    power, freq = psd(channel, NFFT=win_size, Fs=samp_rate)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    return power[start_index:end_index], freq[start_index:end_index]

# Realtime plot function
def animate(i, start_time=start_time):

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

        elapsed_time = time.time() - start_time

        if (elapsed_time > 2 and samp_count >= win_size):

            start_time = time.time()
            
            chann1 = emg_data[0][-win_size:]
            chann2 = emg_data[2][-win_size:]


            #### TODO: Preguntar al profe por qué lecturas son muy altas a veces

            for item in chann1:
                if item > 200:
                    chann1.remove(item)

            for item in chann2:
                if item > 200:
                    chann2.remove(item)

            ####

            psd_chann1, freq_chann1 = calcPSD(chann1, win_size, samp_rate)
            psd_chann2, freq_chann2 = calcPSD(chann2, win_size, samp_rate)

            time_axis = []
            inc = 2 / len(chann1)
            for i in range(len(chann1)):
                time_axis.append(elapsed_time)
                elapsed_time += inc

            # Plot EMG - Channel 1
            ax_emg_chann1.clear()
            ax_emg_chann1.plot(time_axis, chann1)
            ax_emg_chann1.title.set_text('EMG - Channel 1')
            ax_emg_chann1.set_xlabel('Time')
            ax_emg_chann1.set_ylabel('micro V')

            # Plot EMG - Channel 2
            ax_emg_chann2.clear()
            ax_emg_chann2.plot(time_axis, chann2, color='red')
            ax_emg_chann2.title.set_text('EMG - Channel 2')
            ax_emg_chann2.set_xlabel('Time')
            ax_emg_chann2.set_ylabel('micro V')

            # Plot PSD - Channel 1
            ax_psd_chann1.clear()
            ax_psd_chann1.plot(freq_chann1, psd_chann1)
            ax_psd_chann1.title.set_text('PSD - Channel 1')
            ax_psd_chann1.set_xlabel('Hz')
            ax_psd_chann1.set_ylabel('Power')

            # Plot PSD - Channel 2
            ax_psd_chann2.clear()
            ax_psd_chann2.plot(freq_chann2, psd_chann2, color='red')
            ax_psd_chann2.title.set_text('PSD - Channel 2')
            ax_psd_chann2.set_xlabel('Hz')
            ax_psd_chann2.set_ylabel('Power')
            
            
    except socket.timeout:
        pass  

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, interval=2000)
plt.tight_layout()
plt.show()
