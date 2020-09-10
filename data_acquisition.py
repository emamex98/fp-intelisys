#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# while True:
    # try:
    #     data, addr = sock.recvfrom(1024*1024)                        
            
    #     values = np.frombuffer(data)       
    #     ns = int(len(values)/n_channels)
    #     samp_count+=ns        

    #     for i in range(ns):
    #         for j in range(n_channels):
    #             emg_data[j].append(values[n_channels*i + j])
            
    #     elapsed_time = time.time() - start_time
    #     if (elapsed_time > 0.1):
    #         start_time = time.time()
    #         print ("Muestras: ", ns)
    #         print ("Cuenta: ", samp_count)
    #         print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
    #         print("")
            
    # except socket.timeout:
    #     pass  


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# Initialize communication with sensor

# This function is called periodically from FuncAnimation
def animate(i, xs, ys, samp_count=0, start_time=start_time):

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


    power = [row[samp_count-1] for row in emg_data][0]

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(power)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('REAL TIME INOUT')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
plt.show()
    
    

