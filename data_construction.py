import numpy as np

def create_matrix(psd_ch1, psd_ch2):
    matrix_x = np.empty([270, 114])
    matrix_y = np.empty([270, 1])

    i = 0
    j = 0

    for posture in psd_ch1:
        for array in range(len(psd_ch1[posture])):
            for n in range(len(psd_ch1[posture][array])):
                matrix_x[i][j] = psd_ch1[posture][array][n]
                matrix_x[i][57+j] = psd_ch2[posture][array][n]
                j+=1
            matrix_y[i][0] = posture
            j=0
            i+=1
    return matrix_x, matrix_y