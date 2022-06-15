    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

if __name__ == '__main__':

    # Obtendo dados dos arquivos
    f_theta = open("psi.txt", "r")
    f_dx = open("dx.txt", "r")

    f_theta_filt = open("psi_filt.txt", "r")
    f_dx_filt = open("dx_filt.txt", "r")

    f_time = open("time.txt", "r")

    theta = eval(f_theta.read())
    dx = eval(f_dx.read())
    theta_filt = eval(f_theta_filt.read())
    dx_filt = eval(f_dx_filt.read())
    time = eval(f_time.read())
    


    # Declarando filtro

    num, den = signal.butter(1, 0.8)
    zi = signal.lfilter_zi(num, den)
    theta_f1, _ = signal.lfilter(num, den, theta, zi=zi*theta[0])

    theta_f1 = np.array(theta_f1)

    num2, den2 = signal.butter(4, 0.03)
    zi2 = signal.lfilter_zi(num2, den2)
    theta_f2, _ = signal.lfilter(num2, den2, theta, zi=zi2*theta[0])

    dx_f1, _ = signal.lfilter(num, den, dx, zi=zi*dx[0])

    dx_f1 = np.array(dx_f1)

    # Traçando gráficos dos dados
    plt.figure('Theta x Tempo')
    plt.plot(time, theta, 'b-', label='Theta')
    plt.plot(time, theta_filt, 'g--', label='Theta filt')

    #plt.plot(time, z, 'g-', label='Theta filtrado lfilter')
    #plt.plot(time, theta_f2, 'y-', label='Theta n=2')
    plt.plot(time, theta_f1, 'r-', label='Theta n=1')

    plt.figure('Dx x Tempo')
    plt.plot(time, dx, 'b-', label='Dx')
    plt.plot(time, dx_filt, 'g--', label='Dx filt')
    plt.plot(time, dx_f1, 'r-', label='Dx n=1')

    plt.legend(loc='best')
    plt.show()




