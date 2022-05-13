    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

if __name__ == '__main__':

    # Obtendo dados dos arquivos
    f_theta = open("theta.txt", "r")
    f_dx = open("dx.txt", "r")
    f_time = open("time.txt", "r")

    theta = eval(f_theta.read())
    dx = eval(f_dx.read())
    time = eval(f_time.read())
    




    # Declarando filtro

    num, den = signal.butter(1, 0.02)
    zi = signal.lfilter_zi(num, den)
    theta_f1, _ = signal.lfilter(num, den, theta, zi=zi*theta[0])


    num2, den2 = signal.butter(2, 0.02)
    zi2 = signal.lfilter_zi(num2, den2)
    theta_f2, _ = signal.lfilter(num2, den2, theta, zi=zi2*theta[0])


    # Traçando gráficos dos dados
    plt.figure('Theta x Tempo')
    plt.plot(time, theta, 'b-', label='Theta')

    #plt.plot(time, z, 'g-', label='Theta filtrado lfilter')
    plt.plot(time, theta_f2, 'y-', label='Theta n=2')
    plt.plot(time, theta_f1, 'r-', label='Theta n=1')


    # plt.figure('Dx x Tempo')
    # plt.plot(time, dx)

    plt.legend(loc='best')
    plt.show()




