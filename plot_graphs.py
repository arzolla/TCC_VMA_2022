    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    # Obtendo dados dos arquivos


    f_map_x = open("ideal_map_x.txt")
    f_map_y = open("ideal_map_y.txt")

    map_x = eval(f_map_x.read())
    map_y = eval(f_map_y.read())


    # Trajeto
    plt.figure('Trajeto')
    plt.plot(map_x, map_y, 'b-', label='Ideal')

    plt.legend(loc='best')
    



    plt.legend(loc='best')
    plt.show()




