    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    # Obtendo dados dos arquivos


    map_x = eval(open("path_ideal_x.txt").read())
    map_y = eval(open("path_ideal_y.txt").read())

    vel15_x = eval(open("path_vel_18_x.txt").read())
    vel15_y = eval(open("path_vel_18_y.txt").read())




    # Trajeto
    plt.figure('Trajeto', frameon=False)
    plt.plot(map_x, map_y, 'b-', label='Simulador')
    #plt.plot(vel15_x, vel15_y, 'r--', label='Controle, v = 18')
    
    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    #plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    plt.grid()
    plt.axis('square')
    #plt.legend(loc='best')
    import matplotlib.ticker as mticker
    #plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))

    plt.savefig('line_plot.pdf', bbox_inches='tight')  
    plt.show()




