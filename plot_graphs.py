    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np


def align_yaxis_np(axes):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array(axes)
    extrema = np.array([ax.get_ylim() for ax in axes])

    # reset for divide by zero issues
    for i in range(len(extrema)):
        if np.isclose(extrema[i, 0], 0.0):
            extrema[i, 0] = -1
        if np.isclose(extrema[i, 1], 0.0):
            extrema[i, 1] = 1

    # upper and lower limits
    lowers = extrema[:, 0]
    uppers = extrema[:, 1]

    # if all pos or all neg, don't scale
    all_positive = False
    all_negative = False
    if lowers.min() > 0.0:
        all_positive = True

    if uppers.max() < 0.0:
        all_negative = True

    if all_negative or all_positive:
        # don't scale
        return

    # pick "most centered" axis
    res = abs(uppers+lowers)
    min_index = np.argmin(res)

    # scale positive or negative part
    multiplier1 = abs(uppers[min_index]/lowers[min_index])
    multiplier2 = abs(lowers[min_index]/uppers[min_index])

    for i in range(len(extrema)):
        # scale positive or negative part based on which induces valid
        if i != min_index:
            lower_change = extrema[i, 1] * -1*multiplier2
            upper_change = extrema[i, 0] * -1*multiplier1
            if upper_change < extrema[i, 1]:
                extrema[i, 0] = lower_change
            else:
                extrema[i, 1] = upper_change

        # bump by 10% for a margin
        extrema[i, 0] *= 1.1
        extrema[i, 1] *= 1.1

    # set axes limits
    [axes[i].set_ylim(*extrema[i]) for i in range(len(extrema))]

if __name__ == '__main__':


    # Obtendo dados dos arquivos

    map_x = eval(open("logs\\path_ideal_x.txt").read())
    map_y = eval(open("logs\\path_ideal_y.txt").read())

    vel25_time = eval(open("logs\\vel_25_time.txt").read())
    vel25_psi = eval(open("logs\\vel_25_psi.txt").read())
    vel25_dx = eval(open("logs\\vel_25_dx.txt").read())
    vel25_steer = eval(open("logs\\vel_25_steer.txt").read())

    vel20_time = eval(open("logs\\vel_20_time.txt").read())
    vel20_psi = eval(open("logs\\vel_20_psi.txt").read())
    vel20_dx = eval(open("logs\\vel_20_dx.txt").read())
    vel20_steer = eval(open("logs\\vel_20_steer.txt").read())

    vel15_time = eval(open("logs\\vel_15_time.txt").read())
    vel15_psi = eval(open("logs\\vel_15_psi.txt").read())
    vel15_dx = eval(open("logs\\vel_15_dx.txt").read())
    vel15_steer = eval(open("logs\\vel_15_steer.txt").read())


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



    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax2 = ax1.twinx() # psi
    ax3 = ax1.twinx() # dx

    ax3.spines.right.set_position(("axes", 1.2))
    ax1.invert_yaxis()

    ax3.plot(vel25_time, vel25_steer, 'g-', linewidth=0.5)
    ax2.plot(vel25_time, vel25_psi, 'b-', linewidth=0.5)
    ax1.plot(vel25_time, vel25_dx, 'r-', linewidth=0.5)


    align_yaxis_np([ax1, ax2, ax3])


    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    #plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    ax1.set_xlabel('Tempo [s]')
    ax3.set_ylabel('Estercamento [$^\circ$]', color='g')
    ax2.set_ylabel('$\psi$ [$^\circ$]', color='b')
    ax1.set_ylabel('$d_x$ [m]', color='r')


    ax1.grid()
    #plt.legend(loc='best')


   # plt.savefig('line_plot.pdf', bbox_inches='tight')  
    plt.show()




