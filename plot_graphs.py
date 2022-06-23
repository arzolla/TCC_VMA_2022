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


def plot_map():

 # Obtendo dados dos arquivos

    map_x = eval(open("logs\\path_ideal_x.txt").read())
    map_y = eval(open("logs\\path_ideal_y.txt").read())



    # Trajeto
    fig = plt.figure('Trajeto', frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(map_x, map_y, 'b-', label='Percurso')
    #plt.plot(vel15_x, vel15_y, 'r--', label='Controle, v = 18')
    
    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')



    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(-500, 501, 100)
    minor_ticks = np.arange(-550, 551, 50)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    #plt.xlim(-450,450)
    #plt.ylim(-450,450)

    ax.axis('square')
    
    plt.legend(loc='upper left')

    plt.savefig('4_percurso.pdf', bbox_inches='tight')  
    plt.show()



def plot_error(time, psi, dx, steer, figure = None):

    time = np.array(time)
    time = time - time[0]

    fig = plt.figure(figure, frameon=False)
    ax1 = fig.add_subplot(1, 1, 1)

    #fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax2 = ax1.twinx() # psi
    ax3 = ax1.twinx() # dx

    ax3.spines.right.set_position(("axes", 1.2))
    ax1.invert_yaxis()

    ax3.plot(time, steer, 'g-', linewidth=0.5)
    ax2.plot(time, psi, 'b-', linewidth=0.5)
    ax1.plot(time, dx, 'r-', linewidth=0.5)


    align_yaxis_np([ax1, ax2, ax3])


    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    #plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    ax1.set_xlabel('Tempo [s]')
    ax3.set_ylabel('Estercamento [$^\circ$]', color='g')
    ax2.set_ylabel('$\psi$ [$^\circ$]', color='b')
    ax1.set_ylabel('$d_x$ [m]', color='r')


    ax1.grid()
    #plt.legend(loc='best')


  
    #plt.show()


if __name__ == '__main__':

    #plot_map()

    
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
    plot_error(vel25_time, vel25_psi, vel25_dx, vel25_steer)
    plot_error(vel20_time, vel20_psi, vel20_dx, vel20_steer)
    plot_error(vel15_time, vel15_psi, vel15_dx, vel15_steer)
    plt.show()

