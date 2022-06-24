    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

from matplotlib import pyplot as plt
import numpy as np

from scipy.interpolate import interp1d

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


def plot_map(map_x,map_y, figure = 'Figura'):

 # Obtendo dados dos arquivos





    # Trajeto
    fig = plt.figure(figure, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(map_x, map_y, 'b-', label='Percurso')
    #plt.plot(vel15_x, vel15_y, 'r--', label='Controle, v = 18')
    
    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    plt.arrow(map_x[0], map_y[0], map_x[1]-map_x[0], map_y[1]-map_y[0], shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
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

    #plt.savefig('4_percurso.pdf', bbox_inches='tight')  



def plot_error(time, psi, dx, steer, figure = None):

    time = np.array(time)
    time = time - time[0]

    fig, (axes) = plt.subplots(3, sharex=True)
 
    #fig, ax1 = plt.subplots()
    #fig.subplots_adjust(right=0.75)
    axes[0].get_shared_x_axes().join(axes[0], axes[1], axes[2])
    #ax2 = ax1.twinx() # psi
    #ax3 = ax1.twinx() # dx

    #ax3.spines.right.set_position(("axes", 1.2))
    axes[0].invert_yaxis()

    axes[2].plot(time, steer, 'g-', linewidth=0.5)
    axes[1].plot(time, psi, 'b-', linewidth=0.5)
    axes[0].plot(time, dx, 'r-', linewidth=0.5)



    align_yaxis_np([axes[0], axes[1], axes[2]])
    ranges = [(3, 6),(9,12)]

    for range in ranges:
        for ax in axes:
            ax.axvspan(*range, color='k', alpha=0.2)


    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    #plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    axes[2].set_xlabel('Tempo [s]')
    axes[2].set_ylabel('Estercamento $\delta$ [$^\circ$]')
    axes[1].set_ylabel('Erro angular $\psi$ [$^\circ$]')
    axes[0].set_ylabel('Erro lateral $d_x$ [m]')

    #ax1.set_xticklabels([])
    #ax2.set_xticklabels([])

    fig.set_figwidth(10)
    fig.set_figheight(12)
    
    #fig.xlim(-450,450)
    plt.xlim(0,time[len(time)-1])

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
    #plt.legend(loc='best')


  
    #plt.show()

def plot(time,data,legend, color, figure = None):
    time = np.array(time)
    time = time - time[0]

    fig = plt.figure(figure, frameon=False)
    ax = fig.add_subplot(1, 1, 1)



    #ax.invert_yaxis()

    ax.plot(time, data, color, linewidth=1)


    #plt.scatter(vel15_x[0], np.negative(vel15_y[0]), s = 100, marker = '*' , color = 'green', label='Inicio', zorder=3)
    #plt.arrow(map_x[0], map_y[0], 0, -1, shape='full', color='green', length_includes_head=True, zorder=0, head_length=40., head_width=20, label='Inicio')
    ax.set_xlabel('Tempo [s]')
    ax.set_ylabel(legend)


    fig.set_figwidth(10)
    fig.set_figheight(4)
    plt.xlim(0,time[len(time)-1])
    #fig.xlim(-450,450)


    ax.grid()

def get_arc_length(x, y):
    npts = len(x)

    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
    return arc


if __name__ == '__main__':


    map_x = eval(open("logs\\path_ideal_x.txt").read())
    map_y = eval(open("logs\\path_ideal_y.txt").read())

    #plot_map(map_x,map_y)

    
    vel25_time = eval(open("logs\\vel_25_time.txt").read())
    vel25_psi = eval(open("logs\\vel_25_psi.txt").read())
    vel25_dx = eval(open("logs\\vel_25_dx.txt").read())
    vel25_steer = eval(open("logs\\vel_25_steer.txt").read())

    #plot_error(vel25_time, vel25_psi, vel25_dx, vel25_steer)

    #plt.savefig('4_todos_dados.pdf', bbox_inches='tight')


    map_x = eval(open("logs\\path_ideal_x.txt").read())
    map_y = eval(open("logs\\path_ideal_y.txt").read())


    ideal_x = eval(open("ideal_x.txt").read())
    ideal_y = eval(open("ideal_y.txt").read())


    i = np.arange(len(ideal_x))
    interp_i = np.linspace(0, i.max(),   i.max())

    new_map_x = interp1d(i, ideal_x, kind='cubic')(interp_i)
    new_map_y = interp1d(i, ideal_y, kind='cubic')(interp_i)

    #plt.plot(new_map_x,new_map_y, color='r') #melhor curva
    #plt.plot(ideal_x,ideal_y)
    plot_map(new_map_x,new_map_y , 'mais um')

    plt.show()
    #get_curvature(new_map_x,new_map_y)
    

