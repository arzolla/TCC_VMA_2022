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
    plt.arrow(map_x[0], map_y[0], map_x[1]-map_x[0], map_y[1]-map_y[0], shape='full', color='green', length_includes_head=True, zorder=3, head_length=40., head_width=20, label='Inicio')
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
    
    points = [(325, 34), (-450,-75), (-425,-330),  (-75,-330),  (75,280),  (325,280),]
    index = ['A', 'B', 'C', 'D', 'E', 'F']
    for n, point in enumerate(points):

        plt.text(*point,  index[n], c='gray', ha='center', va='center', size='large', weight='roman')


    plt.legend(loc='upper left')





def plot_error(time, psi, dx, steer, figure = None):

    time = np.array(time)
    time = time - time[0]

    fig, (axes) = plt.subplots(3, sharex=True, num=figure)
 
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
    ranges = [(5, 7.5), (27.5,32.5), (35.5,45.0), (47.5,58.0),  (70.0,76.5),  (78.5,85)]
    ratio = 85.63046646118164/time[len(time)-1]
    ranges = [(range[0] / ratio, range[1] / ratio) for range in ranges]

    index = ['A', 'B', 'C', 'D', 'E', 'F']
    for n, range in enumerate(ranges):
        for ax in axes:
            ax.axvspan(*range, color='k', alpha=0.2)

        plt.text(((range[1]+range[0])/2),0.8*min(axes[2].get_ylim()),  index[n], c='gray', ha='center', size='large', weight='roman')


    axes[2].set_xlabel('Tempo [s]')
    axes[2].set_ylabel('Esterçamento  $\delta$ [$^\circ$]', family='serif', size='x-large')
    axes[1].set_ylabel('Erro angular  $\psi$ [$^\circ$]', family='serif', size='x-large')
    axes[0].set_ylabel('Erro lateral  $d_x$ [m]', family='serif', size='x-large')


    fig.set_figwidth(10)
    fig.set_figheight(10)
    print(time[len(time)-1])
    plt.xlim(0,time[len(time)-1])
    plt.xticks(np.arange(0, time[len(time)-1], step=5))
    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
    fig.tight_layout()
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
    
    print('------------------------------------------')
    from statistics import stdev
    # vels = [15, 20, 25]
    # for vel in vels:
    #     vel_time = eval(open("logs\\vel_"+str(vel)+"_time.txt").read())
    #     vel_psi = eval(open("logs\\vel_"+str(vel)+"_psi.txt").read())
    #     vel_dx = eval(open("logs\\vel_"+str(vel)+"_dx.txt").read())
    #     vel_steer = eval(open("logs\\vel_"+str(vel)+"_steer.txt").read())
    #     #print(vel_dx)
    #     plot_error(vel_time, vel_psi, vel_dx, vel_steer, figure=('vel_'+str(vel)))
    #     print('rms dx '+str(vel)+':',np.sqrt(np.mean(np.array(vel_dx)**2)))
    #     #print('rms psi '+str(vel),np.sqrt(np.mean(np.array(vel_psi)**2)))
    #     #print('stdev dx '+str(vel)+':',stdev(vel_dx))
    #     print('max, min dx '+str(vel)+':',np.round(min(vel_dx),3)," \\ ; \\ ", np.round(max(vel_dx),3) )
    #     print('------------------------------------------')

    
        #plt.savefig('4_todos_dados_'+str(vel)+'.png', bbox_inches='tight')


    ######## mapa #############

    # ideal_x = eval(open("logs\\ideal_x.txt").read())
    # ideal_y = eval(open("logs\\ideal_y.txt").read())


    # i = np.arange(len(ideal_x))
    # interp_i = np.linspace(0, i.max(),   i.max())

    # new_map_x = interp1d(i, ideal_x, kind='cubic')(interp_i)
    # new_map_y = interp1d(i, ideal_y, kind='cubic')(interp_i)

    #plot_map(new_map_x,new_map_y , 'mais um')

    #plt.savefig('4_percurso.png', bbox_inches='tight')

    # print(get_arc_length(new_map_x, new_map_y))
    
        

    #plt.show()

    vel_time = eval(open("logs\\time_processing_only.txt").read())
    vel_time = np.array(vel_time)

    periodo = []
    for n in range(len(vel_time)-1):
        print(n)
        print(vel_time[0])
        print(vel_time[1])
        periodo.append(vel_time[n+1] - vel_time[n])

    print('periodo lista',periodo)
    print('media:',np.mean(periodo))


    #print(periodo[0])

