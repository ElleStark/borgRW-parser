# Functions based on runtimeDiagnostics library by David Gold:
# https://github.com/davidfgold/runtimeDiagnostics/blob/master/runtime_visualization_functions.py#L17
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from yellowbrick.features import RadViz
def plot_text(ax, problem, num_obj, runtime, i):
    """

    Plots text in one subplot with summary information


    Parameter ax: a Matplotlib subplot
    Parameter probem: name of the problem (string)
    Parameter num_objectives: number of objectives
    Parameter freq: frequency of runtime output
    Parameter i: snapshot number

    """
    ax.text(0, 4, 'Runtime Diagnostic Dashboard', fontsize=28)
    ax.text(0, 3.5, 'Problem: ' + problem, fontsize=24)
    ax.text(0, 3, 'Num obj: ' + str(num_obj), fontsize=24)
    ax.text(0, 2, 'Current NFE: ' + str(runtime['NFE'][i]), fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')
    ax.set_ylim([0, 4])


def plot_3Dscatter(ax, objectives, i):
    """

    Plots a scatter 3D plot of objectives


    Parameter ax: a Matplotlib subplot
    Parameter objectives: a numpy array of objectives
    Parameter i: snapshot number

    """

    ax.scatter(objectives[i][:, 0], objectives[i][:, 1], objectives[i][:, 2], color='b', alpha=.8)
    ax.view_init(25, 10)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])
    ax.set_xlabel('Obj 1 $\longrightarrow$')
    ax.set_ylabel('$\longleftarrow$ Obj 2')
    ax.set_zlabel('Obj 3 $\longrightarrow$')


def plot_operators(ax, runtime, total_NFE, i):
    """

    Plots a line plot of operator probabilities vs. NFE


    Parameter ax: a Matplotlib subplot
    Parameter runtime: a dict storing runtime information
    Parameter total_NFE: total NFE for the MOEA run
    Parameter i: snapshot number

    """

    ax.plot(runtime['NFE'][:i], runtime['SBX'][:i], color='blue')
    ax.plot(runtime['NFE'][:i], runtime['DE'][:i], color='orange')
    ax.plot(runtime['NFE'][:i], runtime['PCX'][:i], color='green')
    ax.plot(runtime['NFE'][:i], runtime['SPX'][:i], color='red')
    ax.plot(runtime['NFE'][:i], runtime['UNDX'][:i], color='purple')
    ax.plot(runtime['NFE'][:i], runtime['UM'][:i], color='yellow')

    ax.set_xlim(0, total_NFE)
    ax.set_ylim(0, 1)
    ax.set_xlabel('NFE')
    ax.set_ylabel('Operator Probability')
    ax.legend(['SBX', 'DE', 'PCX', 'SPX', 'UNDX', 'UM'], loc='upper left', ncol=3)


def plot_metric(ax, metric, metric_name, NFE, total_NFE, ymax, i):
    """

    Plots a line plot of HV vs. NFE


    Parameter ax: a Matplotlib subplot
    Parameter metric: a numpy array of the performance metric
    Parameter NFE: a numpy array of NFE corresponding to HVs
    Parameter total_NFE: total NFE for the MOEA run
    Parameter ymax: max y-axis value
    Parameter i: snapshot number

    """

    ax.plot(NFE[:i], metric[:i], c='b')
    ax.set_xlim([0, total_NFE])
    ax.set_ylim([0, ymax])
    ax.set_ylabel(metric_name)


def plot_paxis(ax, objs, i, obj_names):
    """

    Plots a parallel axis plot of objectives


    Parameter ax: a Matplotlib subplot
    Parameter objs: a numpy array of objectives
    Parameter i: snapshot number

    """
    n_obj = len(obj_names)

    for pol in objs[i]:
        ys = pol
        xs = range(len(ys))
        ax.plot(xs, ys, c='b', alpha=.8, linewidth=.5)
    # for j in range(len(objs[i][:, 0])):
    #     ys = objs[i][j, :]
    #     xs = range(len(ys))
    #     ax.plot(xs, ys, c='b', alpha=.8, linewidth=.5)

    ax.set_ylabel('Objective val \n $\longleftarrow$ Preference', size=12)
    #ax.set_ylim([0, 2])
    #ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.setxticks(range(0, n_obj))
    ax.set_xticklabels(obj_names, fontsize=12)
    ax.set_xlim([0, n_obj])


def plot_Radvis(objectives, ax, name):
    class_dummy = np.zeros(len(objectives))
    visualizer = RadViz(classes=[name], ax=ax, alpha=.75)
    visualizer.fit(objectives, class_dummy)
    visualizer.show()




