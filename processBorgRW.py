"""Example usage script"""

import borg_parser
from matplotlib import pyplot as plt
from celluloid import Camera

def main():
    # Change path name to your desired runtime file to analyze
    path_to_runtime = borg_parser.datasets.BorgRW_data('data/T3_FE20000_allC_8Traces/RunTime.Parsable.txt')

    decision_names = ["Mead_Surplus_DV Row cat 0",
                      "Mead_Surplus_DV Row cat 1",
                      "Mead_Shortage_e_DV Row cat 0",
                      "Mead_Shortage_e_DV Row cat 1",
                      "Mead_Shortage_e_DV Row cat 2",
                      "Mead_Shortage_e_DV Row cat 3",
                      "Mead_Shortage_e_DV Row cat 4",
                      "Mead_Shortage_e_DV Row cat 5",
                      "Mead_Shortage_e_DV Row cat 6",
                      "Mead_Shortage_e_DV Row cat 7",
                      "Mead_Shortage_V_DV Row cat 0",
                      "Mead_Shortage_V_DV Row cat 1",
                      "Mead_Shortage_V_DV Row cat 2",
                      "Mead_Shortage_V_DV Row cat 3",
                      "Mead_Shortage_V_DV Row cat 4",
                      "Mead_Shortage_V_DV Row cat 5",
                      "Mead_Shortage_V_DV Row cat 6",
                      "Mead_Shortage_V_DV Row cat 7",
                      "Powell_Tier_Elevation_DV.txt Row cat 0",
                      "Powell_Tier_Elevation_DV.txt Row cat 1",
                      "Powell_Tier_Elevation_DV.txt Row cat 2",
                      "Powell_Tier_Elevation_DV.txt Row cat 3",
                      "Powell_Tier_Elevation_DV.txt Row cat 4",
                      "Powell_Mead_Reference_Elevation_DV Row cat 0",
                      "Powell_Mead_Reference_Elevation_DV Row cat 1",
                      "Powell_Mead_Reference_Elevation_DV Row cat 2",
                      "Powell_Mead_Reference_Elevation_DV Row cat 3",
                      "Powell_Mead_Reference_Elevation_DV Row cat 4",
                      "Powell_Range_Number_DV Row cat 0", "Powell_Range_Number_DV Row cat 1",
                      "Powell_Range_Number_DV Row cat 2", "Powell_Range_Number_DV Row cat 3",
                      "Powell_Range_Number_DV Row cat 4", "Powell_Balance_Tier_Variable_DV Row cat 0",
                      "Powell_Balance_Tier_Variable_DV Row cat 1", "Powell_Balance_Tier_Variable_DV Row cat 2",
                      "Powell_Balance_Tier_Variable_DV Row cat 3", "Powell_Balance_Tier_Variable_DV Row cat 4",
                      "Powell_Primary_Release_Volume_DV Row cat 0", "Powell_Primary_Release_Volume_DV Row cat 1",
                      "Powell_Primary_Release_Volume_DV Row cat 2", "Powell_Primary_Release_Volume_DV Row cat 3",
                      "Powell_Primary_Release_Volume_DV Row cat 4", "Powell_Increment_EQ_DV",
                      ]

    objective_names = [
        "P.3490", "P.WY.Release",
        "LF.Def", "Avg.Combo.Stor",
        "M.1000", "Avg.LB.Short",
        "Max.LB.Short", "Max.Delta.Short",
        ]

    metric_names = []

    constraint_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

    # Create runtime object
    runtime = borg_parser.BorgRuntimeDiagnostic(
        path_to_runtime,
        n_decisions=len(decision_names),
        n_objectives=len(objective_names),
        n_metrics=len(metric_names),
        n_constraints=len(constraint_names)
    )
    runtime.set_decision_names(decision_names)
    runtime.set_objective_names(objective_names)
    runtime.set_metric_names(metric_names)
    runtime.set_constraint_names(constraint_names)

    # Get NFEs to first feasible solution
    first_feasible = runtime.get_first_feasible()
    print(first_feasible)

    # Parallel coordinates plot of objectives at multiple NFEs
    obj_plot = runtime.plot_objectives_multiple_nfes()
    obj_plot.to_html("borgRW_objectives_nfes.html")

    # Separate parallel coordinates plots for objectives at each NFE
    nfe_targets = [1000, 4000, 10000, 20000]
    nfe_list = runtime.get_NFEs_from_targets(target_list=nfe_targets)
    obj_ranges = runtime.get_objective_ranges()
    for nfe in nfe_list:
        obj_plot = runtime.plot_objectives_parcoord(obj_ranges=obj_ranges, nfe=nfe)
        file_name = 'BorgRW_objs_NFE' + str(nfe) + '.html'
        obj_plot.to_html(file_name)


    # Animated dashboard, code modified from David Gold's runtimeDiagnostics library
    # source: https://github.com/davidfgold/runtimeDiagnostics/blob/master/rutime_vis_main.py
    # blog post: https://waterprogramming.wordpress.com/2020/05/06/beyond-hypervolume-dynamic-visualization-of-moea-runtime/
    ################### Diagnostic Dashboard ##############################

    # Get snapshots of run at desired frequency (since our runtime has many FEs)
    # Number of intervals is approximate, since Runtime file doesn't have predictable FE intervals (due to restarts etc)
    # This code uses floor division to calculate sampling interval to get close to number of desired intervals
    n_intervals = 200
    snaps = runtime.get_snapshots(n_intervals)
    objs = snaps['Objectives']
    HV = snaps['Hypervolume']

    # create the figure object to store subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2)

    # information axis
    text_ax = fig.add_subplot(gs[0:1, 0])

    # parallel axis plot axis
    px_ax = fig.add_subplot(gs[1, :])

    # HV axis
    HV_ax = fig.add_subplot(gs[2, :])

    # operator probabilities
    op_ax = fig.add_subplot(gs[3, :])

    # set up camera for animation
    camera = Camera(fig)
    freq = snaps['NFE'][1] - snaps['NFE'][0]
    total_NFE = snaps['NFE'][-1]

    # loop through runtime snapshots and plot data
    # capture each with camera
    for i in range(0, len(snaps['NFE'])):
        visualization_functions.plot_text(text_ax, 'Baseline', 8, snaps, i)
        visualization_functions.plot_operators(op_ax, snaps, total_NFE, i)
        visualization_functions.plot_metric(HV_ax, HV, "Hypervolume", snaps['NFE'], total_NFE, HV[-1], i)
        visualization_functions.plot_paxis(px_ax, objs, i, objective_names)
        fig.tight_layout()
        camera.snap()

    # use Celluloid to stitch animation
    animation = camera.animate()

    animation.save('BorgRW_runtime.gif', writer='PillowWriter')
################################ End Animated Dashboard #########################################

    # Improvements (epsilon progress) vs. NFEs line plot
    fig = runtime.plot_improvements()
    fig.savefig("borgRW_improvements.jpg")

    # Objectives parallel coordinates plot (final archive)
    obj_plot = runtime.plot_objectives_parcoord()
    obj_plot.to_html("borgRW_objectives.html")

    # Decisions parallel coordinates plot (final archive)
    mead_plot, powell_plot = runtime.plot_decisions_parcoord()
    mead_plot.to_html('borgRW_mead_decisions.html')
    powell_plot.to_html('borgRW_powell_decisions.html')

    # Extreme Point metrics vs. NFEs line plots
    nadir_plot = runtime.plot_real_nadir_change()
    nadir_plot.savefig('nadir_change.jpg')
    ideal_plot = runtime.plot_real_ideal_change()
    ideal_plot.savefig('ideal_change.jpg')

    # Hypervolume Indicator vs. NFEs line plot
    hv_plot = runtime.plot_hypervolume()
    hv_plot.savefig("borgRW_hypervolume.jpg")


if __name__ == '__main__':
    main()