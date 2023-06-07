"""Example usage script"""

import borg_parser
from matplotlib import pyplot as plt
from celluloid import Camera
import visualization_functions


def main():
    # Change path name to your desired runtime file to analyze
    path_to_runtime = borg_parser.datasets.BorgRW_data('data/T3_FE20000_allC_8Traces_partial15300/RunTime.Parsable.txt')

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
        "Powell_3490", "Powell_WY_Release",
        "Lee_Ferry_Deficit", "Avg_Combo_Storage",
        "Mead_1000", "LB_Shortage_Volume",
        "Max_Annual_LB_Shortage", "Max_Delta_Annual_Shortage",
        ]

    # ranges used for parallel coordinates plotting
    objective_ranges = [
        ('Powell_3490', 0, 25), ('Powell_WY_Release', 8600000, 8780000),
        ('Lee_Ferry_Deficit', 0, 15), ('Avg_Combo_Shortage', -13900000, -11200000),
        ('Mead_1000', 0, 50), ('LB_Shortage_Volume', 940000, 1130000),
        ('Max_Annual_LB_Shortage', 1400000, 2400000), ('Max_Delta_Annual_Shortage', 900000, 2400000)
    ]

# Very wide range of objectives:
    # objective_ranges = [
    #     ('Powell_3490', 0, 100), ('Powell_WY_Release', 8000000, 9500000),
    #     ('Lee_Ferry_Deficit', 0, 100), ('Avg_Combo_Shortage', -21000000, -4000000),
    #     ('Mead_1000', 0, 100), ('LB_Shortage_Volume', 500000, 1700000),
    #     ('Max_Annual_LB_Shortage', 0, 2400000), ('Max_Delta_Annual_Shortage', 0, 2400000)
    # ]

    metric_names = []

    # Create runtime object
    runtime = borg_parser.BorgRuntimeDiagnostic(
        path_to_runtime,
        n_decisions=len(decision_names),
        n_objectives=len(objective_names),
        n_metrics=len(metric_names),
    )
    runtime.set_decision_names(decision_names)
    runtime.set_objective_names(objective_names)
    runtime.set_metric_names(metric_names)

    # Animated dashboard, code modified from David Gold's runtimeDiagnostics library
    # source: https://github.com/davidfgold/runtimeDiagnostics/blob/master/rutime_vis_main.py
    # blog post: https://waterprogramming.wordpress.com/2020/05/06/beyond-hypervolume-dynamic-visualization-of-moea-runtime/
    ################### Diagnostic Dashboard ##############################

    # Get snapshots of run at desired frequency (since our runtime has every FE) - added by EStark
    # For 20,000 FEs, every 100 FEs should produce nice animation

    # Number of runtime intervals is approximate, since the Runtime file doesn't actually output every FE
    # This code uses floor division for index sampling interval to get close to number of desired intervals
    snaps = runtime.get_snapshots(200)
    #objs = snaps['Objectives']

    # create the figure object to store subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 2)

    # information axis
    text_ax = fig.add_subplot(gs[0:2, 0])

    # 3D scatter axis
    scatter_ax = fig.add_subplot(gs[0:2, 1], projection='3d')

    # parallel axis plot axis
    px_ax = fig.add_subplot(gs[2, :])

    # HV axis
    HV_ax = fig.add_subplot(gs[3, :])

    # operator probabilities
    op_ax = fig.add_subplot(gs[4, :])

    # set up camera for animation
    camera = Camera(fig)
    freq = snaps['NFE'][1] - snaps['NFE'][0]
    total_NFE = snaps['NFE'][-1]

    # loop through runtime snapshots and plot data
    # capture each with camera
    for i in range(0, len(snaps['NFE'])):
        visualization_functions.plot_text(text_ax, 'Baseline', 8, snaps, i)
        #visualization_functions.plot_3Dscatter(scatter_ax, objs_3, i)
        visualization_functions.plot_operators(op_ax, snaps, total_NFE, i)
        #visualization_functions.plot_metric(HV_ax, HV, "Hypervolume", seed0['NFE'], len(seed0['NFE']) * 25, 1, i)
        #visualization_functions.plot_paxis(px_ax, objs, i)
        fig.tight_layout()
        camera.snap()

    # use Celluloid to stitch animation
    animation = camera.animate()

    animation.save('BorgRW_runtime.gif', writer='PillowWriter')
##################################################################################################

    # # Improvements
    # fig = runtime.plot_improvements()
    # fig.savefig("borgRW_improvements.jpg")
    #
    # # Objectives
    # obj_plot = runtime.plot_objectives_parcoord()
    # obj_plot.to_html("borgRW_objectives.html")
    #
    # # Decisions
    # mead_plot, powell_plot = runtime.plot_decisions_parcoord()
    # mead_plot.to_html('borgRW_mead_decisions.html')
    # powell_plot.to_html('borgRW_powell_decisions.html')
    #
    # # Extreme Point metrics
    # nadir_plot = runtime.plot_real_nadir_change()
    # nadir_plot.savefig('nadir_change.jpg')
    # ideal_plot = runtime.plot_real_ideal_change()
    # ideal_plot.savefig('ideal_change.jpg')
    #
    # # Hypervolume
    # #reference = [0, 0, 0, -60000000, 0, 0, 0, 0]
    # reference = [100, 10000000, 100, 0, 100, 2400000, 2400000, 2400000]
    # hv_plot = runtime.plot_hypervolume(reference)
    # hv_plot.savefig("borgRW_hypervolume.jpg")

if __name__ == '__main__':
    main()