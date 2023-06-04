"""Example usage script"""

import borg_parser
from pymoo.indicators.hv import Hypervolume

def main():
    # Change path name to your desired runtime file to analyze
    path_to_runtime = borg_parser.datasets.BorgRW_data('data/T3_FE20000_allC_8Traces_partial6.2/RunTime.Parsable.txt')

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

    # Improvements
    #fig = runtime.plot_improvements()
    #fig.savefig("borgRW_improvements.jpg")

    # Objectives
    #obj_plot = runtime.plot_objectives_parcoord(objective_ranges)
    #obj_plot.to_html("borgRW_objectives.html")

    # Test Ideal calcs
    nadir_plot = runtime.plot_real_nadir_change()
    nadir_plot.savefig('nadir_change.jpg')

    # Hypervolume
    #reference = [0, 0, 0, -60000000, 0, 0, 0, 0]
    reference = [100, 10000000, 100, 0, 100, 2400000, 2400000, 2400000]
    #hv_plot = runtime.plot_hypervolume(reference)
    #hv_plot.savefig("borgRW_hypervolume.jpg")

if __name__ == '__main__':
    main()