"""Script to compare multiple runs with different constraints.
Runs must all have the same decision variables and objectives"""

import borg_parser
import hiplot as hip
import pygmo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Runtime path dictionary. keys are run names, values are paths to runtime
    runtime_paths = {'8 Obj No Constraints': 'data/Exp1_FE3600_8Obj_noC/RunTime.Parsable.txt',
                     '4 Obj No Constraints': 'data/Exp3_FE3600_4Obj_noC/RunTime.Parsable.txt'}

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

    objective_names_dict = {'8 Obj No Constraints': [
        "P.3490", "P.WY.Release",
        "LF.Def", "Avg.Combo.Stor",
        "M.1000", "Avg.LB.Short",
        "Max.LB.Short", "Max.Delta.Short",
        ], '4 Obj No Constraints': [
        "P.3490", "P.WY.Release",
        "M.1000", "Avg.LB.Short"
        ]
        }

    long_objective_names = [
        'Objectives.Objective_Powell_3490',
        'Objectives.Objective_Powell_WY_Release',
        'Objectives.Objective_Lee_Ferry_Deficit',
        'Objectives.Objective_Avg_Combo_Storage',
        'Objectives.Objective_Mead_1000',
        'Objectives.Objective_LB_Shortage_Volume',
        'Objectives.Objective_Max_Annual_LB_Shortage',
        'Objectives.Objective_Max_Delta_Annual_Shortage'
        ]

    metric_names = []

    constraint_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

    # Create dictionary of runtime objects for all runs
    runtime_dict = {}
    for name, path in runtime_paths.items():
        objective_names = objective_names_dict[name]
        n_decisions = len(decision_names)
        n_objectives = len(objective_names)
        n_metrics = len(metric_names)
        n_constraints = len(constraint_names)
        path_to_runtime = borg_parser.datasets.BorgRW_data(path)
        #constraint_names = constraint_dict[name]
        # Create runtime object
        runtime = borg_parser.BorgRuntimeDiagnostic(
                path_to_runtime,
                n_decisions=n_decisions,
                n_objectives=n_objectives,
                n_metrics=n_metrics,
                n_constraints=n_constraints
        )
        runtime.set_decision_names(decision_names)
        runtime.set_objective_names(objective_names)
        runtime.set_metric_names(metric_names)
        runtime.set_constraint_names(constraint_names)

        # If less than 8 objectives, need to add other objectives to compare sets in full objective space
        if n_objectives < 8:
            # get remaining objectives from metrics tracked during runtime in all values file
            all_vals = pd.read_table('src/borg_parser/data/Exp3_FE3600_4Obj_noC/AllValues.txt', delimiter=' ')
            all_vals_obj = all_vals.iloc[:, n_decisions:(n_decisions + n_objectives)]

            full_obj_set = {}
            # Loop through archive of objectives at each NFE and find corresponding metrics (max across traces)
            for nfe, objs in runtime.archive_objectives.items():
                # lookup each row in objs and take metric vals for remaining objectives
                obj_list = []  # list of lists of objectives for archive at a given NFE
                for obj in objs:
                    # metrics are last columns (order: DVs, objectives, constraints, metrics)
                    truth_df = all_vals_obj == obj
                    archive_pol = all_vals.loc[truth_df.all(axis=1) == True, ["Objectives.Objective_Powell_3490",
                    "Objectives.Objective_Powell_WY_Release",
                    "Metrics.Objective_Lee_Ferry_Deficit_Avg",
                    "Metrics.Objective_Avg_Combo_Storage_Avg",
                    "Objectives.Objective_Mead_1000",
                    "Objectives.Objective_LB_Shortage_Volume",
                    "Metrics.Objective_Max_Annual_LB_Shortage_Avg",
                    "Metrics.Objective_Max_Delta_Annual_Shortage_Avg"]]
                    archive_pol = archive_pol.iloc[0, :]
                    archive_pol = archive_pol.tolist()
                    obj_list.append(archive_pol)
                full_obj_set[nfe] = obj_list

        # Replace archive_objectives with all 8 objectives at each NFE
            runtime.archive_objectives = full_obj_set
            runtime.n_objectives = 8

        runtime_dict[name] = runtime

############### Code below for converting from average to max objectives for a run if needed ########################
    #
    # # Convert archive_objectives for '4 Constraints on Max' run to metrics so that we can compare to other runs
    # obj_maxC_df = pd.read_csv('src/borg_parser/data/T6_FE5000_MaxC_AveObj_8Traces/metrics_max.csv')
    #
    # # Look up max obj metrics for archive objectives at each NFE (AllValues file doesn't include archive at each FE)
    # all_vals = pd.read_table('src/borg_parser/data/T6_FE5000_MaxC_AveObj_8Traces/AllValues.txt', delimiter=' ')
    # all_vals_obj = all_vals.iloc[:, n_decisions:(n_decisions + n_objectives)]
    #
    # rt_max = runtime_dict['4 Constraints on Max']
    # objs_max = {}
    # # start_idx = 0
    # allval_metric_names = all_vals.columns[-8:]
    #
    # # Loop through archive of objectives at each NFE and find corresponding metrics (max across traces)
    # for nfe, objs in rt_max.archive_objectives.items():
    #     # lookup each row in objs and take metric vals
    #     max_list = [] # list of lists of objectives for archive at a given NFE
    #     for obj in objs:
    #         # metrics are last columns (order: DVs, objectives, constraints, metrics)
    #         truth_df = all_vals_obj == obj
    #         archive_pol = all_vals.loc[truth_df.all(axis=1) == True, :]
    #         archive_pol = archive_pol.iloc[0, -n_objectives:]
    #         archive_pol = archive_pol.tolist()
    #         max_list.append(archive_pol)
    #     objs_max[nfe] = max_list
    #
    # # Replace archive_objectives for run with max objectives
    # rt_max.archive_objectives = objs_max
    #
    # # Replace run dictionary item with updated dictionary for '4 C on max' item
    # runtime_dict['4 Constraints on Max'] = rt_max

######################### End conversion from average to max objectives #############################################


    # Get first feasible for each run
    # first_feasible = []
    # for run in runtime_dict.values():
    #     ff = run.get_first_feasible()
    #     first_feasible.append(ff)
    # print(first_feasible)

    # For reference point, use max of max_nadir for each run
    # Also calculate minimum of min_ideal for extreme point changes
    max_list = []
    min_list = []
    for run in runtime_dict.values():
        run.compute_real_nadir()
        run.compute_real_ideal()
        max = run.max_nadir
        max_list.append(max)
        min = run.min_ideal
        min_list.append(min)

    hv = pygmo.hypervolume(max_list)
    refpoint = hv.refpoint()
    ideal_pt = pygmo.ideal(min_list)

    # Create two dataframes, one for final archive objectives and one for runtime metrics
    metrics_df = pd.DataFrame()
    obj_df = pd.DataFrame()

    for name, run in runtime_dict.items():
        # Create dataframe with final archive objectives
        nfe = run.nfe[-1]  # final front
        #nfe = 3600  # partial run results
        o_temp_df = pd.DataFrame(run.archive_objectives[nfe], columns=objective_names_dict['8 Obj No Constraints'])
        o_temp_df['Run'] = name
        obj_df = pd.concat([obj_df, o_temp_df], ignore_index=True)

        # Create dataframe with hypervolume, improvements, extreme point changes
        run.compute_hypervolume(reference_point=refpoint)
        run.compute_extreme_pt_changes(tau_ideal=ideal_pt, tau_nadir=refpoint)
        temp_df = pd.DataFrame()
        temp_df['Hypervolume'] = pd.Series(run.hypervolume)
        temp_df['Improvements'] = pd.Series(run.improvements)
        temp_df['Ideal_Change'] = pd.Series(run.ideal_change)
        temp_df['Nadir_Change'] = pd.Series(run.nadir_change)
        temp_df['NFE'] = temp_df.index
        temp_df['Run'] = name
        metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)

    # Plot Metrics for each run
    metric_list = ['Hypervolume', 'Improvements', 'Ideal_Change', 'Nadir_Change']
    for metric in metric_list:
        sns.set()
        fig, ax = plt.subplots()
        sns.lineplot(
            data=metrics_df,
            x='NFE',
            y=metric,
            ax=ax,
            hue='Run'
        )
        plt.ylabel(metric)
        plt.xlabel('Function Evaluations')
        ax.set_xlim(100, 5750)
        fig.show()

    # Plot final archive objectives for each run in parallel coordinates
    col_names = objective_names
    col_names.append('Run')
    cols = col_names
    cols.reverse()
    color_col = 'Run'
    exp = hip.Experiment.from_dataframe(obj_df)
    exp.parameters_definition[color_col].colormap = 'interpolateViridis'
    exp.parameters_definition['LF.Def'].type = hip.ValueType.NUMERIC  # LF Def sometimes detected as categorical
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        {'order': cols,
         'hide': ['uid', 'from_uid']}
    )
    exp.display_data(hip.Displays.TABLE).update(
        {'hide': ['uid', 'from_uid']}
    )

    exp.to_html('ConstraintsTests_allPolicies.html')

    # Create dataframe of objectives on max trace at ~5kFE for comparing policy sets
    # # Add archive of objectives for 2 constraints, 4 constraints at ~5k FE
    # rt = runtime_dict['2 Scaled Constraints']
    # nfe = rt.nfe[-1]
    # df_objs = pd.DataFrame(
    #     rt.archive_objectives[nfe],
    #     columns=objective_names
    # )
    # df_objs['Run'] = '2C_scaled'
    #
    # # for 4 constraints run, FE 5319 is closest actual FE to 5000 due to restart
    # rt = runtime_dict['4 Unscaled Constraints']
    # nfe = 5319
    # obj_4c_df = pd.DataFrame(
    #     rt.archive_objectives[nfe],
    #     columns=objective_names
    # )
    # obj_4c_df['Run'] = '4C_unscaled'

    # df_objs = pd.concat([df_objs, obj_4c_df], ignore_index=True)
    #
    # # 68 policies in final archive:
    # obj_maxC_df = obj_maxC_df.tail(68)
    # obj_maxC_df['Run'] = '4C_onMax'
    # df_objs = pd.concat([df_objs, obj_maxC_df], ignore_index=True)

    # Perform non-dominated sorting
    ndf, dl, dc, ndl = pygmo.fast_non_dominated_sorting(obj_df.loc[:, obj_df.columns != 'Run'])
    nondom_df = obj_df.loc[ndf[0], :]

    # Plot non-dominated parcoords plot
    col_names = objective_names
    col_names.append('Run')
    cols = col_names
    cols.reverse()
    color_col = 'Run'
    exp = hip.Experiment.from_dataframe(nondom_df)
    exp.parameters_definition[color_col].colormap = 'interpolateViridis'
    exp.parameters_definition['LF.Def'].type = hip.ValueType.NUMERIC  # LF Def sometimes detected as categorical
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        {'order': cols,
         'hide': ['uid', 'from_uid']}
    )
    exp.display_data(hip.Displays.TABLE).update(
        {'hide': ['uid', 'from_uid']}
    )

    exp.to_html('ConstraintsTests_nondominated.html')


if __name__ == '__main__':
        main()