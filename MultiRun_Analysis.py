"""Script to compare multiple runs with different constraints.
Runs must all have the same decision variables and objectives"""
from platypus.core import Problem, EpsilonBoxArchive, Solution

import borg_parser
import hiplot as hip
import pygmo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from platypus import *

def main():
    # Runtime path dictionary. keys are run names, values are paths to runtime
    runtime_paths = {'Seed 8 Min Mead': 'data/Final_runs/CurrentParadigm_seed8_base_minMead/RunTime.Parsable.txt',
                     'Seed 26 Min Mead': 'data/Final_runs/CurrentParadigm_seed26_base_minMead/RunTime.Parsable.txt',
                    }

    allvalues_paths = {'Seed 8 Min Mead': 'C:/Users/elles/Documents/CU_Boulder/Research/Borg_processing_code/borgRW-parser/src/borg_parser/data/Final_runs/CurrentParadigm_seed8_base_minMead/AllValues.txt',
                      'Seed 26 Min Mead': 'C:/Users/elles/Documents/CU_Boulder/Research/Borg_processing_code/borgRW-parser/src/borg_parser/data/Final_runs/CurrentParadigm_seed26_base_minMead/AllValues.txt'}

    decision_names = ["Mead_Surplus_DV",
                      "Mead_Shortage_e_DV Row 0",
                      "Mead_Shortage_e_DV Row 1",
                      "Mead_Shortage_e_DV Row 2",
                      "Mead_Shortage_e_DV Row 3",
                      "Mead_Shortage_e_DV Row 4",
                      "Mead_Shortage_e_DV Row 5",
                      "Mead_Shortage_e_DV Row 6",
                      "Mead_Shortage_e_DV Row 7",
                      "Mead_Shortage_V_DV Row 0",
                      "Mead_Shortage_V_DV Row 1",
                      "Mead_Shortage_V_DV Row 2",
                      "Mead_Shortage_V_DV Row 3",
                      "Mead_Shortage_V_DV Row 4",
                      "Mead_Shortage_V_DV Row 5",
                      "Mead_Shortage_V_DV Row 6",
                      "Mead_Shortage_V_DV Row 7",
                      "Powell_Tier_Elevation_DV Row 0",
                      "Powell_Tier_Elevation_DV Row 1",
                      "Powell_Tier_Elevation_DV Row 2",
                      "Powell_Tier_Elevation_DV Row 3",
                      "Powell_Tier_Elevation_DV Row 4",
                      "Powell_Primary_Release_Volume_DV Row cat 0", "Powell_Primary_Release_Volume_DV Row cat 1",
                      "Powell_Primary_Release_Volume_DV Row cat 2", "Powell_Primary_Release_Volume_DV Row cat 3",
                      "Powell_Primary_Release_Volume_DV Row cat 4",
                      "Powell_Mead_Reference_Elevation_DV Row 0",
                      "Powell_Mead_Reference_Elevation_DV Row 1",
                      "Powell_Mead_Reference_Elevation_DV Row 2",
                      "Powell_Mead_Reference_Elevation_DV Row 3",
                      "Powell_Mead_Reference_Elevation_DV Row 4",
                      "Powell_Balance_Max_Offset_DV Row 0",
                      "Powell_Balance_Max_Offset_DV Row 1",
                      "Powell_Balance_Max_Offset_DV Row 2",
                      "Powell_Balance_Max_Offset_DV Row 3",
                      "Powell_Balance_Max_Offset_DV Row 4",
                      "Powell_Balance_Min_Offset_DV Row 0",
                      "Powell_Balance_Min_Offset_DV Row 1",
                      "Powell_Balance_Min_Offset_DV Row 2",
                      "Powell_Balance_Min_Offset_DV Row 3",
                      "Powell_Balance_Min_Offset_DV Row 4",
                      ]

    objective_names = [
        "P.3490", "P.WY.Release",
        "M.1000", "Avg.LB.Short"
    ]

    metric_names = []

    constraint_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

    # Create dictionary of runtime objects for all runs
    runtime_dict = {}
    for name, path in runtime_paths.items():
        #objective_names = objective_names_dict[name]
        n_decisions = len(decision_names)
        n_objectives = len(objective_names)
        n_metrics = len(metric_names)
        n_constraints = len(constraint_names)
        path_to_runtime = borg_parser.datasets.BorgRW_data(path)

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

        # If less than 8 objectives, can get metrics from AllValues.txt to compare sets in 8-objective space if desired:
        # runtime.convert_to_8_objs(name, allvalues_paths, n_decisions, n_objectives) # CHECK ALLVALUES PATHS!

        runtime_dict[name] = runtime

##########CODE TO FILTER POLICIES BY MAX OBJECTIVE VALUE IF DESIRED#################

    # obj_df = pd.DataFrame()
    # obj_names = [
    #     "P.3490", "P.WY.Release",
    #     "LF.Def", "Avg.Combo.Stor",
    #     "M.1000", "Avg.LB.Short",
    #     "Max.LB.Short", "Max.Delta.Short",
    #     ]
    #
    # for name, run in runtime_dict.items():
    #     # Create dataframe with final archive objectives
    #     nfe = run.nfe[-1]  # final front
    #     #nfe = 1889  # partial run results
    #     #while nfe not in run.nfe:
    #     #    nfe += 1
    #     o_temp_df = pd.DataFrame(run.archive_objectives[nfe], columns=obj_names)
    #     o_temp_df['Run'] = name
    #     obj_df = pd.concat([obj_df, o_temp_df], ignore_index=True)
    # # Plot UNFILTERED final archive objectives for each run in parallel coordinates
    # # With defined axes ranges to compare with filtered version
    # obj_ranges = [["P.3490",0,100], ["P.WY.Release", 7835000, 8875000],
    #     ["LF.Def",0,35], ["Avg.Combo.Stor",-30000000, -9000000],
    #     ["M.1000",0,100], ["Avg.LB.Short",275000, 1600000],
    #     ["Max.LB.Short", 100000, 2400000], ["Max.Delta.Short", 12500, 2400000]]
    #
    # col_names = obj_names.copy()
    # col_names.append('Run')
    # cols = col_names
    # cols.reverse()
    # color_col = 'Run'
    # exp = hip.Experiment.from_dataframe(obj_df)
    # exp.parameters_definition[color_col].colormap = 'interpolateViridis'
    # exp.parameters_definition['LF.Def'].type = hip.ValueType.NUMERIC  # LF Def sometimes detected as categorical
    # exp.display_data(hip.Displays.PARALLEL_PLOT).update(
    #     {'order': cols,
    #      'hide': ['uid', 'from_uid']}
    # )
    # exp.display_data(hip.Displays.TABLE).update(
    #     {'hide': ['uid', 'from_uid']}
    # )
    # for name, low, high in obj_ranges:
    #     exp.parameters_definition[name].force_range(low, high)
    #
    # exp.to_html('ConstraintTests_unfiltered_allPolicies.html')

########### END FILTERING ######################


############### Code below for converting from average to max objectives for a run if needed ########################

    # # Dictionary of runs that need a filter on the archive to mimic constraints
    # # With its associated path to the all values file
    # runs_to_constrain = {'8 Obj No Const': 'src/borg_parser/data/Exp1_FE3600_8Obj_noC/AllValues.txt',
    #                      '4 Obj No Const': 'src/borg_parser/data/Exp3_FE3600_4Obj_noC/AllValues.txt'}
    #
    # obj_labels = {'8 Obj No Const': ["Objectives.Objective_Powell_3490", "Objectives.Objective_Powell_WY_Release",
    #                "Objectives.Objective_Lee_Ferry_Deficit", "Objectives.Objective_Avg_Combo_Storage",
    #                "Objectives.Objective_Mead_1000", "Objectives.Objective_LB_Shortage_Volume",
    #                "Objectives.Objective_Max_Annual_LB_Shortage", "Objectives.Objective_Max_Delta_Annual_Shortage"],
    #               '4 Obj No Const': ["Objectives.Objective_Powell_3490",
    #                 "Objectives.Objective_Powell_WY_Release",
    #                 "Metrics.Objective_Lee_Ferry_Deficit_Avg",
    #                 "Metrics.Objective_Avg_Combo_Storage_Avg",
    #                 "Objectives.Objective_Mead_1000",
    #                 "Objectives.Objective_LB_Shortage_Volume",
    #                 "Metrics.Objective_Max_Annual_LB_Shortage_Avg",
    #                 "Metrics.Objective_Max_Delta_Annual_Shortage_Avg"]
    #               }
    #
    # max_obj_labels = ["Metrics.Objective_Powell_3490_Max", "Metrics.Objective_Powell_WY_Release_Max",
    #                   "Metrics.Objective_Lee_Ferry_Deficit_Max", "Metrics.Objective_Avg_Combo_Storage_Max",
    #                   "Metrics.Objective_Mead_1000_Max", "Metrics.Objective_LB_Shortage_Volume_Max",
    #                   "Metrics.Objective_Max_Annual_LB_Shortage_Max", "Metrics.Objective_Max_Delta_Annual_Shortage_Max"]
    #
    # for name, path in runs_to_constrain.items():
    #     # Look up max obj metrics for archive objectives at each NFE (AllValues file doesn't include archive at each FE)
    #     all_vals = pd.read_table(path, delimiter=' ')
    #     all_vals_obj = all_vals.loc[:, obj_labels[name]]
    #
    #     rt_max = runtime_dict[name]
    #     objs_max = {}
    #
    #     # Loop through archive of objectives at each NFE and find corresponding metrics (max across traces)
    #     for nfe, objs in rt_max.archive_objectives.items():
    #         # lookup each row in objs and take metric vals
    #         max_list = [] # list of lists of objectives for archive at a given NFE
    #         for obj in objs:
    #             # metrics are last columns (order: DVs, objectives, constraints, metrics)
    #             truth_df = all_vals_obj == obj
    #             archive_pol = all_vals.loc[truth_df.all(axis=1) == True, max_obj_labels]
    #             archive_pol = archive_pol.iloc[0, :]
    #             archive_pol = archive_pol.tolist()
    #             max_list.append(archive_pol)
    #         objs_max[nfe] = max_list
    #
    #     # Replace archive_objectives for run with max objectives
    #     # rt_max.archive_objectives = objs_max
    #
    #     # and/or Filter final archive by constraints on max
    #     last_nfe = rt_max.nfe[-1]
    #     # In this case, we are filtering by M.1000<30 and P.3490<30 on max trace
    #     pols_to_include = []
    #     i = 0
    #     for objs in objs_max[last_nfe]:
    #         if (objs[0] < 30) & (objs[4] < 30):
    #             pols_to_include.append(i)
    #         i += 1

    #     archive_objs = rt_max.archive_objectives[last_nfe]
    #     filtered_archive_objs = [archive_objs[i] for i in pols_to_include]
    #     rt_max.archive_objectives[last_nfe] = filtered_archive_objs
    #
    #     # Replace run dictionary item with updated dictionary for '4 C on max' item
    #     runtime_dict[name] = rt_max

######################### End conversion from average to max objectives #############################################


    # Get first feasible for each run
    # first_feasible = []
    # for run in runtime_dict.values():
    #     ff = run.get_first_feasible()
    #     first_feasible.append(ff)
    # print(first_feasible)

    # For reference point, use max of max_nadir for each run
    # Also calculate minimum of min_ideal for extreme point changes and obj_ranges for plotting
    max_list = []
    min_list = []
    obj_range_list = []
    for run in runtime_dict.values():
        run.compute_real_nadir()
        run.compute_real_ideal()
        temp_max = run.max_nadir
        max_list.append(temp_max)
        temp_min = run.min_ideal
        min_list.append(temp_min)
        obj_range = run.get_objective_ranges()
        obj_range_list.append(obj_range)

    hv = pygmo.hypervolume(max_list)
    refpoint = hv.refpoint()
    ideal_pt = pygmo.ideal(min_list)
    min_objs = []
    max_objs = []
    for i in range(0, n_objectives):
        o_ranges = [obj_r[i] for obj_r in obj_range_list]
        min_objs.append(min([x[0] for x in o_ranges]))
        max_objs.append(max([x[1] for x in o_ranges]))
    obj_ranges = list(zip(objective_names, min_objs, max_objs))

    # Create two dataframes, one for final archive objectives and one for runtime metrics
    metrics_df = pd.DataFrame()
    obj_df = pd.DataFrame()
    obj_names = [
        "P.3490", "P.WY.Release",
        "M.1000", "Avg.LB.Short"
        ]

    for name, run in runtime_dict.items():
        # Create dataframe with final archive objectives
        nfe = run.nfe[-1]  # final front
        #nfe = 1889  # partial run results
        #while nfe not in run.nfe:
        #    nfe += 1
        o_temp_df = pd.DataFrame(run.archive_objectives[nfe], columns=obj_names)
        o_temp_df['Run'] = name
        # if 'Coarse' in name:
        #     o_temp_df['Epsilons'] = 'Coarse'
        # else:
        #     o_temp_df['Epsilons'] = 'Fine'
        # if '26' in name:
        #     o_temp_df['Random Seed'] = '26'
        # else:
        #     o_temp_df['Random Seed'] = '8'
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
        ax.set_xlim(100, 3600)
        fig.show()

    # Plot final archive objectives for each run in parallel coordinates
    col_names = obj_names.copy()
    # col_names.append('Run')
    # col_names.append('Epsilons')
    # col_names.append('Random Seed')
    cols = col_names
    cols.reverse()
    color_col = 'M.1000'
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
    for name, low, high in obj_ranges:
        exp.parameters_definition[name].force_range(low, high)

    exp.to_html('Compare_allPolicies.html')

    # Perform non-dominated sorting
    # ndf, dl, dc, ndl = pygmo.fast_non_dominated_sorting(obj_df.loc[:, obj_df.columns != 'Run'])
    # nondom_df = obj_df.loc[ndf[0], :]

    # Perform epsilon non-dominated sorting. From code developed by J. Kasprzyk.
    # See linked notebook below for additional details and documentation
    # https://colab.research.google.com/drive/1fpMQoU4yZSrk71s-RUgAayKeNXEHpGeo?usp=sharing
    epsilons = [1, 50000, 1, 50000, 5, 100000, 100000, 100000]
    num_obj = 4
    problem = Problem(nvars=0, nobjs=num_obj, nconstrs=0)
    all_solutions_df = obj_df
    all_solutions_df['Eps Nd'] = False

    # create Platypus object to store the e-nondominated set (pt stands for platypus)
    eps_solutions_pt = EpsilonBoxArchive(epsilons)
    for index, row in obj_df.iterrows():
        # create solution object
        solution = Solution(problem)

        # save an id for which row of the original
        # dataframe this solution came from. really important
        # for cross-referencing things later!
        solution.id = index

        for j in range(num_obj):
            solution.objectives[j] = row[obj_names[j]]
        # calling the 'add' function on an EpsilonBoxArchive
        # orchestrates the archive update algorithm: it only
        # puts a solution into the archive if it's epsilon non-dominated
        # (and subsequently deletes solutions that end up being dominated!)
        eps_solutions_pt.add(solution)

    # save a list of the ids of the epsilon non-dominated solutions
    eps_ids = [sol.id for sol in eps_solutions_pt]

    # earlier, we initiated this flag to be False. We set it to True
    # if the solution's id matches the one in the list
    for id_num in eps_ids:
        all_solutions_df.at[id_num, "Eps Nd"] = True

    # create a new dataframe that only contains the rows that were
    # epsilon non-dominated
    eps_solutions_df = all_solutions_df[all_solutions_df["Eps Nd"]].copy(deep=True)

    # Save epsilon non-dominated set as csv (objective values)
    head_txt = obj_names + ['Run', 'Eps Nd']
    #eps_solutions_df.to_csv('C:/Users/elles/Documents/CU_Boulder/Research/Borg_processing_code/borgRW-parser/src/borg_parser/data/outputs/eps_nd_set_objs.csv',
    #                        header=head_txt, index=None, sep=' ')

    # Plot non-dominated parcoords plot
    col_names = obj_names.copy()
    # col_names.append('Run')
    # col_names.append('Epsilons')
    # col_names.append('Random Seed')
    cols = col_names
    cols.reverse()
    color_col = 'M.1000'
    exp = hip.Experiment.from_dataframe(eps_solutions_df)
    exp.parameters_definition[color_col].colormap = 'interpolateViridis'
    exp.parameters_definition['LF.Def'].type = hip.ValueType.NUMERIC  # LF Def sometimes detected as categorical
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        {'order': cols,
         'hide': ['uid', 'from_uid']}
    )
    exp.display_data(hip.Displays.TABLE).update(
        {'hide': ['uid', 'from_uid']}
    )
    #for name, low, high in obj_ranges:
    #    exp.parameters_definition[name].force_range(low, high)

    exp.to_html('Compare_nondominated.html')



if __name__ == '__main__':
        main()