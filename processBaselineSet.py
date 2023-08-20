""" Script to take csv of Baseline set objective values,
find the decision variables and metrics from multiple archive.txt files,
combine into a single csv of Baseline policy data (assigning policy IDs),
and create parallel coordinate plots of objectives and metrics"""

import pandas as pd
import os
import hiplot as hip

# Set working directory
os.chdir('C:/Users/elles/Documents/CU_Boulder/Research/Borg_processing_code/borgRW-parser/src/borg_parser')

# Specify file paths for csv of Baseline objective values and for all archive files for looking up the remaining data
obj_csv_path = 'data/outputs/eps_nd_set_objs.csv'
# In this case, we performed 2 borg runs with different random seeds (8 and 26):
seed8_archive_path = 'data/outputs/archive_seed8.txt'
seed26_archive_path = 'data/outputs/archive_seed26.txt'

# import data as pandas (pd) data frames
base_objs = pd.read_csv(obj_csv_path, sep=' ')
archive_s8 = pd.read_table(seed8_archive_path, sep=' ')
archive_s26 = pd.read_table(seed26_archive_path, sep=' ')

# Set up data frame for collecting all Baseline set data
col_names = archive_s8.columns
baseline_df = pd.DataFrame(columns=col_names)

# Loop through each policy in the baseline objectives table to find metric & DVs in archive tables
for index, row in base_objs.iterrows():
    # Determine which archive data frame to use (which random seed run the policy came from)
    if row['Run'] == 'Seed 8 Min Mead':
        archive = archive_s8
    elif row['Run'] == 'Seed 26 Min Mead':
        archive = archive_s26
    else:
        print('Error: Run not found')

    # Each policy has a unique value for P.WY.Release, so we can match policies across data frames using that value
    p_wy_rel = row['P.WY.Release']
    pol_data = archive.loc[archive['Objectives.Powell_WY_Release'] == p_wy_rel, :]
    # pol_data_df = pd.DataFrame([pol_data])

    # Append row with policy data to the Baseline set data frame
    baseline_df = pd.concat([baseline_df, pol_data], ignore_index=True)

# save txt file of Baseline policy set data
baseline_txt_path = 'data/outputs/baseline_set_data.txt'
baseline_df.to_csv(baseline_txt_path, sep=' ')

# PRELIMINARY PLOTTING OF BASELINE POLICY SET

# subset data frame of 4 objectives + 4 metrics (8 performance metrics from Reclamation formulation)
# perf_metrics is list of column names from txt file of desired columns:
perf_metrics = ['Objectives.Powell_3490', 'Objectives.Powell_WY_Release', 'Objectives.Mead_1000',
                'Objectives.LB_Shortage_Volume', 'Metrics.Lee_Ferry_Deficit_Avg', 'Metrics.Avg_Combo_Storage_Avg',
                'Metrics.Max_Annual_LB_Shortage_Avg', 'Metrics.Max_Delta_Annual_Shortage_Avg']
base_perf_df = baseline_df[perf_metrics]
# drop index and make column names shorter for plotting
base_perf_df = base_perf_df.rename(columns={'Objectives.Powell_3490': 'P.3490', 'Objectives.Powell_WY_Release': 'P.WY.Release',
                             'Objectives.Mead_1000': 'M.1000', 'Objectives.LB_Shortage_Volume': 'LB.Short.Avg',
                             'Metrics.Lee_Ferry_Deficit_Avg': 'LF.Def',
                             'Metrics.Avg_Combo_Storage_Avg': 'Combo_Storage',
                             'Metrics.Max_Annual_LB_Shortage_Avg': 'Max.Annual.Short',
                             'Metrics.Max_Delta_Annual_Shortage_Avg': 'Max.Delta.Short'}, inplace=False)

# parallel coordinates plot of Baseline policies in 8-objective space
cols = ['P.3490', 'P.WY.Release', 'M.1000', 'LB.Short.Avg', 'LF.Def', 'Combo_Storage', 'Max.Annual.Short',
             'Max.Delta.Short']
#cols = col_names
#cols.reverse()
color_col = 'LB.Short.Avg'
exp = hip.Experiment.from_dataframe(base_perf_df)
exp.parameters_definition[color_col].colormap = 'interpolateTurbo'
exp.parameters_definition['LF.Def'].type = hip.ValueType.NUMERIC  # LF Def sometimes detected as categorical
exp.display_data(hip.Displays.PARALLEL_PLOT).update(
    {'order': cols,
     'hide': ['uid', 'from_uid']}
)
exp.display_data(hip.Displays.TABLE).update(
    {'hide': ['uid', 'from_uid']}
)
# for name, low, high in obj_ranges:
#    exp.parameters_definition[name].force_range(low, high)

exp.to_html('data/outputs/baseline_8obj.html')
