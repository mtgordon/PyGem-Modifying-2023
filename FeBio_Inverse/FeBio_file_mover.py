'''
This file is meant to be used to clean up the file management for Automate_Febio.
In the past, when stopping the automate process early, one would have to go into the folder where
the generated files were located, obtain the number that were generated, and remove that amount of variables from the
feb_variables.csv to ensure that the nest time Automate_Febio was started, it would pick up where it left off.

Now, all that is needed is to run this file if one needs to end the process early and pick it up from the same spot,
augmenting the feb_variables.csv accordingly. The process will also move the generated files to a new folder to
keep all of the different runs separate, since it is typically easier to combine old groups of runs instead of separate
them.

The final product of the process will be a new folder stemmed off of the provided file path, with all of the generated
files from that stopped run, a csv containing the variables extracted from the feb_variables.csv, and the original
feb_variables.csv having those recently run variables removed.
'''

import glob
import shutil
import os
import re
import pandas
from datetime import datetime

#The location of the generated files, in which they will be extracted from
target_path = "D:\\Gordon\\Automate FEB Runs\\2023_10_11 auto\\*" #'D:\\Gordon\\Target_test\\*'

#The location of the folder where the files will be moved, and the already_ran_feb_variables.csv
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2023_10_11 Completed'

date = datetime.today().strftime('D%Y%m%d_T%H%M%S')

final_folder = Results_Folder + '\\' + date
os.mkdir(final_folder)

#Get the current run_variables from the feb_variables.csv
current_vars = pandas.read_csv('feb_variables.csv')
changed_vars_header = []
for col in current_vars.columns:
    changed_vars_header.append(col)

already_ran_df = pandas.DataFrame(columns=changed_vars_header)

for feb_name in glob.glob(target_path):
    #Move the file to the new folder
    shutil.move(feb_name, final_folder + '\\' + os.path.basename(feb_name))

    #Extract the variables used (via the file name, uses the parentheses)
    if '.feb' in feb_name:
        param_list = re.findall('\(.*?\)',feb_name)
        # print(param_list)

        #Get all the changed variables (not equal to 1)
        var_list = [x for x in param_list if '(1)' not in x]
        # print(var_list)

        #Trim off the parentheses
        for i, var in enumerate(var_list):
            var = var.replace('(', '')
            var = var.replace(')', '')
            var = float(var)
            var_list[i] = var

        #Add to the dataframe
        already_ran_df.loc[len(already_ran_df)] = var_list

#Save the already ran variables
already_ran_df.to_csv(final_folder + '\\' + date + '_already_ran_variables.csv', index=False)

#Update the run variables csv to not have the variables that were just ran
updated_vars = pandas.concat([already_ran_df, current_vars], ignore_index=True)
updated_vars.drop_duplicates(keep=False, inplace=True)
print(updated_vars)
updated_vars.to_csv('feb_variables.csv', index=False)
