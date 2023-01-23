# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:34:32 2020

@author: mgordon
"""

import csv
# from random import random
# import numpy
import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
import random

number_of_runs = 1000

hist_data = 0
norm_data = 1


def rand_correlated_uniform(num_samples, r):
    method = 'eigenvectors'
    
    # num_samples = 400
    
    # # The desired covariance matrix.
    # r = np.array([
    #         [  3.40, -2.75, -2.00],
    #         [ -2.75,  5.50,  1.50],
    #         [ -2.00,  1.50,  1.25]
    #     ])
    
    # r = np.array([
    #         [  1, 0.7, -0.7],
    #         [ 0.7,  1,  -0.7],
    #         [ -0.7,  -0.7,  1]
    #     ])
    
    # def random_
    
    num_params = len(r[:])
    print('@@@@@@@@@@@@@@@@@@@@@@', num_params)
    
    # Generate samples from three independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x = norm.rvs(size=((num_params), num_samples))
    print(x)
    
    # reassign each value to be uniform distribution
    for i in range(0,num_samples):
        for j in range(0,num_params):
            x[j][i]= random.uniform(0,1)
            
    print(x)
            
    
    
    # We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
    # the Cholesky decomposition, or the we can construct `c` from the
    # eigenvectors and eigenvalues.
    
    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    else:
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = eigh(r)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
    
    # Convert the data to correlated random variables. 
    y = np.dot(c, x)
    
    print('before normalizing:', y)
    # print(y[:][0])
    
    # I believe this is normalizing each value between the min and max...
    # ...but that guarantees a min and max which seems incorrect
    for i in range(0,num_params):
        # print('starting prints')
        # print(i)
        # print(y[:][i])
        # print(min(y[:][i]))
        # print(max(y[:][i]))
        y[:][i] = (y[:][i]-min(y[:][i]))/(max(y[:][i])-min(y[:][i]))
        # print(min(y[:][i]))
        # print(max(y[:][i]))
        
        
    # print(y)
    # print('the matrix is:', np.transpose(y))
    
    return(np.transpose(y))
    

def rand_correlated(r):
    method = 'eigenvectors'

    
    num_params = len(r[:])

    
    # Generate samples from three independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x = norm.rvs(size=((num_params), 1))
    print(x)
    
       
    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    else:
        # Compute the eigenvalues and eigenvectors.
        print('r:', r)
        evals, evecs = eigh(r)
        print('evals:', evals)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
        print('c:', c)
    
    # Convert the data to correlated random variables. 
    y = np.dot(c, x)
      
    print('Random Numbers:', np.transpose(y))
    return(np.transpose(y)[0])



def get_hist_values(hist_props, random_values):
    prop_values = []
    

    for prop_index, prop in enumerate(hist_props):
        print(prop)
        file = prop + '_Histogram.csv'
        values = []
        counts = []
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                values.append(row['Value'])
                counts.append(float(row['Count']))
        
        total_count = 0
        for count in counts:
            total_count += count
        
        percents = []
        current_count = 0
        for count in counts:
            current_count += count
            percents.append(current_count/total_count)
            
        random_number = random_values[prop_index]
        print(random_number)
        # print(percents)
        for index, value in enumerate(values):
            # print(value)
            if random_number <= percents[index]:
                prop_values.append(value)
                break
    return(prop_values)



material = 'CL'


PC1_mean = 0
PC2_mean = 0
PC1_SD = 11
PC2_SD = 11
Correlation_Between_Hiatus_and_PC1 = .7
Correlation_Between_Hiatus_and_PC2 = .7
Correlation_beteween_PC1_and_PC2 = .7

hist_props = ['Hiatus_Size','AVW_Length','Apical_Shift', 'CL_Strain', 'US_Strain', 'CL_Material', 'US_Material']
# hist_props = ['Hiatus_Size','AVW_Length','Apical_Shift', 'CL_Strain', 'US_Strain', 'CL_Material', 'US_Material', 'Levator_Plate_PC1', 'Levator_Plate_PC2']

prop_means = [35, 1, 0, 0, 0, 200, 200, PC1_mean]
r = [[7**2,0,-0.5*7*3,0,0,0,0,Correlation_Between_Hiatus_and_PC1*PC1_SD*7],
    [0,.18**2,0,0,0,0,0,0],
    [-0.5*7*3,0,3**2,0,0,0,0,0],
    [0,0,0,.04**2,0.00064,0,0,0],
    [0,0,0,.00064,.02**2,0,0,0],
    [0,0,0,0,0,100**2,0,0],
    [0,0,0,0,0,0,100**2,0],
    [Correlation_Between_Hiatus_and_PC1*PC1_SD*7,0,0,0,0,0,0,PC1_SD**2]]

prop_means = [35, 1, 0, 0, 0, 200, 200, PC1_mean, PC2_mean]
r = [[7**2,0,-0.5*7*3,0,0,0,0,Correlation_Between_Hiatus_and_PC1*PC1_SD*7, Correlation_Between_Hiatus_and_PC2*PC2_SD*7],
    [0,.18**2,0,0,0,0,0,0,0],
    [-0.5*7*3,0,3**2,0,0,0,0,0,0],
    [0,0,0,.04**2,0.00064,0,0,0,0],
    [0,0,0,.00064,.02**2,0,0,0,0],
    [0,0,0,0,0,100**2,0,0,0],
    [0,0,0,0,0,0,100**2,0,0],
    [Correlation_Between_Hiatus_and_PC1*PC1_SD*7,0,0,0,0,0,0,PC1_SD**2,Correlation_beteween_PC1_and_PC2*PC1_SD*PC2_SD],
    [Correlation_Between_Hiatus_and_PC2*PC2_SD*7,0,0,0,0,0,0,Correlation_beteween_PC1_and_PC2*PC1_SD*PC2_SD,PC2_SD**2]]

# r = [[7**2,0,-0.5*7*3,0,0,0,0,Correlation_Between_Hiatus_and_PC1*PC1_SD*7,0],
#     [0,.18**2,0,0,0,0,0,0,0],
#     [-0.5*7*3,0,3**2,0,0,0,0,0,0],
#     [0,0,0,.04**2,0.00064,0,0,0,0],
#     [0,0,0,.00064,.02**2,0,0,0,0],
#     [0,0,0,0,0,100**2,0,0,0],
#     [0,0,0,0,0,0,100**2,0,0],
#     [Correlation_Between_Hiatus_and_PC1*PC1_SD*7,0,0,0,0,0,0,PC1_SD**2,0],
#     [0,0,0,0,0,0,0,0,100**2]]

# prop_means = [35, 1, 0, 0, 0, 200, 200, PC1_mean, PC2_mean]
# r = [[7**2,1.1,-0.5*7*3,0,0,0,0, 4, 4],
#      # [[7**2,1.1,-0.5*7*3,0,0,0,0, Correlation_Between_Hiatus_and_PC1*PC1_SD*7, Correlation_Between_Hiatus_and_PC2*PC2_SD*7],
#     [1.1,.18**2,0,0,0,0,0,0,0],
#     [-0.5*7*3,0,3**2,0,0,0,0,0,0],
#     [0,0,0,.04**2,0.00064,0,0,0,0],
#     [0,0,0,.00064,.02**2,0,0,0,0],
#     [0,0,0,0,0,100**2,0,0,0],
#     [0,0,0,0,0,0,100**2,0,0],
#     # [0,0,0,0,0,0,0,100**2,0],
#     # [0,0,0,0,0,0,0,0,100**2]
#         [4,0,0,0,0,0,0,PC1_SD**2,4],
#     [ 4,0,0,0,0,0,0, 4, PC2_SD**2]
#     # [Correlation_Between_Hiatus_and_PC1*PC1_SD*7,0,0,0,0,0,0,PC1_SD**2,Correlation_beteween_PC1_and_PC2*PC1_SD*PC2_SD],
#     # [ Correlation_Between_Hiatus_and_PC2*PC2_SD*7  ,0,0,0,0,0,0, Correlation_beteween_PC1_and_PC2*PC1_SD*PC2_SD  , PC2_SD**2]
# ]


# prop_means = [35, 1, 0, 0, 0, 200, 200]
# prop_means = [35, 1, 0, 0, 0, 200, 200, PC1_mean]
# prop_means = [35, 1, 0, 0, 0, 200, 200, PC1_mean, PC2_mean]

# random_values = np.random.multivariate_normal([43.4,0.5,0.5,0.5,0.5,0.5,0.5], r)




with open('Run_Variables.csv', 'w', newline='') as output_file:
    header = []
    for prop in hist_props:
            header.append(prop)
    writer = csv.writer(output_file)
    writer.writerow(header)

    if hist_data == 1:
        random_run_values = rand_correlated_uniform(number_of_runs, r)
        print('*************')
        print(random_run_values)


# Look through to write a row for each run
    for i in range(0,number_of_runs):
        if hist_data == 1:
            # rand_correlated_uniform(num_samples, r)
            prop_values = get_hist_values(hist_props, random_run_values[i])
            
        elif norm_data == 1:
            prop_values = rand_correlated(r) + prop_means
        else:
            print('ERROR: NO DATA TYPE INPUT')
            
        print('PROP VALUES:', prop_values)

#  Here is wher eyou can set CL_Strain equal to US_Strain
        # prop_values[1] = prop_values[2]
        writer.writerow(prop_values)