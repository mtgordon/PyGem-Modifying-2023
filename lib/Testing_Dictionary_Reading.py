# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:14:44 2020

@author: mgordon
"""

'''
Script: Testing_Dictionary_Reading.py
'''

import csv

dictionary_file = 'Testing_Dictionary.csv'

dictionary = csv.DictReader(open(dictionary_file))