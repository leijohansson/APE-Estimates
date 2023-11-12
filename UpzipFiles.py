# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:39:19 2023

@author: Linne
"""


import os 
import zipfile


for year in range(1950, 2024):
    with zipfile.ZipFile(f'ZipData/EN.4.2.2.analyses.g10.{year}.zip', 'r') as zip_ref:
        zip_ref.extractall('./Data')