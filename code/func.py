import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def isfloat(item):

    # A float is a float
    if isinstance(item, float):
        return True

    # Ints are okay
    if isinstance(item, int):
        return True

   # Detect leading white-spaces
    if len(item) != len(item.strip()):
        return False

    # Some strings can represent floats or ints ( i.e. a decimal )
    if isinstance(item, str):
        # regex matching
        int_pattern = re.compile("^[0-9]*$")
        float_pattern = re.compile("^[0-9]*.[0-9]*$")
        if float_pattern.match(item) or int_pattern.match(item):
            return True
        else:
            return False




def unique_pd(series, condition=None, sort_values = True, sort_index=False):
    ''' make Series of unique values and count amaunt of uniqe '''
    if condition is None:
        condition = series
    unq = series.groupby(condition).count()
    if sort_values is True:
        unq = unq.sort_values()
    if sort_index is True:
        unq = unq.sort_index()
    return unq


def find_and_replace_not_num_values(series, replace_to=0, inplace=False, astype=0, lops=False, list_values=False):
    '''find and replace values in series that are not numeric'''
    list_str_unique_values = unique_pd(series[~series.fillna('ND').str.isnumeric()]).index.to_list()
    list_for_comparison = []
    count = 0
    while len(list_str_unique_values) != list_for_comparison:
        list_for_comparison = len(list_str_unique_values)
        try:
            for i in range(0, len(list_str_unique_values)):
                # print(list_str_unique_values[i])
                if isfloat(list_str_unique_values[i]) is True:
                    list_str_unique_values.pop(i)
        except :'list index out of range'

        count += 1
    series.replace(list_str_unique_values, replace_to, inplace=inplace)
    if astype is True:
        series = series.astype('float64')
    if lops is True:
        print(f'lops is: {count}')
    if list_values is True:
        print(f'list string unique values is: {list_str_unique_values}')
    return series, list_str_unique_values
