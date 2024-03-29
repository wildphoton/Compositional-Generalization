#!/usr/bin/env python

def merge_params(dictionary):
    """
    Function to merge a two-level parameter dictionary
    :param dictionary:
    :return:
    """
    res = {}
    for meta_key in dictionary:
        for key in dictionary[meta_key]:
            res[f'{meta_key}_{key}'] = dictionary[meta_key][key]

    return res