#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:38:47 2023

@author: chris
"""


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]