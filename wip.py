#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:45:25 2023

@author: chris
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.joinpath("src")))

#%%

from neural_network.training import Trainer

trainer = Trainer()
trainer.train()