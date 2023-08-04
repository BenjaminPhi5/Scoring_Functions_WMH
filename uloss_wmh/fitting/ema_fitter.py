import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
TODO: make something that can do EMA weight fitting as justin recomended. Hopefully its just a case of modifying an 'on step end' type function or whatever.
need also to add code that allows me to keep a duplicate of the model paremters etc. Maybe I can just clone the entire model. Is gonna use 2x gpu memory.
"""