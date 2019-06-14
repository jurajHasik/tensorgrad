import sys
sys.path.insert(0, '../')

import torch
import time
from args import args
from ipeps import IPEPS

class ENV(torch.nn.Module):
    def __init__(self, args, ipeps, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(ENV, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        # initialize environment tensors
        self.C = dict()
        self.T = dict()

        # for each pair (coord, site) create corresponding T's and C's
        # y\x -1 0 1
        #  -1  C T C
        #   0  T A T
        #   1  C T C 
        # where the directional vectors are given as coord(env-tensor) - coord(A)
        # C(-1,-1)   T        (1,-1)C 
        #            |(0,-1)
        # T--(-1,0)--A(0,0)--(1,0)--T 
        #            |(0,1)
        # C(-1,1)    T         (1,1)C
        # and analogously for corners C
        #
        # The dimension-position convention is as follows: 
        # Start from index in direction "up" <=> (0,-1) and
        # continue anti-clockwise
        # 
        # C--1 0--T--2 0--C
        # |       |       |
        # 0       1       1
        # 0               0
        # |               |
        # T--2         1--T
        # |               |
        # 1               2
        # 0       0       0
        # |       |       |
        # C--1 1--T--2 1--C
        for coord, site in ipeps.sites.items():
            for vec in [(-1,0), (1,0), (0,-1), (0,1)]:
                #T[(coord,vec)]=torch.tensor()
                #self.T[(coord,vec)]="T"+str((coord,vec))
                self.T[(coord,vec)]="T"+str(ipeps.site(coord))
            for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                #T[(coord,vec)]=torch.tensor()
                #self.C[(coord,vec)]="C"+str((coord,vec))
                self.C[(coord,vec)]="C"+str(ipeps.site(coord))