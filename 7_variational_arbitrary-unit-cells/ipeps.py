import sys
sys.path.insert(0, '../')

import torch 
import time
from args import args

class IPEPS(torch.nn.Module):
    def __init__(self, args, sites, vertexToSite,
        dtype=torch.float64, device='cpu', use_checkpoint=False):
        
        super(IPEPS, self).__init__()
        self.use_checkpoint = use_checkpoint

        # Dict of non-equivalent on-site tensors 
        #
        # A B                     A B C
        # B A results in {A, B};  C A B results in {A,B,C}; etc
        #
        #                                             u s 
        #                                             |/ 
        # each site has indices in following order l--A--r  <=> A[s,u,l,d,r]
        #                                             |
        #                                             d
        # (anti-clockwise direction)
        self.sites = sites
        
        # A mapping function from coord on a square lattice
        # to one of the on-site tensor <key>
        #
        # y\x -2 -1 0 1 2 3  =>    1 2 0 1 2 0
        #  -2  A  B C A B C  =>  1 A B C A B C
        #  -1  B  C A B C A  =>  2 B C A B C A
        #   0  C  A B C A B  =>  0 C A B C A B
        #   1  A  B C A B C  =>  1 A B C A B C
        #   2  B  C A B C A  =>  2 B C A B C A
        #
        # given (x,y) pair on square lattice, apply appropriate
        # transformation and returns <key> of the appropriate 
        # tensor
        self.vertexToSite = vertexToSite

    # return site at coord=(x,y)
    def site(self, coord):
        return self.sites[self.vertexToSite(coord)]

