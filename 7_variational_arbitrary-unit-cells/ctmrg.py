import torch
from torch.utils.checkpoint import checkpoint
from env import ENV

from .adlib import SVD 
svd = SVD.apply



def renormalize(*tensors):
    # T(up,left,down,right), u=up, l=left, d=down, r=right
    # C(d,r), EL(u,r,d), EU(l,d,r)

    C, E, T, chi = tensors

    dimT, dimE = T.shape[0], E.shape[0]
    D_new = min(dimE*dimT, chi)

    # step 1: contruct the density matrix Rho
    Rho = torch.tensordot(C,E,([1],[0]))        # C(ef)*EU(fga)=Rho(ega)
    Rho = torch.tensordot(Rho,E,([0],[0]))      # Rho(ega)*EL(ehc)=Rho(gahc)
    Rho = torch.tensordot(Rho,T,([0,2],[0,1]))  # Rho(gahc)*T(ghdb)=Rho(acdb)
    Rho = Rho.permute(0,3,1,2).contiguous().view(dimE*dimT, dimE*dimT)  # Rho(acdb)->Rho(ab;cd)

    Rho = Rho+Rho.t()
    Rho = Rho/Rho.norm()

    # step 2: Get Isometry P
    U, S, V = svd(Rho)
    truncation_error = S[D_new:].sum()/S.sum()
    P = U[:, :D_new] # projection operator
    
    #can also do symeig since Rho is symmetric 
    #S, U = symeig(Rho)
    #sorted, indices = torch.sort(S.abs(), descending=True)
    #truncation_error = sorted[D_new:].sum()/sorted.sum()
    #S = S[indices][:D_new]
    #P = U[:, indices][:, :D_new] # projection operator

    # step 3: renormalize C and E
    C = (P.t() @ Rho @ P) #C(D_new, D_new)

    ## EL(u,r,d)
    P = P.view(dimE,dimT,D_new)
    E = torch.tensordot(E, P, ([0],[0]))  # EL(def)P(dga)=E(efga)
    E = torch.tensordot(E, T, ([0,2],[1,0]))  # E(efga)T(gehb)=E(fahb)
    E = torch.tensordot(E, P, ([0,2],[0,1]))  # E(fahb)P(fhc)=E(abc)

    # step 4: symmetrize C and E
    C = 0.5*(C+C.t())
    E = 0.5*(E + E.permute(2, 1, 0))

    return C/C.norm(), E, S.abs()/S.abs().max(), truncation_error


def CTMRG(T, chi, max_iter, use_checkpoint=False):
    # T(up, left, down, right)

    threshold = 1E-12 if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold

    # C(down, right), E(up,right,down)
    C = T.sum((0,1))  #
    E = T.sum(1).permute(0,2,1)
    # C = torch.rand(chi, chi, dtype=T.dtype, device=T.device)
    # C = C + C.permute(1,0)
    # E = torch.rand(chi, T.shape[0], chi, dtype=T.dtype, device=T.device)
    # E = E + E.permute(2,1,0)

    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 1E1
    bvar  = 1E1
    bvar3 = 1E1
    for n in range(max_iter):
        tensors = C, E, T, torch.tensor(chi)
        if use_checkpoint: # use checkpoint to save memory
            C, E, s, error = checkpoint(renormalize, *tensors)
        else:
            C, E, s, error = renormalize(*tensors)

        Enorm = E.norm()
        E = E/Enorm
        truncation_error += error.item()
        if (s.numel() == sold.numel()):
            diff = (s-sold).norm().item()
            bvar = boundaryVariance(C, E)
            #bvar3 = boundaryVariance3(T, C, E)
            #print( s, sold )
            #print('ctmrg iteration %d to %.5e, bvar2 %.10e, bvar3 %.10e'%(n, diff, bvar, bvar3) )
        #print( 'n: %d, Enorm: %g, error: %e, diff: %e' % (n, Enorm, error.item(), diff) )
        #if (diff < threshold):
        if (bvar < threshold):
            break
        sold = s
    print('ctmrg converged at iterations %d to %.5e, bvar2 %.5e, bvar3 %.5e, \
        truncation error: %.5f'%(n, diff, bvar, bvar3, truncation_error))

    return C, E

def boundaryVariance(env, coord, dir, dbg = False):
    # C-- 1 -> 0
    # | 0
    # | 0
    # C-- 1 -> 1
    LB = torch.tensordot(C, C, ([0],[0])) # C(ab)C(ac)=LB(bc)

    # "Norm" of the <Left|Right>
    # C-- 0 1--C
    # |        | 0 -> 1
    # C-- 1 -> 0
    LBRB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=bnorm(bc)
    # C-------C
    # |       | 1
    # |       | 0
    # C--0 1--C
    LBRB = torch.tensordot(LBRB,C,([0,1],[1,0])) # LB(ab)C(ba)=bnorm()
    bnorm = LBRB.item()

    # apply transfer operator T <=> EE 
    #
    # C--0 0--E-- 2
    # |       | 1
    # |      
    # C--1 -> 0
    LB = torch.tensordot(LB,E,([0],[0])) # LB(ab)E(acd)=LB(bcd)
    # C-------E--2 -> 0
    # |       | 1
    # |       | 1  
    # C--0 0--E--2 -> 1
    LB = torch.tensordot(LB,E,([0,1],[0,1])) # LB(abc)E(abd)=LB(cd)

    # Evaluate the <Left|T|Right>
    # C--E--0 1--C
    # |          | 0 -> 1
    # |       
    # C--E--1 -> 0
    LBTRB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=LBTRB(bc)
    # C--E-------C
    # |          | 1
    # |          | 0
    # C--E--0 1--C
    LBTRB = torch.tensordot(LBTRB,C,([0,1],[1,0])) # LBTRB(ab)C(ba)=LBTRB()
    lbtrb = LBTRB.item()

    # apply transfer operator T <=> EE 
    #
    # C--E--0 0--E--2
    # |          | 1
    # |      
    # C--E--1 -> 0
    LB = torch.tensordot(LB,E,([0],[0])) # LB(ab)E(acd)=LB(bcd)
    # C--E-------E--2 -> 0
    # |          | 1
    # |          | 1
    # C--E--0 0--E--2 -> 1
    LB = torch.tensordot(LB,E,([0,1],[0,1])) # LB(abc)E(abd)=LB(cd)

    # Evaluate the <Left|TT|Right>
    # C--E--E--0 1--C
    # |             | 0 -> 1
    # |       
    # C--E--E--1 -> 0
    LB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=LB(bc)
    # C--E--E-------C
    # |             |1
    # |             |0
    # C--E--E--0 1--C
    LB = torch.tensordot(LB,C,([0,1],[1,0])) # LB(ab)C(ba)=LB()
    lbttrb = LB.item()

    if dbg:
        print('<L|R> = %.10e ; <L|TT|R>/<L|R> = %.10e ; <L|T|R>/<L|R> = %.10e'%(bnorm, lbttrb/bnorm, lbtrb/bnorm))
    return abs(lbttrb/bnorm) - (lbtrb/bnorm)*(lbtrb/bnorm)

def boundaryVariance3(A, C, E, dbg = False):
    # C-- 1 -> 0
    # | 0
    # | 0
    # E-- 1
    # | 2
    LB = torch.tensordot(C, E, ([0],[0])) # C(ab)E(acd)=LB(bcd)

    # C-- 0
    # E-- 1
    # | 2
    # | 0
    # C-- 1 -> 2
    LB = torch.tensordot(LB, C, ([2],[0])) # LB(abc)C(cd)=LB(bd)

    # "Norm" of the <Left|Right>
    # C-- 0 1--C
    # |        | 0 -> 2
    # E--1 -> 0
    # C--2 -> 1
    LBRB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LBRB(bcd)
    # C-----------C
    # |           |2
    # |           |0
    # E--0 1------E       
    # C--1->0     |2->1
    LBRB = torch.tensordot(LBRB,E,([0,2],[1,0])) # LBRB(abc)E(cad)=LBRB(bd)
    # C-----------C
    # E-----------E    
    # |           |1
    # |           |0
    # C--0 1------C
    LBRB = torch.tensordot(LBRB,C,([0,1],[1,0])) # LBRB(ab)C(ba)=LBRB()
    bnorm = LBRB.item()

    # apply transfer operator T <=> EAE 
    #
    # C--0 0--E--2->3
    # |       |1->2
    # E--1->0     
    # C--2->1
    LB = torch.tensordot(LB,E,([0],[0])) # LB(abc)E(ade)=LB(bcde)
    # C-------E--3->1
    # |       |2
    # |       |0
    # E--0 1--A--3
    # |       |2 
    # C--1->0
    LB = torch.tensordot(LB,A,([0,2],[1,0])) # LB(abcd)A(ceaf)=LB(bdef)
    # C-------E--1->0
    # E-------A--3->1
    # |       |2
    # |       |1 
    # C--0 0--E--2->2
    LB = torch.tensordot(LB,E,([0,2],[0,1])) # LB(abcd)E(ace)=LB(bde)

    # Evaluate the <Left|T|Right>
    # C--E--0 1--C
    # |  |       |0->2
    # E--A--1->0       
    # C--E--2->1
    LBTRB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LBTRB(bcd)
    # C--E-------C
    # |  |       |2
    # |  |       |0
    # E--A--0 1--E       
    # C--E--1->0 |2->1
    LBTRB = torch.tensordot(LBTRB,E,([0,2],[1,0])) # LBTRB(abc)E(cad)=LBTRB(bd)
    # C--E-------C
    # E--A-------E
    # |  |       |1
    # |  |       |0
    # C--E--0 1--C
    LBTRB = torch.tensordot(LBTRB,C,([0,1],[1,0])) # LBTRB(ab)C(ba)=LBTRB()
    lbtrb = LBTRB.item()

    # apply transfer operator T <=> EE 
    #
    # C--E--0 0--E--2->3
    # |  |       |1->2
    # E--A--1->0      
    # C--E--2->1
    LB = torch.tensordot(LB,E,([0],[0])) # LB(abc)E(ade)=LB(bcde)
    # C--E-------E--3->1
    # |  |       |2
    # |  |       |0
    # E--A--0 1--A--3
    # |  |       |2 
    # C--E--1->0
    LB = torch.tensordot(LB,A,([0,2],[1,0])) # LB(abcd)A(ceaf)=LB(bdef)
    # C--E-------E--1->0
    # E--A-------A--3->1
    # |  |       |2
    # |  |       |1   
    # C--E--0 0--E--2
    LB = torch.tensordot(LB,E,([0,2],[0,1])) # LB(abcd)E(ace)=LB(bde)

    # Evaluate the <Left|TT|Right>
    # C--E--E--0 1--C
    # |  |  |       |0->2
    # E--A--A--1->0       
    # C--E--E--2->1
    LB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LB(bcd)
    # C--E--E-------C
    # |  |  |       |2
    # |  |  |       |0
    # E--A--A--0 1--E       
    # C--E--E--1->0 |2->1
    LB = torch.tensordot(LB,E,([0,2],[1,0])) # LB(abc)E(cad)=LB(bd)
    # C--E--E-------C
    # E--A--A-------E
    # |  |  |       |1
    # |  |  |       |0
    # C--E--E--0 1--C
    LB = torch.tensordot(LB,C,([0,1],[1,0])) # LB(ab)C(ba)=LB()
    lbttrb = LB.item()

    if dbg:
        print('<L|R> = %.10e ; <L|TT|R>/<L|R> = %.10e ; <L|T|R>/<L|R> = %.10e'%(bnorm, lbttrb/bnorm, lbtrb/bnorm))
    return abs(lbttrb/bnorm) - (lbtrb/bnorm)*(lbtrb/bnorm)

def c2x2_LU():
# C--10--T--2
# 0      1

# C------T--2->1
# 0      1->0
# 0
# T--2->3
# 1->2

# C------T--1->0
# |      0
# |      0
# T--3 1 A--3 
# 2->1   2

# permute 0123->1203
# reshape (12)(03)->01

# return

# C2x2--1
# |
# 0

def c2x2_RU():
# 0--C
#    1
#    0
# 1--T
#    2   

# 2<-0--T--2 0--C
#    3<-1       |
#         0<-1--T
#            1<-2

# 1<-2--T-------C
#       3       |
#       0       |
# 2<-1--A--3 0--T
#    3<-2    0<-1

# permute 0123->1203
# reshape (12)(03)->01

# return 

# 0--C2x2
#    |
#    1

def c2x2_RD():
#    1<-0       0
# 2<-1--T--2 1--C

#         2<-0
#      3<-1--T
#            2
#    0<-1    0
# 1<-2--T----C


#    2<-0    1<-2
# 3<-1--A--3 3--T
#    2          |
#    0          |
# 0<-1--T-------C

# permute 0123->1203
# reshape (12)(03)->01

# return
#    0
#    |
# 1--C2x2

def c2x2_LD():
# 0->1
# T--2
# 1
# 0
# C--1->0

# 1->0
# T--2->1
# |
# |       0->2
# C--0 1--T--2->3

# 0       0->2
# T--1 1--A--3
# |       2
# |       2
# C-------T--3->1

# permute 0123->0213
# reshape (02)(13)

# return
# 0
# |
# C2x2--1