import torch
from torch.utils.checkpoint import checkpoint

from .adlib import RSVD 
svd = RSVD.apply
# from .adlib import ARNOLDISVD 
# svd = ARNOLDISVD.apply
#from .adlib import EigenSolver
#symeig = EigenSolver.apply

def renormalize(*tensors):
    # T(up,left,down,right), u=up, l=left, d=down, r=right
    # C(d,r), EL(u,r,d), EU(l,d,r)

    C, E, T, chi, tsvd_extra = tensors

    dimT, dimE = T.shape[0], E.shape[0]
    D_new = min(dimE*dimT, int(chi))

    # step 1: contruct the density matrix Rho
    Rho = torch.tensordot(C,E,([1],[0]))        # C(ef)*EU(fga)=Rho(ega)
    Rho = torch.tensordot(Rho,E,([0],[0]))      # Rho(ega)*EL(ehc)=Rho(gahc)
    Rho = torch.tensordot(Rho,T,([0,2],[0,1]))  # Rho(gahc)*T(ghdb)=Rho(acdb)
    Rho = Rho.permute(0,3,1,2).contiguous().view(dimE*dimT, dimE*dimT)  # Rho(acdb)->Rho(ab;cd)

    Rho = Rho+Rho.t()
    Rho = Rho/Rho.norm()

    # step 2: Get Isometry P
    U, S, V = svd(Rho,D_new+int(tsvd_extra))
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


def CTMRGTSVD(T, chi, tsvd_extra, max_iter, use_checkpoint=False):
    # T(up, left, down, right)

    threshold = 1E-8 if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold

    # C(down, right), E(up,right,down)
    C = T.sum((0,1))  #
    E = T.sum(1).permute(0,2,1)

    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 1E1
    bvar  = 1E1
    bvar3 = 1E1
    for n in range(max_iter):
        tensors = C, E, T, torch.tensor(chi), torch.tensor(tsvd_extra)
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
        #print( 'n: %d, Enorm: %g, error: %e, diff: %e' % (n, Enorm, error.item(), diff) )
        #if (diff < threshold):
        if (bvar < threshold):
            break
        sold = s
    print('ctmrg converged at iterations %d to %.5e, bvar2 %.5e, bvar3 %.5e, \
        truncation error: %.5f'%(n, diff, bvar, bvar3, truncation_error))

    return C, E


def boundaryVariance(C, E, dbg = False):
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

if __name__=='__main__':
    import time
    torch.manual_seed(42)
    D = 6
    chi = 80
    tsvd_extra = 20
    max_iter = 100
    device = 'cpu'

    # T(u,l,d,r)
    T = torch.randn(D, D, D, D, dtype=torch.float64, device=device, requires_grad=True)

    T = (T + T.permute(0, 3, 2, 1))/2.      # left-right symmetry
    T = (T + T.permute(2, 1, 0, 3))/2.      # up-down symmetry
    T = (T + T.permute(3, 2, 1, 0))/2.      # skew-diagonal symmetry
    T = (T + T.permute(1, 0, 3, 2))/2.      # digonal symmetry
    T = T/T.norm()

    C, E = CTMRGTSVD(T, chi, max_iter, tsvd_extra, use_checkpoint=True)
    C, E = CTMRGTSVD(T, chi, max_iter, tsvd_extra, use_checkpoint=False)
    print( 'diffC = ', torch.dist( C, C.t() ) )
    print( 'diffE = ', torch.dist( E, E.permute(2,1,0) ) )

