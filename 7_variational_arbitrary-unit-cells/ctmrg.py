import torch
from torch.utils.checkpoint import checkpoint
from ipeps import IPEPS
from env import ENV

from .adlib import SVD 
svd = SVD.apply

def truncated_svd(M, chi, abs_tol=None, rel_tol=None):
    """
    Performs a truncated SVD on a matrix M.     
    M ~ (Ut)(St)(Vt)^{T}

    
    inputs:
        M (torch.Tensor):
            tensor of shape (dim0, dim1)

        chi (int):
            maximum allowed dimension of S

        abs_tol (float):
            absolute tollerance on singular eigenvalues

        rel_tol (float):
            relative tollerance on singular eigenvalues

    where S is diagonal matrix of of shape (dimS, dimS)
    and dimS <= chi

    returns Ut, St, Vt
    """
    U, S, V = svd(M)
    St = S[:chi]
    if abs_tol is not None: St = St[S > abs_tol]
    if rel_tol is not None: St = St[S/torch.mean(S) > rel_tol]
    Ut = Ut[:, :St.shape[0]]
    Vt = Vt[:, :St.shape[0]]
    return Ut, St, Vt



def CTMRG(ipeps, env):
    # TODO 0) 
    # x) Create double-layer (DL) tensors, preserving the same convenction
    # for order of indices 
    #
    #     /           /
    #  --A^dag-- = --a--
    #   /|          /
    #    |/
    #  --A--
    #   /
    #
    sitesDL=dict()
    for coord,A in ipeps.sites.items():
        dimsA = A.size()
        a = torch.einsum('mefgh,mabcd->eafbgchd',(A,A)).contiguous()
            .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2)
        sitesDL[coord]=a
    ipepsDL = IPEPS(None,sitesDL,ipeps.vertexToSite)

    # x) Initialize env tensors C,T
    env = init_random(env)

    # 1) 
    for i in range(50):
        ctm_MOVE_UP(ipepsDL, env)
        ctm_MOVE_LEFT(ipepsDL, env)
        ctm_MOVE_RIGHT(ipepsDL, env)
        ctm_MOVE_DOWN(ipepsDL, env)
    
        #if ctm_converged():
        #    break

    return env

# def boundaryVariance(env, coord, dir, dbg = False):
#     # C-- 1 -> 0
#     # | 0
#     # | 0
#     # C-- 1 -> 1
#     LB = torch.tensordot(C, C, ([0],[0])) # C(ab)C(ac)=LB(bc)

#     # "Norm" of the <Left|Right>
#     # C-- 0 1--C
#     # |        | 0 -> 1
#     # C-- 1 -> 0
#     LBRB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=bnorm(bc)
#     # C-------C
#     # |       | 1
#     # |       | 0
#     # C--0 1--C
#     LBRB = torch.tensordot(LBRB,C,([0,1],[1,0])) # LB(ab)C(ba)=bnorm()
#     bnorm = LBRB.item()

#     # apply transfer operator T <=> EE 
#     #
#     # C--0 0--E-- 2
#     # |       | 1
#     # |      
#     # C--1 -> 0
#     LB = torch.tensordot(LB,E,([0],[0])) # LB(ab)E(acd)=LB(bcd)
#     # C-------E--2 -> 0
#     # |       | 1
#     # |       | 1  
#     # C--0 0--E--2 -> 1
#     LB = torch.tensordot(LB,E,([0,1],[0,1])) # LB(abc)E(abd)=LB(cd)

#     # Evaluate the <Left|T|Right>
#     # C--E--0 1--C
#     # |          | 0 -> 1
#     # |       
#     # C--E--1 -> 0
#     LBTRB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=LBTRB(bc)
#     # C--E-------C
#     # |          | 1
#     # |          | 0
#     # C--E--0 1--C
#     LBTRB = torch.tensordot(LBTRB,C,([0,1],[1,0])) # LBTRB(ab)C(ba)=LBTRB()
#     lbtrb = LBTRB.item()

#     # apply transfer operator T <=> EE 
#     #
#     # C--E--0 0--E--2
#     # |          | 1
#     # |      
#     # C--E--1 -> 0
#     LB = torch.tensordot(LB,E,([0],[0])) # LB(ab)E(acd)=LB(bcd)
#     # C--E-------E--2 -> 0
#     # |          | 1
#     # |          | 1
#     # C--E--0 0--E--2 -> 1
#     LB = torch.tensordot(LB,E,([0,1],[0,1])) # LB(abc)E(abd)=LB(cd)

#     # Evaluate the <Left|TT|Right>
#     # C--E--E--0 1--C
#     # |             | 0 -> 1
#     # |       
#     # C--E--E--1 -> 0
#     LB = torch.tensordot(LB,C,([0],[1])) # LB(ab)C(ca)=LB(bc)
#     # C--E--E-------C
#     # |             |1
#     # |             |0
#     # C--E--E--0 1--C
#     LB = torch.tensordot(LB,C,([0,1],[1,0])) # LB(ab)C(ba)=LB()
#     lbttrb = LB.item()

#     if dbg:
#         print('<L|R> = %.10e ; <L|TT|R>/<L|R> = %.10e ; <L|T|R>/<L|R> = %.10e'%(bnorm, lbttrb/bnorm, lbtrb/bnorm))
#     return abs(lbttrb/bnorm) - (lbtrb/bnorm)*(lbtrb/bnorm)

# def boundaryVariance3(A, C, E, dbg = False):
#     # C-- 1 -> 0
#     # | 0
#     # | 0
#     # E-- 1
#     # | 2
#     LB = torch.tensordot(C, E, ([0],[0])) # C(ab)E(acd)=LB(bcd)

#     # C-- 0
#     # E-- 1
#     # | 2
#     # | 0
#     # C-- 1 -> 2
#     LB = torch.tensordot(LB, C, ([2],[0])) # LB(abc)C(cd)=LB(bd)

#     # "Norm" of the <Left|Right>
#     # C-- 0 1--C
#     # |        | 0 -> 2
#     # E--1 -> 0
#     # C--2 -> 1
#     LBRB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LBRB(bcd)
#     # C-----------C
#     # |           |2
#     # |           |0
#     # E--0 1------E       
#     # C--1->0     |2->1
#     LBRB = torch.tensordot(LBRB,E,([0,2],[1,0])) # LBRB(abc)E(cad)=LBRB(bd)
#     # C-----------C
#     # E-----------E    
#     # |           |1
#     # |           |0
#     # C--0 1------C
#     LBRB = torch.tensordot(LBRB,C,([0,1],[1,0])) # LBRB(ab)C(ba)=LBRB()
#     bnorm = LBRB.item()

#     # apply transfer operator T <=> EAE 
#     #
#     # C--0 0--E--2->3
#     # |       |1->2
#     # E--1->0     
#     # C--2->1
#     LB = torch.tensordot(LB,E,([0],[0])) # LB(abc)E(ade)=LB(bcde)
#     # C-------E--3->1
#     # |       |2
#     # |       |0
#     # E--0 1--A--3
#     # |       |2 
#     # C--1->0
#     LB = torch.tensordot(LB,A,([0,2],[1,0])) # LB(abcd)A(ceaf)=LB(bdef)
#     # C-------E--1->0
#     # E-------A--3->1
#     # |       |2
#     # |       |1 
#     # C--0 0--E--2->2
#     LB = torch.tensordot(LB,E,([0,2],[0,1])) # LB(abcd)E(ace)=LB(bde)

#     # Evaluate the <Left|T|Right>
#     # C--E--0 1--C
#     # |  |       |0->2
#     # E--A--1->0       
#     # C--E--2->1
#     LBTRB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LBTRB(bcd)
#     # C--E-------C
#     # |  |       |2
#     # |  |       |0
#     # E--A--0 1--E       
#     # C--E--1->0 |2->1
#     LBTRB = torch.tensordot(LBTRB,E,([0,2],[1,0])) # LBTRB(abc)E(cad)=LBTRB(bd)
#     # C--E-------C
#     # E--A-------E
#     # |  |       |1
#     # |  |       |0
#     # C--E--0 1--C
#     LBTRB = torch.tensordot(LBTRB,C,([0,1],[1,0])) # LBTRB(ab)C(ba)=LBTRB()
#     lbtrb = LBTRB.item()

#     # apply transfer operator T <=> EE 
#     #
#     # C--E--0 0--E--2->3
#     # |  |       |1->2
#     # E--A--1->0      
#     # C--E--2->1
#     LB = torch.tensordot(LB,E,([0],[0])) # LB(abc)E(ade)=LB(bcde)
#     # C--E-------E--3->1
#     # |  |       |2
#     # |  |       |0
#     # E--A--0 1--A--3
#     # |  |       |2 
#     # C--E--1->0
#     LB = torch.tensordot(LB,A,([0,2],[1,0])) # LB(abcd)A(ceaf)=LB(bdef)
#     # C--E-------E--1->0
#     # E--A-------A--3->1
#     # |  |       |2
#     # |  |       |1   
#     # C--E--0 0--E--2
#     LB = torch.tensordot(LB,E,([0,2],[0,1])) # LB(abcd)E(ace)=LB(bde)

#     # Evaluate the <Left|TT|Right>
#     # C--E--E--0 1--C
#     # |  |  |       |0->2
#     # E--A--A--1->0       
#     # C--E--E--2->1
#     LB = torch.tensordot(LB,C,([0],[1])) # LB(abc)C(da)=LB(bcd)
#     # C--E--E-------C
#     # |  |  |       |2
#     # |  |  |       |0
#     # E--A--A--0 1--E       
#     # C--E--E--1->0 |2->1
#     LB = torch.tensordot(LB,E,([0,2],[1,0])) # LB(abc)E(cad)=LB(bd)
#     # C--E--E-------C
#     # E--A--A-------E
#     # |  |  |       |1
#     # |  |  |       |0
#     # C--E--E--0 1--C
#     LB = torch.tensordot(LB,C,([0,1],[1,0])) # LB(ab)C(ba)=LB()
#     lbttrb = LB.item()

#     if dbg:
#         print('<L|R> = %.10e ; <L|TT|R>/<L|R> = %.10e ; <L|T|R>/<L|R> = %.10e'%(bnorm, lbttrb/bnorm, lbtrb/bnorm))
#     return abs(lbttrb/bnorm) - (lbtrb/bnorm)*(lbtrb/bnorm)

def ctm_MOVE_UP(ipeps, env):
    # Loop over all non-equivalent sites of ipeps
    # and compute projectors P(coord), P^tilde(coord)
    for coord,site in ipeps.items():
        # TODO compute isometries
        P[(coord)], Pt[coord] = result_of_compute_isometry

    # Loop over all non-equivalent sites of ipeps
    # and perform absorption and truncation
    for coord,site in ipeps.items():
        absorb_truncate_CTM_MOVE_UP(coord, ipeps, env, P, Pt)

def ctm_MOVE_LEFT(ipeps, env):
    # Loop over all non-equivalent sites of ipeps
    # and compute projectors P(coord), P^tilde(coord)
    for coord,site in ipeps.items():
        # TODO compute isometries
        P[(coord)], Pt[coord] = result_of_compute_isometry

    # Loop over all non-equivalent sites of ipeps
    # and perform absorption and truncation
    for coord,site in ipeps.items():
        absorb_truncate_CTM_MOVE_LEFT(coord, ipeps, env, P, Pt)

def ctm_MOVE_DOWN(ipeps, env):
    # Loop over all non-equivalent sites of ipeps
    # and compute projectors P(coord), P^tilde(coord)
    for coord,site in ipeps.items():
        # TODO compute isometries
        P[(coord)], Pt[coord] = result_of_compute_isometry

    # Loop over all non-equivalent sites of ipeps
    # and perform absorption and truncation
    for coord,site in ipeps.items():
        absorb_truncate_CTM_MOVE_DOWN(coord, ipeps, env, P, Pt)

def ctm_MOVE_RIGHT(ipeps, env):
    # Loop over all non-equivalent sites of ipeps
    # and compute projectors P(coord), P^tilde(coord)
    for coord,site in ipeps.items():
        # TODO compute isometries
        P[(coord)], Pt[coord] = result_of_compute_isometry

    # Loop over all non-equivalent sites of ipeps
    # and perform absorption and truncation
    for coord,site in ipeps.items():
        absorb_truncate_CTM_MOVE_RIGHT(coord, ipeps, env, P, Pt)

def ctm_get_porjectors(R, Rt, chi, use_QR = False, tol = 1e-10):
    """
    Given the two tensor T and Tt (T tilde) this computes the projectors
    Computes The projectors (P, P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf)
    The inices of the input R, Rt are

        R (torch.Tensor):
            tensor of shape (dim0, dim1, dim2, dim3)
        Rt (torch.Tensor):
            tensor of shape (dim0, dim1, dim2, dim3)
        chi (int):
            auxiliary bond dimension  

    --------------------
    |        T         |
    --------------------
      |    |    |    |
     dim0 dim1 dim2 dim3
      |    |    |    |
    ---------  
     \\ P //   
     -------
        |
       chi
        |


        |
       chi
        |
     -------   
    // Pt  \\
    ---------
      |    |    |    |    
     dim0 dim1 dim2 dim3
      |    |    |    |    
    --------------------
    |        Rt        |
    --------------------
    """
    assert R.shape == Rt.shape
    assert len(R.shape) == 4
    #
    dim0, dim1, dim2, dim4 = R.shape
    R = R.view(dim0 * dim1, dim2, dim3)
    R = R.view(dim0 * dim1, dim2 * dim3)
    Rt = Rt.view(dim0 * dim1, dim2, dim3)
    Rt = Rt.view(dim0 * dim1, dim2 *  dim3)

    # QR decomposition (I do not understand why this is usefull)
    if use_QR:
        Q_qr, R_qr = torch.qr(R)
        Qt_qr, Rt_qr = torch.qr(Rt)
        R = R_qr
        Rt = Rt_qr

    #  SVD decomposition
    M = torch.mm(R.transpose(1, 0), Rt)
    U, S, V = truncated_svd(M, chi, tol) # M = USV^{T}
    S_sqrt = 1 / torch.sqrt(S)

    # 
    P = torch.mm(S_sqrt, torch.mm(U.transpose(1, 0),R.transpose(1, 0)))
    Pt = torch.mm(S_sqrt, torch.mm(V.transpose(1, 0),Rt.transpose(1, 0)))

    return P, Pt
#####################################################################
# functions performing absorption and truncation step
#####################################################################
def absorb_truncate_CTM_MOVE_UP(coord, ipeps, env, P, Pt):
    C1 = env.C[(coord,(1,-1))]
    T1 = env.T[(coord,(1,0))]
    T = env.T[(coord,(0,-1))]
    T2 = env.T[(coord,(-1,0))]
    C2 = env.C[(coord,(-1,-1))]
    A = ipeps.site(coord)

    # 0--C1
    #    1
    #    0
    # 1--T1
    #    2 
    nC1 = torch.tensordot(C1,T1,([1],[0]))

    #        --0 0--C1
    #       |       |
    # 0<-2--Pt      |
    #       |       | 
    #        --1 1--T1
    #               2->0
    C1 = torch.tensordot(Pt, nC1,([0,1],[0,1]))

    # C2--1->0
    # 0
    # 0
    # T2--2
    # 1
    nC2 = torch.tensordot(C2, T2,([0],[0])) 

    # C2--0 0--
    # |        |        
    # |        P--2->1
    # |        |
    # T2--2 1--
    # 1->0
    C2 = torch.tensordot(P, nC2,([0,2],[0,1]))

    #        --0 0--T--2->3
    #       |       1->2
    # 1<-2--Pt
    #       |
    #        --1->0 
    nT = torch.tensordot(Pt, T, ([0],[0]))

    #        -------T--3->1
    #       |       2
    # 0<-1--Pt      | 
    #       |       0
    #        --0 1--A--3
    #               2 
    nT = torch.tensordot(nT, A,([0,2],[1,0]))

    #     -------T--1 0--
    #    |       |       |
    # 0--Pt      |       P--2
    #    |       |       |
    #     -------A--3 1--
    #            2->1 
    T = torch.tensordot(nT, P,([1,3],[0,1]))
    T = T.contiguous()

def absorb_truncate_CTM_MOVE_LEFT(coord, ipeps, env, P, Pt):
    C1 = env.C[(coord,(-1,-1))]
    T1 = env.T[(coord,(0,-1))]
    T = env.T[(coord,(-1,0))]
    T2 = env.T[(coord,(0,1))]
    C2 = env.C[(coord,(-1,1))]
    A = ipeps.site(coord)

    # C1--1 0--T1--2
    # |        |
    # 0        1
    nC1 = torch.tensordot(C1,T1,([1],[0]))

    # C1--1 0--T1--2->1
    # |        |
    # 0        1
    # 0        1
    # |___Pt___|
    #     2->0
    C1 = torch.tensordot(Pt, nC1,([0,1],[0,1]))

    # 0        0->1
    # C2--1 1--T2--2
    nC2 = torch.tensordot(C2, T2,([1],[1])) 

    #    2->0
    # ___P____
    # 0      1
    # 0      1  
    # C2-----T2--2->1
    C2 = torch.tensordot(P, nC2,([0,1],[0,1]))

    #    2->1
    # ___P___
    # 0     1->0
    # 0
    # T--2->3
    # 1->2
    nT = torch.tensordot(P, T,([0],[0]))

    #    1->0
    # ___P_____
    # |       0
    # |       0
    # T--3 1--A--3
    # 2->1    2
    nT = torch.tensordot(nT, A,([0,3],[0,1]))

    #    0
    # ___P_____
    # |       |
    # |       |
    # T-------A--3->1
    # 1       2
    # 0       1
    # |___Pt__|
    #     2
    T = torch.tensordot(nT, Pt,([1,2],[0,1]))
    T = T.permute(0,2,1).contiguous()

def absorb_truncate_CTM_MOVE_DOWN(coord, ipeps, env, P, Pt):
    C1 = env.C[(coord,(-1,1))]
    T1 = env.T[(coord,(-1,0))]
    T = env.T[(coord,(0,1))]
    T2 = env.T[(coord,(1,0))]
    C2 = env.C[(coord,(1,1))]
    A = ipeps.site(coord)

    # 0->1
    # T1--2->2
    # 1
    # 0
    # C1--1->0
    nC1 = torch.tensordot(C1,T1,([0],[1]))

    # 1->0
    # T1--2 0--
    # |        |        
    # |        Pt--2->1
    # |        |
    # C1--0 1--   
    C1 = torch.tensordot(nC1, Pt, ([0,2],[1,0]))

    #    1<-0
    # 2<-1--T2
    #       2
    #       0
    # 0<-1--C2
    nC2 = torch.tensordot(C2, T2,([0],[2])) 

    #            0<-1
    #        --1 2--T2
    #       |       |
    # 1<-2--P       |
    #       |       | 
    #        --0 0--C2
    C2 = torch.tensordot(nC2, P, ([0,2],[0,1]))

    #        --1->0
    #       |
    # 1<-2--P
    #       |       0->2
    #        --0 1--T--2->3 
    nT = torch.tensordot(P, T, ([0],[1]))

    #               0->2
    #        --0 1--A--3 
    #       |       2 
    # 0<-1--P       |
    #       |       2
    #        -------T--3->1
    nT = torch.tensordot(nT, A,([0,2],[1,2]))

    #               2->1
    #        -------A--3 1--
    #       |       |       |
    #    0--P       |       Pt--2
    #       |       |       |
    #        -------T--1 0--
    T = torch.tensordot(nT, Pt,([1,3],[0,1]))
    T = T.permute(1,0,2).contiguous()

def absorb_truncate_CTM_MOVE_RIGHT(coord, ipeps, env, P, Pt):
    C1 = env.C[(coord,(1,1))]
    T1 = env.T[(coord,(0,1))]
    T = env.T[(coord,(1,0))]
    T2 = env.T[(coord,(0,-1))]
    C2 = env.C[(coord,(1,-1))]
    A = ipeps.site(coord)

    #       0->1     0
    # 2<-1--T1--2 1--C1
    nC1 = torch.tensordot(C1, T1,([1],[2])) 

    #          2->0
    #        __Pt__
    #       1     0
    #       1     0
    # 1<-2--T1----C1
    C1 = torch.tensordot(Pt, nC1,([0,1],[0,1]))

    # 1<-0--T2--2 0--C2
    #    2<-1     0<-1
    nC2 = torch.tensordot(C2,T2,([0],[2]))

    # 0<-1--T2----C2
    #       2     0
    #       1     0
    #       |__P__|
    #          2->1
    C2 = torch.tensordot(nC2, P,([0,2],[0,1]))

    #    1<-2
    #    ___Pt___
    # 0<-1      0
    #           0
    #     2<-1--T
    #        3<-2
    nT = torch.tensordot(Pt, T,([0],[0]))

    #       0<-1 
    #       ___Pt___
    #       0       |
    #       0       |
    # 2<-1--A--3 2--T
    #    3<-2    1<-3
    nT = torch.tensordot(nT, A,([0,2],[0,3]))

    #          0
    #       ___Pt___
    #       |       |
    #       |       |
    # 1<-2--A-------T
    #       3       1
    #       1       0
    #       |___P___|
    #           2 
    T = torch.tensordot(nT, P,([1,3],[0,1]))
    T = T.contiguous()

#####################################################################
# functions building pair of 4x2 (or 2x4) halves of 4x4 TN
#####################################################################
def halves_of_4x4_CTM_MOVE_UP(coord, ipeps, env):
    # C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
    # T A B(coord) T   C2x2_LD(coord+(-1,-1)) C2x2(coord+(0,1))
    # T C D        T
    # C T T        C

    # C2x2--1->0 0--C2x2(coord) =     _0 0_
    # |0           1|                |     |
    # |0           0|             half2    half1
    # C2x2--1    1--C2x2             |_1 1_|
    C2x2_1 = c2x2_RU(coord, ipeps, env)
    C2x2_2 = c2x2_RD((coord[0], coord[1]+1), ipeps, env)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[0]))

    C2x2_1 = c2x2_LU((coord[0]-1, coord[1]), ipeps, env)
    C2x2_2 = c2x2_LD((coord[0]-1, coord[1]-1), ipeps, env)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[0]))

    return half1, half2

def halves_of_4x4_CTM_MOVE_LEFT(coord, ipeps, env):
    # C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
    # T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
    # T C        D T
    # C T        T C

    # C2x2(coord)--1 0--C2x2 = half1
    # |0               1|      |0  |1
    # 
    # |0            1<-0|      |0  |1
    # C2x2--1 1---------C2x2   half2
    C2x2_1 = c2x2_LU(coord, ipeps, env)
    C2x2_2 = c2x2_RU((coord[0]+1, coord[1]), ipeps, env)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[0]))

    C2x2_1 = c2x2_LD((coord[0], coord[1]+1), ipeps, env)
    C2x2_2 = c2x2_RD((coord[0]+1, coord[1]+1), ipeps, env)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([1],[1]))

    return half1, half2

def halves_of_4x4_CTM_MOVE_DOWN(coord, ipeps, env):
    # C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
    # T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
    # T C(coord) D T
    # C T        T C

    # C2x2---------1    1<-0--C2x2 =     _1 1_
    # |0                      |1        |     |
    # |0                      |0      half1    half2
    # C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|
    C2x2_1 = c2x2_LD(coord, ipeps, env)
    C2x2_2 = c2x2_LU((coord[0], coord[1]-1), ipeps, env)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([0],[0]))

    C2x2_1 = c2x2_RD((coord[0]+1, coord[1]), ipeps, env)
    C2x2_2 = c2x2_RU((coord[0]+1, coord[1]-1), ipeps, env)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[1]))

    return half1, half2

def halves_of_4x4_CTM_MOVE_RIGHT(coord, ipeps, env):
    # C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
    # T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
    # T C D(coord) T
    # C T T        C

    # C2x2--1 0--C2x2        = half2
    # |0->1      |1->0         |1  |0
    # 
    # |0->1      |0            |1  |0
    # C2x2--1 1--C2x2(coord)   half1
    C2x2_1 = c2x2_RD(coord, ipeps, env)
    C2x2_2 = c2x2_LD((coord[0]-1, coord[1]), ipeps, env)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[1]))

    C2x2_1 = c2x2_RU((coord[0], coord[1]-1), ipeps, env)
    C2x2_2 = c2x2_LU((coord[0]-1, coord[1]-1), ipeps, env)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[1]))

    return half1, half2

#####################################################################
# functions building 2x2 Corner
#####################################################################
def c2x2_LU(coord, ipeps, env):
    C = env.C[(coord,(-1,-1))]
    T1 = env.T[(coord,(0,-1))]
    T2 = env.T[(coord,(-1,0))]
    A = ipeps.site[coord]

    # C--10--T1--2
    # 0      1
    C2x2 = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 A--3 
    # 2->1    2
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[0,1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T1.size()[2]*A.size()[3],T2.size()[1]*A.size()[2])

    # C2x2--1
    # |
    # 0
    return C2x2

def c2x2_RU(coord, ipeps, env):
    C = env.C[(coord,(1,-1))]
    T1 = env.T[(coord,(1,0))]
    T2 = env.T[(coord,(-1,0))]
    A = ipeps.site[coord]

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2 = torch.tensordot(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2 = torch.tensordot(C2x2, T2, ([0],[2]))

    # 1<-2--T2------C
    #       3       |
    #       0       |
    # 2<-1--A--3 0--T1
    #    3<-2    0<-1
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[3,0]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T2.size()[0]*A.size()[1],T1.size()[2]*A.size()[2])
 
    # 0--C2x2
    #    |
    #    1
    return C2x2

def c2x2_RD(coord, ipeps, env):
    C = env.C[(coord,(1,1))]
    T1 = env.T[(coord,(0,1))]
    T2 = env.T[(coord,(1,0))]
    A = ipeps.site[coord]

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2 = torch.tensordot(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2 = torch.tensordot(C2x2, T2, ([0],[2]))

    #    2<-0    1<-2
    # 3<-1--A--3 3--T2
    #       2       |
    #       0       |
    # 0<-1--T1------C
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[3,0]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T2.size()[0]*A.size()[0],T1.size()[1]*A.size()[1])

    #    0
    #    |
    # 1--C2x2
    return C2x2

def c2x2_LD(coord, ipeps, env):
    C = env.C[(coord,(-1,1))]
    T1 = env.T[(coord,(-1,0))]
    T2 = env.T[(coord,(1,0))]
    A = ipeps.site[coord]

    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2 = torch.tensordot(C, T1, ([0],[1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T1--2->3
    C2x2 = torch.tensordot(C2x2, T2, ([0],[1]))

    # 0       0->2
    # T1--1 1--A--3
    # |        2
    # |        2
    # C--------T2--3->1
    C2x2 = torch.tensordot(C2x2, A, ([1,2],[1,2]))

    # permute 0123->0213
    # reshape (02)(13)
    C2x2 = C2x2.permute(0,2,1,3).contiguous().view(T1.size()[0]*A.size()[0],T2.size()[2]*A.size()[3])

    # 0
    # |
    # C2x2--1
    return C2x2