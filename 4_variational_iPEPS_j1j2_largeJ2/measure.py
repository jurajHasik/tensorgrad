import torch

def get_obs(Asymm, H, Sx, Sy, Sz, C, E ):
    # A(phy,u,l,d,r), C(d,r), E(u,r,d)
    
    Da = Asymm.size()
    Td = torch.einsum('mefgh,nabcd->eafbgchdmn',(Asymm,Asymm)).contiguous().view(Da[1]**2, Da[2]**2, Da[3]**2, Da[4]**2, Da[0], Da[0])
    #print( torch.dist( Td, Td.permute(0,3,2,1,4,5) ) )    # test left-right reflection symmetry of Td

    # get double layer tensor (u,l,d,r)
    AA = torch.einsum('mefgh,mabcd->eafbgchd',(Asymm,Asymm)).contiguous().view(Da[1]**2, Da[2]**2, Da[3]**2, Da[4]**2)
    
    # C-d-E--a
    # |   |
    # 1   g
    CE = torch.tensordot(C,E,([1],[0]))         # C(1d)E(dga)->CE(1ga)
    
    # C----E--a
    # |1   |
    # E--e g
    # |
    # 2
    EL = torch.tensordot(E,CE,([2],[0]))        # E(2e1)CE(1ga)->EL(2ega)  use E(2e1) == E(1e2)

    # C---E--a
    # |   |g
    # E-e-AA--r
    # |   |
    # 2   d
    C_lu = torch.tensordot(EL,AA,([1,2],[1,0]))  # EL(2ega)AA(gedr)->C_lu(2adr)   
    
    # C---E--a
    # |   |g
    # E-e-Td--b
    # |   | \
    # 2   h  mn
    EL = torch.tensordot(EL,Td,([1,2],[1,0]))   # EL(2ega)T(gehbmn)->EL(2ahbmn)
    
    EL3x1 = torch.tensordot(EL,CE,([0,2],[0,1]))   # EL(2ahbmn)CE(2hc)->EL3x1(abmnc), use CE(2hc) == CE(1ga) 
    Rho3x4 = torch.tensordot(EL3x1,EL3x1,([0,1,4],[0,1,4])).permute(0,2,1,3).contiguous().view(Da[0]**2,Da[0]**2)
    
    # print( (Rho3x4-Rho3x4.t()).norm() )
    Rho3x4 = 0.5*(Rho3x4 + Rho3x4.t())
    
    Tnorm = Rho3x4.trace()
    Mx = torch.mm(Rho3x4,Sx).trace()/Tnorm
    My = torch.mm(Rho3x4,Sy).trace()/Tnorm
    Mz = torch.mm(Rho3x4,Sz).trace()/Tnorm
   
    # C--E---a      a---E--C
    # E--Td--b   b<=r--AA--E
    # |  | \        |   | 
    # 2  h mn       d   2
    #                   =>1                                          2adr
    Rho4x4 = torch.tensordot(EL,C_lu,([1,3],[1,3])) # EL(2ahbmn)C_lu(1adb)->Rho4x4(2hmn1d)

    # C--E-------E--C
    # E--Td------A--E
    # |  | \     |  | 
    # 2  h mn    d  1
    #                     
    #    =>h
    # 2  d
    # |  |
    # E--AA--r
    # |  |
    # C--E---a                                                               2adr
    Rho4x4 = torch.tensordot(Rho4x4,C_lu,([0,1],[0,2])) # Rho4x4(2hmn1d)C_lu(2ahr)->Rho4x4(mn1dar)

    # C--E-------E--C
    # E--Td------A--E
    # |  | \     |  | 
    # |  | mn    d  1  =>qs                   
    # |  |              mn  h  2
    # |  |                \ |  |
    # E--AA--r       r<=b--Td--E
    # |  |                  |  |
    # C--E---a          a---E--C                                                 2ahbmn
    Rho4x4 = torch.tensordot(Rho4x4,EL,([2,3,4,5],[0,2,1,3])) # Rho4x4(mn1dar)EL(2ahrqs)->Rho4x4(mnqs)
    Rho4x4 = Rho4x4.permute(0,2,1,3).contiguous().view(Da[0]**2,Da[0]**2)

    Rho4x4 = 0.5*(Rho4x4 + Rho4x4.t())
    Rho4x4norm = Rho4x4.trace()

    # extract individual tensors
    HnnAA, HnnAB, HnnnAB = torch.split(H,4,0)

    # We are assuming following Ansatz
    #     ...
    #     A B A B
    #     A B A B
    # ... A B A B ...
    #     ...
    # Hence, there are three non-equivalent contributions
    # 
    # 1) <AA S.S AA> = <BB S.S BB> NN vertical
    # 2) <AB S.S AB> NN horizontal
    # 3) <AB S.S AB> NNN diagonal 
    Energy = 0.5*torch.mm(Rho3x4,HnnAA).trace()/Tnorm \
        + 0.5*torch.mm(Rho3x4,HnnAB).trace()/Tnorm \
        + torch.mm(Rho4x4,HnnnAB).trace()/Rho4x4norm

    #print("Tnorm = %g, Energy = %g " % (Tnorm.item(), Energy.item()) )

    return Energy, Mx, My, Mz

