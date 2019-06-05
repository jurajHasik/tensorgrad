'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577 
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import torch
import torch.nn.functional as Functional
import scipy.sparse.linalg

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class ARNOLDISVD(torch.autograd.Function):

    # INPUT:
    # M - input matrix
    # k [int] - desired rank
    # p [int] - oversampling rank. Total rank sampled k+p
    # vnum [int] - ?
    # q [int] - number of matrix-vector multiplications for power scheme
    # s [int] - reorthogonalization
    # OUTPUT
    # U, S, V - truncated svd decomposition M \approx U S V^dag
    @staticmethod
    def forward(self, M, k):
        # check symmetry of M
        #print("norm(0.5*(M-M^t)) "+str(torch.norm(M - M.t())))

        # get M as numpy ndarray
        M_nograd = M.clone().detach().numpy()
        M_nograd = M_nograd @ M_nograd
        #print("FWD k: "+ str(k) + " M: " + str(M_nograd.shape))

        # we get, say (since M is symmetric) left singular vectors of M
        # M = U.S.Vt -> M.Mt = U.S.S.Ut
        s, u = scipy.sparse.linalg.eigsh(M_nograd, k=k)

        # Debug info about properties of output tensors from eigsh
        #print("u "+str(u.shape))
        #print("u "+str(u.strides))

        # find range
        # print(s)
        # if (s[0] is np.nan) or (s[0] != s[0]):
        #     raise Exception('nan')
        # rankS = len(list(filter(lambda e: e > 0., s)))
        # print(rankS)

        # properly order the result (singular values are given in ascending
        # order) by ARPACK
        u = np.copy(u[:,::-1])
        s = np.copy(s[::-1])
        s = np.sqrt(np.abs(s))

        U = torch.as_tensor(u)
        S = torch.as_tensor(s)
        
        # compute right singular vectors as Mt = V.S.Ut /.U => Mt.U = V.S
        # since M = Mt, M.U = V.S
        V = M @ U
        V = Functional.normalize(V, p=2, dim=0)

        # print("U "+str(U.shape))
        # print(U.stride())
        # for i in range(k):
        #     print(str(i)+": "+str(np.linalg.norm(U[:,i])))
        #print(S.shape)
        #print(S)
        # print("V "+str(V.shape))
        # print(V.stride())
        # for i in range(k):
        #     print(str(i)+": "+str(np.linalg.norm(V[:,i])))

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA, None

# def test_svd():
#     M, N = 50, 40
#     torch.manual_seed(2)
#     input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
#     assert(torch.autograd.gradcheck(SVD.apply, input, eps=1e-6, atol=1e-4))
#     print("Test Pass!")

# if __name__=='__main__':
#     test_svd()
