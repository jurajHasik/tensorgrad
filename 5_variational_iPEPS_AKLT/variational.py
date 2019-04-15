'''
Variational PEPS with automatic differentiation and GPU support
'''

import io
import torch
import numpy as np
from args import args
torch.set_num_threads(args.omp_cores)
torch.manual_seed(args.seed)
import subprocess
from utils import kronecker_product as kron
from utils import save_checkpoint, load_checkpoint, printTensorAsCoordJson
from ipeps import iPEPS
import su2

if __name__=='__main__':
    import time
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64
    print ('use', dtype)

    model = iPEPS(args, dtype, device, args.use_checkpoint)

    if args.load is not None:
        try:
            load_checkpoint(args.load, args, model)
            print('load model', args.load)
        except FileNotFoundError:
            print('not found:', args.load)

    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10)
    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)

    key = args.folder
    key += args.model \
          + '_d' + str(args.d) \
          + '_D' + str(args.D) \
          + '_chi' + str(args.chi)
    if (args.float32):
        key += '_float32'
    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)

    if args.model == 'AKLT':
        dimS = args.d
        #Hamiltonian operators on a bond
        #sx = torch.from_numpy(su2.get_su2_op(""), dtype=dtype, device=device)*0.5
        #sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5
        sp = torch.from_numpy(su2.get_su2_op("sp",dimS)) #, dtype=dtype, device=device)
        sm = torch.from_numpy(su2.get_su2_op("sm",dimS)) #, dtype=dtype, device=device)
        sz = torch.from_numpy(su2.get_su2_op("sz",dimS)) #, dtype=dtype, device=device)*0.5
        sx = 0.5*(sp+sm)
        sy = 0.5*(sp-sm)
        idop = torch.from_numpy(su2.get_su2_op("I",dimS)) #, dtype=dtype, device=device)
        rot = torch.from_numpy(su2.get_rot_op(dimS))

        # supply just S.S
        SS = kron(sz,rot.t()@sz@rot)+0.5*(kron(sm, rot.t()@sp@rot)+kron(sp,rot.t()@sm@rot))
        H = (1.0/14.0) * (SS + (7.0/10.0)*SS@SS + (7.0/45.0)*SS@SS@SS \
            + (1.0/90.0)*SS@SS@SS@SS)

        D, U = torch.eig(H, eigenvectors=False)
        print(D)

        Mpx = kron(sx, idop)
        Mpy = kron(sy, idop)
        Mpz = kron(sz, idop)
    
    else:
        print ('what model???')
        sys.exit(1)

    print ('Hamiltonian:\n', H)

    def closure():
        optimizer.zero_grad()
        t0_fwd = time.time()
        loss, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi)
        t1_fwd = time.time()
        t0_bk = time.time()
        loss.backward()
        t1_bk = time.time()
        #print("fwd[s]: "+str(t1_fwd-t0_fwd)+" bk[s]: "+str(t1_bk-t0_bk))
        #print (model.A.norm().item(), model.A.grad.norm().item(), loss.item(), Mx.item(), My.item(), Mz.item(), torch.sqrt(Mx**2+My**2+Mz**2).item(), forward-start, time.time()-forward)
        return loss

    with io.open(key+'.log', 'a', buffering=1, newline='\n') as logfile:
        for epoch in range(args.Nepochs):
            t0_iter = time.time()
            loss = optimizer.step(closure)
            t1_iter = time.time()
            if (epoch%args.save_period==0):
                save_checkpoint(key+'/peps.tensor'.format(epoch), model, optimizer)

            with torch.no_grad():
                En, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi if args.chi_obs is None else args.chi_obs)
                Mg = torch.sqrt(Mx**2+My**2+Mz**2)
                message = ('{} ' + 6*'{:.8f} ').format(epoch, En, Mx, My, Mz, Mg, t1_iter-t0_iter)
                print ('epoch, En, Mx, My, Mz, Mg, t[s]', message)
                logfile.write(message + u'\n')
                printTensorAsCoordJson(model.A, key+'/peps.json')
