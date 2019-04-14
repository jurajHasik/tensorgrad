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
          + '_J2' + str(args.J2) \
          + '_D' + str(args.D) \
          + '_chi' + str(args.chi)
    if (args.float32):
        key += '_float32'
    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)

    if args.model == 'J1J2':
        #Hamiltonian operators on a bond
        sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
        sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5
        sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
        sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
        id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

        # supply just S.S
        Hnn  = args.J1*(2*kron(sz,4*sx@sz@sx)-(kron(sm, 4*sx@sp@sx)+kron(sp,4*sx@sm@sx)))
        Hnnn = args.J2*(2*kron(sz,sz)+(kron(sm,sp)+kron(sp,sm)))
        H = torch.cat((Hnn,Hnnn))

        Mpx = kron(sx, id2)
        Mpy = kron(sy, id2)
        Mpz = kron(sz, id2)
    
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
