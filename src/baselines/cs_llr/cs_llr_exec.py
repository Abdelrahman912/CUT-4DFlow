import os 
import time 
import torch
import einops
import argparse
import numpy as np
from .cs_llr_utils import *

def CS_LLR(lamb_tv, lamb_llr, ksp, coils, seg=False, dev='cuda:0'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    """
    LLR and spatiotemporal TV are implementated here (for the FlowMRI_net paper lamb_tv=0)

    You could reconstruct all segments (velocity-encodings) at once by removing the seg loop and indexing
    This will be faster but requires more GPU memory

    ksp : (V, C, T, D, H, W) complex64
    coils : (C, 1, D, H, W) complex64
    """
    V, C, T, X, Y, Z = ksp.shape
    if seg:
        rec = np.zeros((V, T, X, Y, Z)).astype('complex64')
        for v in range(V): 
            V = 1
            ksp_ = ksp[v:v+1]
            mask = ksp_[:,0:1,:,0:1,:,:] !=  0
            norm_k = np.median(np.abs(ksp[:,:,:, X//2, Y//2, Z//2])) / 10
            ksp_ = ksp_ / norm_k

            with torch.no_grad():
                ksp_ = (ksp_ if isinstance(ksp_, torch.Tensor) else torch.from_numpy(ksp_)).to(torch.complex64)
                ksp_ = k2i(ksp_, -3)
                coils = (coils if isinstance(coils, torch.Tensor) else torch.from_numpy(coils)).to(torch.complex64)
                mask = (mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)).to(torch.float32)
                mask = (ksp_.abs().sum(dim=(1, -3), keepdims=True) > 0).float()

            # fixed CS-LLR hyperparameters
            lamb_tv_deps = 0.005            # epsilon to prevent zero gradient and smooth the optimization
            lamb_tv_w = (1, 0.1, 0.1, 0.1)  # weightings for gradients over (t, x, y, z)
            bsz = 8                        # LLR block/patch size

            vtv_reg_multi_v_f_lowmem = get_vtv_reg_multi_v_f_lowmem(V, T, lamb_tv, lamb_tv_deps, lamb_tv_w)
                    
            with torch.no_grad():
                dtype = torch.complex64
                coils = coils.to(dev).to(dtype)
                kspc = ksp_.to(dev).to(dtype)
                
                kspc = einops.rearrange(kspc, 'V C T X Y Z -> C (V T) X Y Z')
                mask = (einops.reduce(kspc.abs(), 'C VT X Y Z -> 1 VT 1 Y Z', 'sum') > 0.).float()
                mask = mask.to(dev)
                
                acq_op = return_cartesian_acq_operator(mask, kspc, dims=[-1, -2], coils=coils, force_fftn_device=None)

                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                start_max_memory = torch.cuda.max_memory_allocated()
                tm0 = time.time()
                recv, itinfo = pgd_recon(kspc, acq_op, lamb_llr, alpha=0.3, tau=0.97, 
                                                        bsz=bsz, debug=True, maxiters=70, 
                                                        custom_regularizer_fn=vtv_reg_multi_v_f_lowmem,
                                                        force_eig_on_cpu=True)
                torch.cuda.synchronize()
                print(f'Elapsed {time.time() - tm0}')
                end_max_memory = torch.cuda.max_memory_allocated()
                
                recv = recv.cpu()
                recv = einops.rearrange(recv, '(V T) X Y Z -> V T X Y Z', V=V, T=T)
                kspc = acq_op = None
                
                rec[v] = recv.cpu().numpy() * norm_k
                print('Mem: ', (end_max_memory - start_max_memory)/1024/1024, end_max_memory/1024/1024, start_max_memory/1024/1024)

        return rec
    else:
        mask = ksp[:,0:1,:,0:1,:,:] !=  0
        norm_k = np.median(np.abs(ksp[:,:,:, X//2, Y//2, Z//2])) / 10
        ksp = ksp / norm_k

        with torch.no_grad():
            ksp = torch.from_numpy(ksp).to(torch.complex64)
            ksp = k2i(ksp, -3)
            coils = torch.from_numpy(coils).to(torch.complex64)
            mask = torch.from_numpy(mask).to(torch.float32)
            mask = (ksp.abs().sum(dim=(1, -3), keepdims=True) > 0).float()

        # fixed CS-LLR hyperparameters
        lamb_tv_deps = 0.005            # epsilon to prevent zero gradient and smooth the optimization
        lamb_tv_w = (1, 0.1, 0.1, 0.1)  # weightings for gradients over (t, x, y, z)
        bsz = 8                        # LLR block/patch size

        vtv_reg_multi_v_f_lowmem = get_vtv_reg_multi_v_f_lowmem(V, T, lamb_tv, lamb_tv_deps, lamb_tv_w)
                
        with torch.no_grad():
            dtype = torch.complex64
            coils = coils.to(dev).to(dtype)
            kspc = ksp.to(dev).to(dtype)
            
            kspc = einops.rearrange(kspc, 'V C T X Y Z -> C (V T) X Y Z')
            mask = (einops.reduce(kspc.abs(), 'C VT X Y Z -> 1 VT 1 Y Z', 'sum') > 0.).float()
            mask = mask.to(dev)
            
            acq_op = return_cartesian_acq_operator(mask, kspc, dims=[-1, -2], coils=coils, force_fftn_device=None)

            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            start_max_memory = torch.cuda.max_memory_allocated()
            tm0 = time.time()
            rec, itinfo = pgd_recon(kspc, acq_op, lamb_llr, alpha=0.3, tau=0.97, 
                                                    bsz=bsz, debug=True, maxiters=70, 
                                                    custom_regularizer_fn=vtv_reg_multi_v_f_lowmem,
                                                    force_eig_on_cpu=True)
            torch.cuda.synchronize()
            print(f'Elapsed {time.time() - tm0}')
            end_max_memory = torch.cuda.max_memory_allocated()
            
            rec = rec.cpu()
            rec = einops.rearrange(rec, '(V T) X Y Z -> V T X Y Z', V=V, T=T)
            kspc = acq_op = None
            
            rec = rec * norm_k

            print('Mem: ', (end_max_memory - start_max_memory)/1024/1024, end_max_memory/1024/1024, start_max_memory/1024/1024)

        return rec.cpu().numpy()