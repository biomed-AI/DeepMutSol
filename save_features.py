import os
import json
import glob
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import HSExposure,DSSP
from tqdm import tqdm
import numpy as np
import xlrd
import pickle

def coors(name, pdb_file):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(name, pdb_file)
    model = structure[0]
    X = []
    for chain in model.get_list():
        for residue in chain.get_list():
            if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                c = residue['C'].get_coord().tolist()
                n = residue['N'].get_coord() .tolist()     # 获取坐标
                ca = residue['CA'].get_coord().tolist()
                o = residue['O'].get_coord().tolist()
                X.append([n, ca, c, o])      # 获取坐标
    X = np.array(X)                       
    return X

def featurize(x):
    """ Pack and pad batch into torch tensors """
    
    mask = np.isfinite(np.sum(x,(1,2))).astype(np.float32)
    #x_pad[isnan] = 0.
    # Conversion
    X = torch.from_numpy(x).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, mask




for j in range(1):
    workbook=xlrd.open_workbook('dataset/train_dataset.xls')
    sheet=workbook.sheet_by_name('sheet1')
    col_id=sheet.col_values(0)    # 获取第0列内容
    col_wt=sheet.col_values(1)
    col_pos=sheet.col_values(2)
    col_mt=sheet.col_values(3)
    isrev='pos'
    for i in range(1,len(col_id)):
        id=col_id[i]
        mutation = col_wt[i]+str(int(col_pos[i]))+col_mt[i]
        
        if isrev=='pos':
            features_wt = torch.tensor(np.load('features/pon2_protrans/'+id+'.npy'))
            features_mt = torch.tensor(np.load('features/pon2_protrans/'+id+'_'+mutation+'.npy'))
        else:
            rev_mutation = mutation[-1]+mutation[1:-1]+mutation[0]    
            features_wt = torch.tensor(np.load('features/pon2_protrans/'+id+'_'+rev_mutation+'.npy'))
            features_mt = torch.tensor(np.load('features/pon2_protrans/'+id+'.npy'))
       
        f =  features_wt-features_mt
        d, E_idx = torch.topk(f.abs().sum(1), 30, dim=-1, largest=True)
        E_idx = E_idx.numpy()
        idx = np.sort(E_idx)
        np.save('features/case_idx/'+id+'_'+mutation+'.npy', idx)

        idx = torch.tensor(idx)
        coors = torch.tensor(np.load('features/coors/'+id+'.npy'))[:,1,:]
        X = torch.index_select(coors,dim=0,index=idx).numpy()
        y = []
        for i in range(X.__len__()):
            y.append(X)
        y = np.array(y)
        x = y.transpose(1, 0, 2)
        a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
        a = 2 / (1 + a / 4)
        for i in range(len(a)):
            a[i,i] = 1
    
        np.save('features/case_adj/'+id+'_'+mutation+'.npy',a)



