import torch
import xlrd,os
from openpyxl import load_workbook
import numpy as np
from torch.utils.data import DataLoader
np.set_printoptions(threshold=np.inf)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def get_features(gi,mutation,isrev,pos):
    if isrev!='rev':
        a = np.load('features/change_large_adj/'+gi+'_'+mutation+'.npy')
        a = normalize(a)
        a = torch.tensor(a)
        E_idx = np.load('features/change_large_idx/'+gi+'_'+mutation+'.npy')
        E_idx = torch.tensor(E_idx)
        attent_wt = torch.tensor(np.load('features/attention/'+gi+'.npy'))
        attent_mt = torch.tensor(np.load('features/attention/'+gi+'_'+mutation+'.npy'))
        features_wt = torch.tensor(np.load('features/pon2_protrans/'+gi+'.npy'))
        features_mt = torch.tensor(np.load('features/pon2_protrans/'+gi+'_'+mutation+'.npy'))
    else:
        rev_mutation = mutation[-1]+mutation[1:-1]+mutation[0]
        a = np.load('features/change_large_adj/'+gi+'_'+rev_mutation+'.npy')
        a = normalize(a)
        a = torch.tensor(a)
        E_idx = np.load('features/change_large_idx/'+gi+'_'+rev_mutation+'.npy')
        E_idx = torch.tensor(E_idx)
        attent_wt = torch.tensor(np.load('features/attention/'+gi+'_'+rev_mutation+'.npy'))
        attent_mt = torch.tensor(np.load('features/attention/'+gi+'.npy'))
        features_wt = torch.tensor(np.load('features/pon2_protrans/'+gi+'_'+rev_mutation+'.npy'))
        features_mt = torch.tensor(np.load('features/pon2_protrans/'+gi+'.npy'))
     
    f_wt = torch.index_select(features_wt,dim=0,index=E_idx)
    f_mt = torch.index_select(features_mt,dim=0,index=E_idx)
    att_wt = torch.index_select(attent_wt,dim=0,index=E_idx)
    att_mt = torch.index_select(attent_mt,dim=0,index=E_idx) 
    features = torch.cat((f_wt,f_mt),dim=1)
    attention = torch.cat((att_wt,att_mt),dim=1)
    
    return features,a,attention

class Dataset():
    def __init__(self, xlsx,sheetname):
        self.data = []
        workbook=xlrd.open_workbook(xlsx)
        sheet=workbook.sheet_by_name(sheetname)
        self.col_gi=sheet.col_values(3)  
        self.col_mutations=sheet.col_values(1)  
        self.col_dds=sheet.col_values(2)   
        self.isrev=sheet.col_values(5)   
        
    def __getitem__(self, index):
        gi = str(int(self.col_gi[index+1]))
        mutation = self.col_mutations[index+1]
        isrev = self.isrev[index+1]
        dds = int(self.col_dds[index+1])
        pos= int(mutation[1:-1])-1
        features,a,attention = get_features(gi,mutation,isrev,pos)
        return features,a,attention,dds

    def __len__(self):
        return len(self.col_gi)-1


def get_loader(xlsx, sheetname, batch_size, shuffle, num_workers):
    trainData = Dataset(xlsx=xlsx,sheetname=sheetname)

    data_loader = torch.utils.data.DataLoader(dataset=trainData,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader