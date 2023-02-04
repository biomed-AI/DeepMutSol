from openpyxl import Workbook
from dataloader import *
#from dataloader1 import *
from model import *
from utrls import *
import time
import os
import torch
import torch.nn as nn
import argparse
#export CUDA_VISIBLE_DEVICES=2 && python test.py
if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_loader('dataset1/test2_dataset.xlsx','test2_dataset',batch_size=1024, shuffle=False,num_workers=1)
    
    net = Model5().to(device)
    output = []
    best_accuracy=0
    fold = 0
    #f_att_hsp: 'cvs_0.5/f_att_hsp/models/fold1_29.ckpt','cvs_0.5/f_att_hsp/models/fold2_26.ckpt','cvs_0.5/f_att_hsp/models/fold3_26.ckpt','cvs_0.5/f_att_hsp/models/fold4_30.ckpt','cvs_0.5/f_att_hsp/models/fold5_26.ckpt','cvs_0.5/f_att_hsp/models/fold6_28.ckpt','cvs_0.5/f_att_hsp/models/fold7_19.ckpt','cvs_0.5/f_att_hsp/models/fold8_27.ckpt','cvs_0.5/f_att_hsp/models/fold9_27.ckpt','cvs_0.5/f_att_hsp/models/fold10_27.ckpt'
    # f: 'cvs_0.5/f/models/fold1_29.ckpt','cvs_0.5/f/models/fold2_19.ckpt','cvs_0.5/f/models/fold3_21.ckpt','cvs_0.5/f/models/fold4_19.ckpt','cvs_0.5/f/models/fold5_23.ckpt','cvs_0.5/f/models/fold6_29.ckpt','cvs_0.5/f/models/fold7_25.ckpt','cvs_0.5/f/models/fold8_29.ckpt','cvs_0.5/f/models/fold9_25.ckpt','cvs_0.5/f/models/fold10_30.ckpt'
    # f: 'cvs_0.5/f/models/fold1_27.ckpt','cvs_0.5/f/models/fold2_19.ckpt','cvs_0.5/f/models/fold3_21.ckpt','cvs_0.5/f/models/fold4_19.ckpt','cvs_0.5/f/models/fold5_23.ckpt','cvs_0.5/f/models/fold6_29.ckpt','cvs_0.5/f/models/fold7_25.ckpt','cvs_0.5/f/models/fold8_29.ckpt','cvs_0.5/f/models/fold9_25.ckpt','cvs_0.5/f/models/fold10_21.ckpt'
    # f_att:
    # pssm: 'cvs_0.5/pssm/models/fold1_1.ckpt','cvs_0.5/pssm/models/fold2_5.ckpt','cvs_0.5/pssm/models/fold3_2.ckpt','cvs_0.5/pssm/models/fold4_1.ckpt','cvs_0.5/pssm/models/fold5_1.ckpt','cvs_0.5/pssm/models/fold6_8.ckpt','cvs_0.5/pssm/models/fold7_2.ckpt','cvs_0.5/pssm/models/fold8_1.ckpt','cvs_0.5/pssm/models/fold9_4.ckpt','cvs_0.5/pssm/models/fold10_1.ckpt']):
    for modelname in list(['cvs_0.5/no_reverse_f_att/models/fold1_7.ckpt','cvs_0.5/no_reverse_f_att/models/fold2_11.ckpt','cvs_0.5/no_reverse_f_att/models/fold3_8.ckpt','cvs_0.5/no_reverse_f_att/models/fold4_8.ckpt','cvs_0.5/no_reverse_f_att/models/fold5_9.ckpt','cvs_0.5/no_reverse_f_att/models/fold6_10.ckpt','cvs_0.5/no_reverse_f_att/models/fold7_8.ckpt','cvs_0.5/no_reverse_f_att/models/fold8_8.ckpt','cvs_0.5/no_reverse_f_att/models/fold9_10.ckpt','cvs_0.5/no_reverse_f_att/models/fold10_13.ckpt']):
        
        fold+=1
        val_loader = get_loader('dataset1/val_dataset.xlsx', 'fold'+str(fold), batch_size=1024,shuffle=True, num_workers=1)
        
        net.load_state_dict(torch.load(f'{modelname}'))
        net.eval()
        best_accuracy_fold=0

        with torch.no_grad():
            for j, (features, adj, attention, label) in enumerate(val_loader):
                features = features.to(device)
                adj = adj.to(device)
                attention = attention.to(device)
                #hsp = hsp.to(device)
                label = label.to(dtype=torch.float32,device=device)
                zero = torch.zeros_like(label)
                one = torch.ones_like(label)
                no_one = -1*one
                output1 = net(features, adj, attention)  # mdoel output
                output1 = output1.reshape(-1)
                valloss = net.criterion(output1, label) 
                print(valloss)
                for a in range(20,50):
                    print(a)
                    for b in range (1,30):
                        pred = torch.where(output1<=-0.01*a, no_one, output1)
                        pred = torch.where(((-0.01*a<output1) * (output1<=0.01*b)), zero, pred)
                        pred = torch.where(output1>0.01*b, one, pred)
                        acc_b = get_acc(label, pred, balance=True, k=3)
                        if acc_b > best_accuracy_fold:
                            best_accuracy_fold, best_a, best_b = acc_b, a, b

                pred = torch.where(output1<=-0.01*best_a, no_one, output1)
                pred = torch.where(((-0.01*best_a<output1) * (output1<=0.01*best_b)), zero, pred)
                pred = torch.where(output1>0.01*best_b, one, pred)
                acc, gc2, res = get_metrics(label, pred, balance=False, k=3)
                ## balance
                acc_b, gc2_b, res_b = get_metrics(label, pred, balance=True, k=3)
            
                #f1 = open('cvs_0.5/f_att/results/test_0.txt', 'a')
                f1 = open('cvs_0.5/no_reverse_f_att/results/test0.txt', 'a')
                f1.write(modelname + '| loss:%.3f\t' % valloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc + '| GC2:{}\t'.format(gc2) +
                'class_0:\t' + '| TP:{}\t'.format(res['tp'][0]) + '| TN:{}\t'.format(res['tn'][0]) + '| FP:{}\t'.format(res['fp'][0]) + '| FN:{}\t'.format(res['fn'][0]) + '| sensitivity:%.3f\t' % res['tpr'][0] + '| specificity:%.3f\t' % res['tnr'][0] + '| PPV:{}\t'.format(res['ppv'][0]) + '| NPV:{}\t'.format(res['npv'][0]) +
                'class_1:\t' + '| TP:{}\t'.format(res['tp'][1]) + '| TN:{}\t'.format(res['tn'][1]) + '| FP:{}\t'.format(res['fp'][1]) + '| FN:{}\t'.format(res['fn'][1]) + '| sensitivity:%.3f\t' % res['tpr'][1] + '| specificity:%.3f\t' % res['tnr'][1] + '| PPV:{}\t'.format(res['ppv'][1]) + '| NPV:{}\t'.format(res['npv'][1]) +
                'class_2:\t' + '| TP:{}\t'.format(res['tp'][2]) + '| TN:{}\t'.format(res['tn'][2]) + '| FP:{}\t'.format(res['fp'][2]) + '| FN:{}\t'.format(res['fn'][2]) + '| sensitivity:%.3f\t' % res['tpr'][2] + '| specificity:%.3f\t' % res['tnr'][2] + '| PPV:{}\t'.format(res['ppv'][2]) + '| NPV:{}\t'.format(res['npv'][2]) + '\n')
                f1.write(modelname + '| loss:%.3f\t' % valloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\t'.format(gc2_b) +
                        'class_0:\t' + '| TP:{}\t'.format(res_b['tp'][0]) + '| TN:{}\t'.format(res_b['tn'][0]) + '| FP:{}\t'.format(res_b['fp'][0]) + '| FN:{}\t'.format(res_b['fn'][0]) + '| sensitivity:%.3f\t' % res_b['tpr'][0] + '| specificity:%.3f\t' % res_b['tnr'][0] + '| PPV:{}\t'.format(res_b['ppv'][0]) + '| NPV:{}\t'.format(res_b['npv'][0]) +
                        'class_1:\t' + '| TP:{}\t'.format(res_b['tp'][1]) + '| TN:{}\t'.format(res_b['tn'][1]) + '| FP:{}\t'.format(res_b['fp'][1]) + '| FN:{}\t'.format(res_b['fn'][1]) + '| sensitivity:%.3f\t' % res_b['tpr'][1] + '| specificity:%.3f\t' % res_b['tnr'][1] + '| PPV:{}\t'.format(res_b['ppv'][1]) + '| NPV:{}\t'.format(res_b['npv'][1]) +
                        'class_2:\t' + '| TP:{}\t'.format(res_b['tp'][2]) + '| TN:{}\t'.format(res_b['tn'][2]) + '| FP:{}\t'.format(res_b['fp'][2]) + '| FN:{}\t'.format(res_b['fn'][2]) + '| sensitivity:%.3f\t' % res_b['tpr'][2] + '| specificity:%.3f\t' % res_b['tnr'][2] + '| PPV:{}\t'.format(res_b['ppv'][2]) + '| NPV:{}\t'.format(res_b['npv'][2]) + '\n')
                f1.close()
        """
        net.load_state_dict(torch.load(f'{modelname}'))
        net.eval()

        with torch.no_grad():
            for j, (features, adj, attention, label) in enumerate(test_loader):
                features = features.to(device)
                adj = adj.to(device)
                attention = attention.to(device)
                #hsp = hsp.to(device)
                label = label.to(dtype=torch.float32,device=device)
                zero = torch.zeros_like(label)
                one = torch.ones_like(label)
                no_one = -1*one
    
                output1 = net(features, adj, attention)  # mdoel output
                output1 = output1.reshape(-1)
                testloss = net.criterion(output1, label) 
                print(testloss)


        output1 = output1.cpu().numpy()
                
        output.append(output1)
    output = torch.tensor(output).to(device)
    output = torch.mean(output,dim=0)
    testloss = net.criterion(output, label) 

    for a in range(10,40):
        for b in range (1,20):
            pred = torch.where(output<=-0.01*a, no_one, output)
            pred = torch.where(((-0.01*a<output) * (output<=0.01*b)), zero, pred)
            pred = torch.where(output>0.01*b, one, pred)
            acc_b = get_acc(label, pred, balance=True, k=3)
            if acc_b > best_accuracy:
                best_accuracy, best_a, best_b = acc_b, a, b

    pred = torch.where(output<=-0.01*best_a, no_one, output)
    pred = torch.where(((-0.01*best_a<output) * (output<=0.01*best_b)), zero, pred)
    pred = torch.where(output>0.01*best_b, one, pred)
    acc, gc2, res = get_metrics(label, pred, balance=False, k=3)
    ## balance
    acc_b, gc2_b, res_b = get_metrics(label, pred, balance=True, k=3)
            
             
    f1 = open('cvs_0.5/no_reverse_f_att/results/test_0.txt', 'a')
    f1.write('test' + '| loss:%.3f\t' % testloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc + '| GC2:{}\t'.format(gc2) +
            'class_0:\t' + '| TP:{}\t'.format(res['tp'][0]) + '| TN:{}\t'.format(res['tn'][0]) + '| FP:{}\t'.format(res['fp'][0]) + '| FN:{}\t'.format(res['fn'][0]) + '| sensitivity:%.3f\t' % res['tpr'][0] + '| specificity:%.3f\t' % res['tnr'][0] + '| PPV:{}\t'.format(res['ppv'][0]) + '| NPV:{}\t'.format(res['npv'][0]) +
            'class_1:\t' + '| TP:{}\t'.format(res['tp'][1]) + '| TN:{}\t'.format(res['tn'][1]) + '| FP:{}\t'.format(res['fp'][1]) + '| FN:{}\t'.format(res['fn'][1]) + '| sensitivity:%.3f\t' % res['tpr'][1] + '| specificity:%.3f\t' % res['tnr'][1] + '| PPV:{}\t'.format(res['ppv'][1]) + '| NPV:{}\t'.format(res['npv'][1]) +
            'class_2:\t' + '| TP:{}\t'.format(res['tp'][2]) + '| TN:{}\t'.format(res['tn'][2]) + '| FP:{}\t'.format(res['fp'][2]) + '| FN:{}\t'.format(res['fn'][2]) + '| sensitivity:%.3f\t' % res['tpr'][2] + '| specificity:%.3f\t' % res['tnr'][2] + '| PPV:{}\t'.format(res['ppv'][2]) + '| NPV:{}\t'.format(res['npv'][2]) + '\n')
    f1.write('test' + '| loss:%.3f\t' % testloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\t'.format(gc2_b) +
                        'class_0:\t' + '| TP:{}\t'.format(res_b['tp'][0]) + '| TN:{}\t'.format(res_b['tn'][0]) + '| FP:{}\t'.format(res_b['fp'][0]) + '| FN:{}\t'.format(res_b['fn'][0]) + '| sensitivity:%.3f\t' % res_b['tpr'][0] + '| specificity:%.3f\t' % res_b['tnr'][0] + '| PPV:{}\t'.format(res_b['ppv'][0]) + '| NPV:{}\t'.format(res_b['npv'][0]) +
                        'class_1:\t' + '| TP:{}\t'.format(res_b['tp'][1]) + '| TN:{}\t'.format(res_b['tn'][1]) + '| FP:{}\t'.format(res_b['fp'][1]) + '| FN:{}\t'.format(res_b['fn'][1]) + '| sensitivity:%.3f\t' % res_b['tpr'][1] + '| specificity:%.3f\t' % res_b['tnr'][1] + '| PPV:{}\t'.format(res_b['ppv'][1]) + '| NPV:{}\t'.format(res_b['npv'][1]) +
                        'class_2:\t' + '| TP:{}\t'.format(res_b['tp'][2]) + '| TN:{}\t'.format(res_b['tn'][2]) + '| FP:{}\t'.format(res_b['fp'][2]) + '| FN:{}\t'.format(res_b['fn'][2]) + '| sensitivity:%.3f\t' % res_b['tpr'][2] + '| specificity:%.3f\t' % res_b['tnr'][2] + '| PPV:{}\t'.format(res_b['ppv'][2]) + '| NPV:{}\t'.format(res_b['npv'][2]) + '\n')
    f1.close()  
    """

    """softm = torch.nn.Softmax(dim=1)
                    score = softm(output)
                    scoreMax = torch.max(score, 1)[1]
                    acc, gc2, metr = get_metrics(label, scoreMax, balance=False, k=3)
                    cv_accuracy.append(acc)
                    cv_gc2.append(gc2)
                    metr['tag'] = metr.index
                    metr['cv'] = 'cv%s' % (i + 1)
                    cv_metrics.append(metr)
                    ## balance
                    acc_b, gc2_b, metr_b = get_metrics(label, scoreMax, balance=True, k=3)
                    cv_accuracy_balance.append(acc_b)
                    cv_gc2_balance.append(gc2_b)
                    metr_b['tag'] = metr_b.index
                    metr_b['cv'] = 'cv%s' % (i + 1)
                    cv_metrics_balance.append(metr_b)
        val_loss = val_loss_sum/10
        res_metr = pd.concat(cv_metrics)
        res_metr_balance = pd.concat(cv_metrics_balance)
        acc = np.mean(cv_accuracy)
        gc2 = np.mean(cv_gc2)
        #print(res_metr)
        metr = res_metr.groupby('tag').mean()
        res = metr.unstack()
        acc_balance = np.mean(cv_accuracy_balance)
        gc2_balance = np.mean(cv_gc2_balance)
        metr_balance = res_metr_balance.groupby('tag').mean()
        res_b = metr_balance.unstack()
        """                