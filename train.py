from dataloader import *
from model import *
from utrls import *
import time
import os
import torch
import torch.nn as nn
import argparse

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(10):
        train_loader = get_loader('dataset1/train_dataset.xlsx', 'fold'+str(fold+1), batch_size=64,shuffle=True, num_workers=4)
        val_loader = get_loader('dataset1/val_dataset.xlsx', 'fold'+str(fold+1), batch_size=1024,shuffle=True, num_workers=2)
        #model_name = 'cvs_0.5/f_att/models/fold7_30.ckpt'
        net = Model().to(device)
        #net.load_state_dict(torch.load(f'{model_name}'))
    
        for epoch in range(30):
            net.train()        
            for j, (protrans_features,adj,attention,label) in enumerate(train_loader):
                net.optimizer.zero_grad()  # clear gradients for this training step
                protrans_features = protrans_features.to(device)
                adj = adj.to(device)
                attention = attention.to(device)
                #hsp = hsp.to(device)
                label = label.to(dtype=torch.float32,device=device)
                output = net(protrans_features, adj,attention) 
                output = output.reshape(-1)
                loss = net.criterion(output, label)   ##########
                print(loss)
                loss.backward()  # backpropagation, compute gradients
                net.optimizer.step()
            net.eval()
            with torch.no_grad():
                for j, (protrans_features,adj,attention,label) in enumerate(val_loader):
                    protrans_features = protrans_features.to(device)
                    adj = adj.to(device)
                    attention = attention.to(device)
                    #hsp = hsp.to(device)
                    label = label.to(dtype=torch.float32,device=device)
                    output = net(protrans_features, adj,attention) 
                    output = output.reshape(-1)
                    valloss = net.criterion(output, label)
            one = torch.ones_like(label)
            no_one = -1*one
            zero = torch.zeros_like(label)
            best_accuracy = 0
            for a in range(1,100):
                for b in range (1,100):
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
            
            f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
            f1.write('epoch{}\t'.format(epoch + 1) + '| loss:%.3f\t' % valloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc + '| GC2:{}\t'.format(gc2) +
            'class_0:\t' + '| TP:{}\t'.format(res['tp'][0]) + '| TN:{}\t'.format(res['tn'][0]) + '| FP:{}\t'.format(res['fp'][0]) + '| FN:{}\t'.format(res['fn'][0]) + '| sensitivity:%.3f\t' % res['tpr'][0] + '| specificity:%.3f\t' % res['tnr'][0] + '| PPV:{}\t'.format(res['ppv'][0]) + '| NPV:{}\t'.format(res['npv'][0]) +
            'class_1:\t' + '| TP:{}\t'.format(res['tp'][1]) + '| TN:{}\t'.format(res['tn'][1]) + '| FP:{}\t'.format(res['fp'][1]) + '| FN:{}\t'.format(res['fn'][1]) + '| sensitivity:%.3f\t' % res['tpr'][1] + '| specificity:%.3f\t' % res['tnr'][1] + '| PPV:{}\t'.format(res['ppv'][1]) + '| NPV:{}\t'.format(res['npv'][1]) +
            'class_2:\t' + '| TP:{}\t'.format(res['tp'][2]) + '| TN:{}\t'.format(res['tn'][2]) + '| FP:{}\t'.format(res['fp'][2]) + '| FN:{}\t'.format(res['fn'][2]) + '| sensitivity:%.3f\t' % res['tpr'][2] + '| specificity:%.3f\t' % res['tnr'][2] + '| PPV:{}\t'.format(res['ppv'][2]) + '| NPV:{}\t'.format(res['npv'][2]) + '\n')
            f1.write('epoch{}\t'.format(epoch + 1) + '| loss:%.3f\t' % valloss+'('+str(-0.01*best_a)+','+str(0.01*best_b)+')' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\t'.format(gc2_b) +
            'class_0:\t' + '| TP:{}\t'.format(res_b['tp'][0]) + '| TN:{}\t'.format(res_b['tn'][0]) + '| FP:{}\t'.format(res_b['fp'][0]) + '| FN:{}\t'.format(res_b['fn'][0]) + '| sensitivity:%.3f\t' % res_b['tpr'][0] + '| specificity:%.3f\t' % res_b['tnr'][0] + '| PPV:{}\t'.format(res_b['ppv'][0]) + '| NPV:{}\t'.format(res_b['npv'][0]) +
            'class_1:\t' + '| TP:{}\t'.format(res_b['tp'][1]) + '| TN:{}\t'.format(res_b['tn'][1]) + '| FP:{}\t'.format(res_b['fp'][1]) + '| FN:{}\t'.format(res_b['fn'][1]) + '| sensitivity:%.3f\t' % res_b['tpr'][1] + '| specificity:%.3f\t' % res_b['tnr'][1] + '| PPV:{}\t'.format(res_b['ppv'][1]) + '| NPV:{}\t'.format(res_b['npv'][1]) +
            'class_2:\t' + '| TP:{}\t'.format(res_b['tp'][2]) + '| TN:{}\t'.format(res_b['tn'][2]) + '| FP:{}\t'.format(res_b['fp'][2]) + '| FN:{}\t'.format(res_b['fn'][2]) + '| sensitivity:%.3f\t' % res_b['tpr'][2] + '| specificity:%.3f\t' % res_b['tnr'][2] + '| PPV:{}\t'.format(res_b['ppv'][2]) + '| NPV:{}\t'.format(res_b['npv'][2]) + '\n')
            f1.close()  
            
            torch.save(net.state_dict(), os.path.join('results/models', 'fold'+str(fold+1)+'_'+str(epoch+1)+'.ckpt'))
        
            