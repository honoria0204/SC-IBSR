import os

import torch
import torch.nn as nn
import network_con as network
import pre_process as prep
from torch.utils.data import DataLoader
import data_list
from data_list import ImageList
from torch.autograd import Variable
import pdb
import scipy.io as sio
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import torch.nn.functional as F
import matplotlib.pyplot as plt

def extract_features(loader, model):
    start_test = True
    myfeatures = []
    mylabels = []
    myoutputs = []

    

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            # if pooling:                            # -- tb
            # labels = labels[::4]
            outputs, features= model(inputs)
            # pdb.set_trace()
            myfeatures.extend(features.tolist())
            myoutputs.extend(outputs.tolist())
            mylabels.extend(labels.tolist())         # -- tb
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy, myfeatures, myoutputs, mylabels, predict




testing = True

if testing:
    print('hello world, start testing!')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ## prepare data
    test_bs = 64
    source_test_list = '/root/autodl-tmp/IBSR_jittor/MI3DOR-1/list_rgb_test.txt'
    target_test_list = '/root/autodl-tmp/IBSR_jittor/MI3DOR-1/list_view_test_4views.txt'
    dsets_src_test = ImageList(open(source_test_list).readlines(), transform=prep.image_test())
    dset_loaders_src_test = DataLoader(dsets_src_test, batch_size=test_bs, shuffle=False, num_workers=4)
    dsets_tgt_test = ImageList(open(target_test_list).readlines(), transform=prep.image_test())
    dset_loaders_tgt_test = DataLoader(dsets_tgt_test, batch_size=test_bs, shuffle=False, num_workers=4)

    ## set base network
    
    model_path = '/root/autodl-tmp/IBSR_jittor/MI3DOR-1/MI3DOR-1/san/best_model.pth.tar'
    base_network = torch.load(model_path)
    # for param in base_network.parameters():
    #     print(param)      
    # print(base_network)
    # input()
    ## test
    base_network.train(False)
    acc_src, features_src, outputs_src, labels_src, src_predict = extract_features(dset_loaders_src_test, base_network)
    # acc_tgt, features_tgt, outputs_tgt, labels_tgt, tgt_predict = extract_features(dset_loaders_tgt_test, base_network)

    sio.savemat('source_MI3DOR1.mat',
                {'source_feature': features_src, 'source_label': labels_src, 'source_outputs': outputs_src, 'source_predict_label': src_predict.cpu().numpy()})
    # sio.savemat('target_MI3DOR1_0.7.mat',
    #             {'target_feature': features_tgt, 'target_label': labels_tgt, 'target_outputs': outputs_tgt, 'target_predict_label': tgt_predict.cpu().numpy()})
    # np.save('source.npy', {'source_feature': features_src, 'source_outputs': outputs_src, 'source_label': labels_src})
    # np.save('target.npy', {'target_feature': features_tgt, 'target_outputs': outputs_tgt, 'target_label': labels_tgt})
    # np.save("source_feature.npy", features_src)   
    # np.save("source_label.npy", labels_src)   
    # np.save("target_feature.npy", features_tgt)   
    # np.save("target_label.npy", labels_tgt)             
    

    # log_str = "acc_src: {:.05f}, acc_tgt: {:.5f}".format(acc_src, acc_tgt)
    log_str = "acc_src: {:.05f}".format(acc_src)
    # log_str = "acc_tgt: {:.05f}".format(acc_tgt)
    print(log_str)

    testing = False

