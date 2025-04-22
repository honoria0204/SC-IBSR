import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network_con as network
import loss
from loss import *
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math

from lib import *
# from sag_resnet import sag_resnet
import datetime
import matplotlib.pyplot as plt
import scipy.stats


torch.autograd.set_detect_anomaly(True)


class CQCLoss(nn.Module):

    def __init__(self, batch_size, tau_cqc):
        super(CQCLoss, self).__init__()
        self.batch_size = batch_size
        self.tau_cqc = tau_cqc
        # self.device = device
        self.COSSIM = nn.CosineSimilarity(dim=-1)
        self.CE = nn.CrossEntropyLoss(reduction="sum")
        self.get_corr_mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    def forward(self, Xa, Zb):
        # pdb.set_trace()

        XaZb = torch.cat([Xa, Zb], dim=0)
        # XbZa = torch.cat([Xb, Za], dim=0)

        Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))
        Rab = torch.diag(Cossim_ab, self.batch_size)
        Lab = torch.diag(Cossim_ab, -self.batch_size)
        Pos_ab = torch.cat([Rab, Lab]).view(2 * self.batch_size, 1)
        Neg_ab = Cossim_ab[self.get_corr_mask].view(2 * self.batch_size, -1)

        # Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))
        # Rba = torch.diag(Cossim_ba, self.batch_size)
        # Lba = torch.diag(Cossim_ba, -self.batch_size)    
        # Pos_ba = torch.cat([Rba, Lba]).view(2 * self.batch_size, 1)
        # Neg_ba = Cossim_ba[self.get_corr_mask].view(2 * self.batch_size, -1)


        logits_ab = torch.cat((Pos_ab, Neg_ab), dim=1)
        logits_ab /= self.tau_cqc

        # logits_ba = torch.cat((Pos_ba, Neg_ba), dim=1)
        # logits_ba /= self.tau_cqc

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        
        loss = self.CE(logits_ab, labels)# + self.CE(logits_ba, labels)
        return loss / (2 * self.batch_size)






def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = labels[::4]    # tb
                # pdb.set_trace()
                outputs, _ = model(inputs, pooling=True)    # tb
                if start_test:
                    all_output = outputs.float()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    # pdb.set_trace()
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy
def fanou(vectora, vectorb):
    return 1 / (torch.norm(vectora - vectorb) + 1)

def init_model():
    global model
    model = sag_resnet(depth=int(50),
                       pretrained=True,
                       num_classes=31,
                       drop=0.5,
                       sagnet=True,
                       style_stage=3)

    # print(model)
    model = torch.nn.DataParallel(model).cuda()
    
# def KL(p, q):
#     KL = 0
#     l = p.size()[0]
#     for i in range(l):
#         for j in range(l):
#             # pdb.set_trace()
#             KL += p[i,j] * math.log((p[i,j].item() + 1e-8)/ (q[i,j].item() + 1e-8))
#             # if kll>0:
#             #     KL += p[i,j] * math.log(kll)
#             # else:
#             #     KL += 0
#     return KL
def KL(p, q):
    KL = 0
    l = p.size()[0]
    p = p / p.sum()
    q = q / q.sum()
    # p = torch.softmax(p, dim=-1)
    # q = torch.softmax(q, dim=-1)

    epsilon = 1e-8  # Small value to prevent log(0)
    for i in range(l):
        for j in range(l):
            p_val = p[i, j] + epsilon
            q_val = q[i, j] + epsilon
            KL += p_val * torch.log(p_val / q_val)
    return KL

def SIM(p1,p2): 
    # P = torch.mm(torch.exp(p1), (torch.exp(p2)).t())
    P = torch.mm(p1, p2.t()) 
    # pdb.set_trace()
    
    # diag = torch.diag(P)
    # P_diag = torch.diag_embed(diag)
    # P = P - P_diag    
    
    # P1 = torch.sum(P, dim=(1,), keepdim=True)
    # P = torch.div(P, P1)
    
    P = torch.exp(0.01*P)/torch.exp(0.01*P).sum(dim=1)
    
    return P

def train(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]
    print('hello world')
    

    prep_dict["source"] = prep.image_target(**config["prep"]['params'])
    prep_dict["target"] = prep.image_target(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs//4, \
            shuffle=True, num_workers=4, drop_last=True)                  # tb
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=False, num_workers=4, drop_last=True)       # tb


    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4, drop_last=True)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    # base_network = sag_resnet(depth=int(50),
    #                     pretrained=True,
    #                     num_classes=21,
    #                     drop=0.5,
    #                     style_stage=3)
    base_network = base_network.cuda()
    
    #### quantization head
    # device = torch.device('cuda')
    # N_books = 8
    # N_words = 16
    # L_word = 16
    # tau_q = 5.
    tau_cqc = 0.5
    # Q = Quantization_Head(N_words, N_books, L_word, tau_q, num_class=21)
    # base_network = nn.Sequential(ResNet_Baseline(BasicBlock, [2, 2, 2, 2]), Q)
    # criterion = CQCLoss(train_bs//4, tau_cqc)

    ## add additional network for some CDANs
    random_layer = None
    ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## ============== train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    self_s_centroid = torch.zeros(31, 256)
    self_t_centroid = torch.zeros(31, 256)
    
    # memory bank
    my_source_feature = torch.zeros(train_bs, 256).cuda()
    my_target_feature = torch.zeros(train_bs, 256).cuda()
    my_source_label = torch.zeros(train_bs).cuda()
    my_target_pselabel = torch.zeros(train_bs).cuda()
    con_src = torch.zeros(1,256)
    con_src = con_src.cuda()
    con_tar = torch.zeros(1,256)
    con_tar = con_src.cuda()
    con_loss = 0
    
    

    for iter_step in range(config["num_iterations"]):
        if iter_step % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(iter_step, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        if iter_step % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(iter_step)))
        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, iter_step, **schedule_param)
        optimizer.zero_grad()
        if iter_step % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if iter_step % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target[::4].cuda()
        #[21,3,224,224] [84,3 224,224]
        
        ########
        # a=[]
        # for i in range(0,train_bs,4):
        #     a.append(i)
        # inputs_target1 = inputs_target[a,:,:,:]

        # view pooling
        # features_source0, _ = base_network(inputs_source)
        #[21,256] [21,21] 
        '''input_x = torch.cat([inputs_source, inputs_target1],dim=0)
        features_source, outputs_source = base_network(input_x, sagnet=True)'''
        # pdb.set_trace()
        
        ##new+++
        # pdb.set_trace()
        # inputs_target = inputs_target[::4]
        # outputs_source0, features_source0= base_network(inputs_source, pooling=False)
        outputs_source0, outputs_source, outputs_target, features_source0, features_source, features_target = base_network(inputs_source, xt = inputs_target, sagnet = True)
        # features_target, outputs_target,  = base_network(inputs_target, spq = True)
        # features_source = torch.cat((Xa, Za), dim=1)
        # features_target = torch.cat((Xb, Zb), dim=1)
        
        # pdb.set_trace()

        
        ########
        
        '''features_target, outputs_target = base_network(inputs_target, pooling = True)'''   # tb
        features = torch.cat((features_source, features_target), dim=0)
        # outputs = torch.cat((outputs_source, outputs_target), dim=0)

        softmax_src = nn.Softmax(dim=1)(outputs_source)
        softmax_tgt = nn.Softmax(dim=1)(outputs_target)
        softmax_out = torch.cat((softmax_src, softmax_tgt), dim=0)
        
        # ======== yy clspse
        clspse_label = torch.max(softmax_tgt, 1)[1]
        clspse_label = clspse_label.cuda().long()
        
        
        ### contrastive ###
        if iter_step >= 0:
            
            # if iter_step % 4 == 0:
            #     my_source_feature[0:18, :] = features_source
            #     my_target_feature[0:18, :] = features_target
            #     my_source_label[0:18] = labels_source
            #     my_target_pselabel[0:18] = clspse_label
            # elif iter_step % 4 == 1:
            #     my_source_feature[18:36, :] = features_source
            #     my_target_feature[18:36, :] = features_target
            #     my_source_label[18:36] = labels_source
            #     my_target_pselabel[18:36] = clspse_label
            # elif iter_step % 4 == 2:
            #     my_source_feature[36:54, :] = features_source
            #     my_target_feature[36:54, :] = features_target
            #     my_source_label[36:54] = labels_source
            #     my_target_pselabel[36:54] = clspse_label
            # elif iter_step % 4 == 3:
            #     my_source_feature[54:72, :] = features_source
            #     my_target_feature[54:72, :] = features_target
            #     my_source_label[54:72] = labels_source
            #     my_target_pselabel[54:72] = clspse_label
            #     # pdb.set_trace()
            
            con = 0
            
            for i in range(train_bs//4):
                for j in range(train_bs//4):
                    if labels_source[i]==labels_target[j]:
                        if con==0:
                            con_src = features_source[i,:].unsqueeze(0)
                            con_tar = features_target[j,:].unsqueeze(0)
                            con = 1
                        else:
                            con_src = torch.cat((con_src, features_source[i,:].unsqueeze(0)), dim=0)
                            con_tar = torch.cat((con_tar, features_target[j,:].unsqueeze(0)), dim=0)
            # pdb.set_trace()
            criterion = CQCLoss(con_src.size()[0], tau_cqc)
            con_loss = criterion(con_src, con_tar)
            # del con_src, con_tar
              
        
        
        
        
        # con_loss = criterion(features_source, features_target)
        # pdb.set_trace()
        
        # ======== CO2 CONSISTENT CONTRAST        
        P = SIM(features_source0, features_source)
        Q = SIM(features_source, features_source0)
        # pdb.set_trace()
        # consistent_loss = 0.5*scipy.stats.entropy(P, Q) + 0.5*scipy.stats.entropy(Q, P)
        consistent_loss = 0.5*KL(P, Q) + 0.5*KL(Q, P)
        
        
        ## SM ##
        semantic_loss, self_s_centroid, self_t_centroid = SM(features_source, features_target, labels_source, labels_target, self_s_centroid,
                                                             self_t_centroid, 31)

        if config['CDAN'] == 'CDAN+E':   #####
            import loss
            entropy = loss.Entropy(softmax_out)
            # pdb.set_trace()
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(iter_step), random_layer)
        else:
            raise ValueError('Method cannot be recognized.')
        
        # print(torch.isnan(softmax_tgt).any())
        # input()

        
        
        if torch.any(torch.sum(softmax_tgt, dim=1) == 0):
            raise ValueError("Zero row found in softmax_tgt")

        # Add epsilon for numerical stability
        epsilon = 1e-6
        batch_size, features = softmax_tgt.shape
        eye_matrices = torch.eye(features, device=softmax_tgt.device).unsqueeze(0).repeat(batch_size, 1, 1)
        softmax_tgt_expanded = softmax_tgt.unsqueeze(2)
        softmax_tgt_perturbed = softmax_tgt_expanded + epsilon * torch.bmm(eye_matrices, softmax_tgt_expanded)
        softmax_tgt_perturbed = softmax_tgt_perturbed.squeeze(2)

        # Run SVD
        try:
            _, s_tgt, _ = torch.svd(softmax_tgt_perturbed)
        except RuntimeError as e:
            print(f"SVD failed: {e}")


        if config["method"]=="BNM":      #####
            method_loss = -torch.mean(s_tgt)

        # ======== opl_loss
        op_loss = OrthogonalProjectionLoss(gamma=0.5)
        if iter_step >1200:
            op_lambda = 0.1
        else:
            op_lambda = 0
        opl = op_loss(features_source, labels_source)


        # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=31, epsilon=0.1)(outputs_source, labels_source)
        classifier_target = CrossEntropyLabelSmooth(num_classes=31, epsilon=0.1)(outputs_target, labels_target)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + classifier_target +\
                     loss_params["lambda_method"] * method_loss + 0.1 * semantic_loss + op_lambda*opl + consistent_loss*10 + 0.1*con_loss

        # total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + classifier_target +\
        #              loss_params["lambda_method"] * method_loss + 0.1 * semantic_loss + op_lambda*opl + consistent_loss
        loss_components = [transfer_loss, classifier_loss, classifier_target, method_loss, semantic_loss, opl, consistent_loss, con_loss]
        for i, loss in enumerate(loss_components):
            if torch.isnan(loss).any():
                raise ValueError(f"Loss component {i} contains NaNs")

        total_loss.backward(retain_graph=True)
        # grad = torch.autograd.grad(outputs=total_loss, inputs=input_tensor, create_graph=True, retain_graph=True)
        optimizer.step()
        if iter_step % config['print_num'] == 0:
            log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, consistent: {:.5f}, con: {:.5f}".format(iter_step, classifier_loss, transfer_loss, consistent_loss*10, 0.1*con_loss)
            # log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, consistent: {:.5f}".format(iter_step, classifier_loss, transfer_loss, consistent_loss)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            if config['show']:
                print(log_str)
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--CDAN', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E'])
    parser.add_argument('--method', type=str, default='BNM', choices=['BNM', 'BFM', 'ENT','NO'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home', 'MI3DOR-1'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./webcam_reorgnized.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./amazon_reorgnized.txt', help="The target dataset path list")
    parser.add_argument('--test_dset_path', type=str, default='./amazon_reorgnized.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--print_num', type=int, default=100, help="print num ")
    parser.add_argument('--batch_size', type=int, default=128, help="number of batch size ")
    parser.add_argument('--num_iterations', type=int, default=1500, help="total iterations")
    parser.add_argument('--snapshot_interval', type=int, default=100, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="parameter for CDAN")
    parser.add_argument('--lambda_method', type=float, default=0.1, help="parameter for method")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--show', type=bool, default=True, help="whether show the loss functions")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config['CDAN'] = args.CDAN
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["show"] = args.show
    config["output_path"] = args.dset + '/' + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off, "lambda_method":args.lambda_method}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":args.test_dset_path, "batch_size":args.batch_size}}            # tb

    if config["dataset"] == "office":
        if ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0005 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0001 # optimal parameters       
        config["network"]["params"]["class_num"] = 31  
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "MI3DOR-1":
        config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 21
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    seed = random.randint(1,10000)
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    #uncommenting the following two lines for reproducing
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
