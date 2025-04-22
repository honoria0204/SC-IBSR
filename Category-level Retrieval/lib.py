from easydl import *
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
# from pynndescent import NNDescent, PyNNDescentTransformer
# from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def compute_dist(input_feature_source, input_feature):
    euclidean_dist = cdist(input_feature, input_feature_source).astype(np.float32)
    return euclidean_dist



def process_zero_value(tensor, nozero=True):
    if (tensor <= 0).sum() != 0:  # you <=0 de shu:
        if nozero:
            tensor[tensor <= 0] = 1e-8
        else:
            tensor[tensor <= 0] = 0
    return tensor
class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.to(output_device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def FeaturePreprocess(s4lp,source_feature_list, source_feature_list_category, source_labels, target_feature_list, args,
                      moving_feature_centeriod_t=False):
    if s4lp == 'all':

        NumS = len(source_feature_list)
        source_labels = torch.cat(source_labels, dim=0)
        target_feature = torch.cat(target_feature_list, dim=0)  # [36, 256]
        source_feature = torch.cat(source_feature_list, dim=0)  # [36, 256]
        ########################## l2 progress of target data
        target_feature = F.normalize(target_feature, dim=1, p=2)
        source_feature = F.normalize(source_feature, dim=1, p=2)
    elif s4lp == 'center':
        NumS = 21
        source_labels = torch.eye(21)
        # target_feature = torch.cat(target_feature_list, dim=0)
        source_feature = torch.zeros(21, 256)
        for i in range(21):
            source_feature[i] = torch.cat(source_feature_list_category[i], dim=0).mean(0)
        #####################  类中心  均值求法

        #####################   l2正则化
        target_feature = F.normalize(target_feature, dim=1, p=2)
        source_feature = F.normalize(source_feature, dim=1, p=2)

    return source_feature, source_labels, NumS, target_feature


def cg_solver(A, B, X0=None, rtol=1e-3, maxiter=None):
    n, m = B.shape
    if X0 is None:
        X0 = B
    if maxiter is None:
        maxiter = 2 * min(n, m)
    X_k = X0
    R_k = B - A.matmul(X_k)
    P_k = R_k
    stopping_matrix = torch.max(rtol * torch.abs(B), 1e-3 * torch.ones_like(B))
    for k in range(1, maxiter + 1):
        fenzi = R_k.transpose(0, 1).matmul(R_k).diag()
        fenmu = P_k.transpose(0, 1).matmul(A).matmul(P_k).diag()
        # fenmu[fenmu == 0] = 1e-8
        alpha_k = fenzi / fenmu
        X_kp1 = X_k + alpha_k * P_k
        R_kp1 = R_k - (A.matmul(alpha_k * P_k))
        residual_norm = torch.abs(A.matmul(X_kp1) - B)
        if (residual_norm <= stopping_matrix).all():
            break
        # fenzi[fenzi ==0] = 1e-8
        beta_k = (R_kp1.transpose(0, 1).matmul(R_kp1) / (fenzi)).diag()
        P_kp1 = R_kp1 + beta_k * P_k

        P_k = P_kp1
        X_k = X_kp1
        R_k = R_kp1
    return X_kp1


def lp(s4lp,GT_labels,source_feature_list, source_feature_list_category, source_labels, target_feature_list, args):
    # s4lp = 'all'
    source_feature, source_labels, NumS, target_feature = FeaturePreprocess(s4lp,source_feature_list,
                                                                            source_feature_list_category,
                                                                            source_labels, target_feature_list,
                                                                            args, moving_feature_centeriod_t=False)

    NumT = len(target_feature_list)
    all_feature = torch.cat((source_feature, target_feature), dim=0)  ### (Ns + Nt) * d  ==826
    target_label_initial = torch.zeros(NumT, 21)
    #.type(torch.DoubleTensor)  ## the initial state make no influence##########

    all_label = torch.cat((source_labels, target_label_initial), dim=0)  ### (Ns + Nt) * c
    for lpiteration in range(5):
        #print('lpiteration:', lpiteration)
        NumST = NumS + NumT

        dis='mul'
        # ======== nndescent
        if dis=='nn':
            nn_data = all_feature.detach().numpy()
            weight = torch.zeros(all_feature.size(0), all_feature.size(0))  # 826,826
            # A阵 !!!
            # question: graph 是全连接图吗?
            knn_indices, knn_value = NNDescent(nn_data, "cosine", {}, 20, random_state=np.random)._neighbor_graph
            # 826,20  826个节点，每个节点最近的20个节点

            weight.scatter_(1, torch.from_numpy(knn_indices), 1 - torch.from_numpy(knn_value).float())
            weight = weight + torch.t(weight)  # 对称一下  formula5
        if dis=='mul':
            weight = torch.matmul(all_feature, all_feature.transpose(0, 1))  ## N * N
            weight[weight < 0] = 0

            values, indexes = torch.topk(weight, 10)
            weight[weight < values[:, -1].view(-1, 1)] = 0

            weight = weight + torch.t(weight)
        weight.diagonal(0).fill_(0)  ### zero the diagonal

        # CG
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, NumST)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(NumST, 1)
        S = D1 * weight * D2
        # formula6
        ############ same with D3 = torch.diag(D_sqrt_inv)
        # S = torch.matmul(torch.matmul(D3, weight), D3)
        ################## A * X =  all_label
        ################################## nn solver, not the


        AlphaGraph = 0.5
        A = torch.eye(NumST) - AlphaGraph * S + 1e-8  # formula7 括号里面
        # .double()
        PredST = cg_solver(A, all_label)
        # F*
        # formula7  826,31   分类概率  CLS作用

        PredT = PredST[NumS:, :]
        _, LabelT = torch.topk(PredT, 1)
        #yy
        PredT = PredT / PredT.sum(1).view(-1, 1)  ### normalize  formula8 xiayihang
        PredT_nozeso = process_zero_value(PredT, nozero=True)  # 795,31
        PredT_zero = process_zero_value(PredT, nozero=False)  # 795,31

        ####### prediction confidence of the label propagation algorithm
        Instance_confidence = 1 + (PredT_zero * PredT_nozeso.log()).sum(1) / math.log(21)
        hpi = (PredT_zero * PredT_nozeso.log()).sum(1)
        prob_lp = PredT
        confi = Instance_confidence


        # pdb.set_trace()
        ### 1 - H(Z)/log(args.num_classes) #  torch.Tensor(size T number)
        # 795  置信度!!!!!!!!!!!!!!!!!!!!!  formula8

        acc_lp_label = torch.sum(torch.Tensor(GT_labels).long() == LabelT[:, 0]).item() / NumT

        if lpiteration == 4:
            acc_lp = acc_lp_label

        #print('acc_lp_label: ', acc_lp_label)
        ########## calculate the target pseudo center  target_feature
        target_feature_list_pseudo = []
        target_confidence_list = []
        for i in range(21):  # initialize
            target_feature_list_pseudo.append([])  ######### each for one categoty
            target_confidence_list.append([])

        for i in range(NumT):
            psuedo_label = LabelT[i]
            target_feature_list_pseudo[psuedo_label].append(target_feature[i].view(1, target_feature.size(1)))
            # 31个子list 每个里面放特征
            target_confidence_list[psuedo_label].append(Instance_confidence[i].view(1, -1))
            # 31个子list 每个里面放对应特征的置信度

        target_center_labels = torch.eye(21)  # 31,31
        target_center_feature = torch.zeros(21, 256)  # 31,2048
        ################### for predicted category with 0 samples
        noempty_list = []

        for i in range(21):
            if len(target_feature_list_pseudo[i]) == 0:
                continue
                # ？？？？？？？？？？？？？？？
            noempty_list.append(i)

            weight_for_this_category = torch.cat(target_confidence_list[i])
            weight_for_this_category = weight_for_this_category / weight_for_this_category.sum()
            # 每个样本对于这个类 的权重 31维
            target_center_feature[i] = (
                    torch.cat(target_feature_list_pseudo[i], dim=0) * weight_for_this_category).sum(0)
            # 类中心 prototype   ###################################
        target_center_labels = target_center_labels[noempty_list]
        # 31,31标准onehot
        target_center_feature = target_center_feature[noempty_list]
        # 31,2048 prototype

        if len(noempty_list) != 21:
            print('noempty', noempty_list)

        target_center_feature = F.normalize(target_center_feature, dim=1, p=2)

        all_feature = torch.cat((target_center_feature, all_feature), dim=0)

        all_label = torch.cat((target_center_labels, all_label), dim=0)

        # 更新all_feature 和 all_label nums 和 numt
        NumS = NumS + len(noempty_list)  # 31--->62--->···

    # lp_pse = LabelT.numpy().tolist()
    lp_pse = torch.reshape(LabelT, (-1, 36))
    lp_pse = lp_pse.squeeze(0)


    return lp_pse#, LabelT, NumT, acc_lp, prob_lp, confi


class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def SM(s_feature, t_feature, y_s, y_t, self_s_centroid, self_t_centroid, n_class):
    decay = 0.3
    self_s_centroid = self_s_centroid.cuda()
    self_t_centroid = self_t_centroid.cuda()

    self_MSEloss, self_BCEloss = nn.MSELoss(), nn.BCEWithLogitsLoss(reduction='mean')
    self_MSEloss, self_BCEloss = self_MSEloss.cuda(), self_BCEloss.cuda()

    n, d = s_feature.shape

    # get labels
    # s_labels, t_labels = y_s, torch.max(y_t, 1)[1]
    s_labels, t_labels = y_s, y_t

    # image number in each class
    ones = torch.ones_like(s_labels, dtype=torch.float)
    zeros = torch.zeros(n_class)

    zeros = zeros.cuda()
    s_n_classes = zeros.scatter_add(0, s_labels, ones)
    t_n_classes = zeros.scatter_add(0, t_labels, ones)

    # image number cannot be 0, when calculating centroids
    ones = torch.ones_like(s_n_classes)
    s_n_classes = torch.max(s_n_classes, ones)
    t_n_classes = torch.max(t_n_classes, ones)

    # calculating centroids, sum and divide
    zeros = torch.zeros(n_class, d)

    zeros = zeros.cuda()
    s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
    t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
    current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(n_class, 1))
    current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(n_class, 1))

    # Moving Centroid
    decay = decay
    s_centroid = (1 - decay) * self_s_centroid + decay * current_s_centroid
    t_centroid = (1 - decay) * self_t_centroid + decay * current_t_centroid
    semantic_loss = self_MSEloss(s_centroid, t_centroid)
    # self_s_centroid = s_centroid.detach()
    # self_t_centroid = t_centroid.detach()

    return semantic_loss, self_s_centroid, self_t_centroid

class OPLCEN(nn.Module):
    def __init__(self, gamma=0.5):
        super(OPLCEN, self).__init__()
        self.gamma = gamma

    def forward(self, features, cen_feas, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        cen_feas= F.normalize(cen_feas, p=2, dim=1)
        labels = labels[:, None]  # extend dim
        cenlabels = torch.range(0,20).long().to(device)
        hang = cenlabels.expand(features.size(0),21)
        lie = labels.expand(features.size(0),21)
        mask_pos = torch.eq(hang,lie).byte().to(device)
        mask_neg = ~mask_pos
        mask_pos = mask_pos.float()
        mask_neg = mask_neg.float()

        # mask_pos = mask.masked_fill(eye, 0).float()
        # mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, cen_feas.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

# yy
def entropyyy(x, eps=1e-5):
    # p = F.softmax(x, dim=-1)
    line_ent = torch.zeros((36,1))
    for i in range(36):
        pi = x[i, :]
        ent_i = -torch.sum(pi * torch.log(pi + eps))
        line_ent[i, :] = ent_i
        # print('ent_i:', ent_i)
    # entropy = -torch.mean(torch.sum(p * torch.log(p ), 1))
    return line_ent