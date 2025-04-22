import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

from randomizations import StyleRandomization, ContentRandomization

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def zero_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = './alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_alexnet.classifier[6]
        self.__in_features = 4096

  def forward(self, x, pooling = False):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    # pdb.set_trace()
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    if pooling:
        n_views = 4
        x = x.view((int(x.shape[0] / n_views), n_views, x.shape[-1]))
        x = torch.max(x, 1)[0].view(x.shape[0], -1)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)

def Soft_Quantization(X, C, N_books, tau_q):
    L_word = int(C.size()[1]/N_books)
    x = torch.split(X, L_word, dim=1)
    c = torch.split(C, L_word, dim=1)
    for i in range(N_books):
        soft_c = nn.functional.softmax(squared_distances(x[i], c[i]) * (-tau_q), dim=-1)
        if i==0:
            Z = soft_c @ c[i]
        else:
            Z = torch.cat((Z, soft_c @ c[i]), dim=1)
    return Z



class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=21, drop = 0.5, style_stage=3):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.drop = drop
    self.style_stage = style_stage
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    
    self.layers_before = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
    self.style_randomization = StyleRandomization()
    self.dropout = nn.Dropout(self.drop)
    # self.content_randomization = ContentRandomization()
            
    # style-biased network
    '''style_layers = []
    if style_stage == 1:
        self.inplanes = 64
        style_layers += [self._make_layer(block, 64, layers[0])]
    if style_stage <= 2:
        self.inplanes = 64 * block.expansion
        style_layers += [self._make_layer(block, 128, layers[1], stride=2)]
    if style_stage <= 3:
        self.inplanes = 128 * block.expansion
        style_layers += [self._make_layer(block, 256, layers[2], stride=2)]
    if style_stage <= 4:
        self.inplanes = 256 * block.expansion
        style_layers += [self._make_layer(block, 512, layers[3], stride=2)]
    self.style_net = nn.Sequential(*style_layers)
            
    self.style_avgpool = nn.AdaptiveAvgPool2d(1)
    self.style_dropout = nn.Dropout(self.drop)
    self.style_fc = nn.Linear(512 * block.expansion, class_num)'''    

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features
    
    self.fc_t = nn.Sequential(nn.LayerNorm(bottleneck_dim),nn.Linear(bottleneck_dim, class_num))
    
    self.fc_out = nn.Linear(model_resnet.fc.in_features, 512)
    self.fc_spq = nn.Linear(512, 128)
    self.C = nn.Parameter(Variable((torch.randn(16, 256)).type(torch.float32), requires_grad=True))
    
    

  def forward(self, x, xt = None, sagnet = False, pooling = False):            # tb added pooling
    n_views = 4
    
    if sagnet:
        x0 = self.layers_before(x)
        x = self.layers_before(x)
        xt = self.layers_before(xt)
    
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if  i + 1 == self.style_stage:
                x = self.style_randomization(x, xt)
            xt = layer(xt)
            x = layer(x)
            x0 = layer(x0)
    
        x = self.avgpool(x)   
        x = x.view(x.size(0), -1)
        x = self.dropout(x)   
        x = x.detach()         
        Xa = self.bottleneck(x)
        
        x0 = self.avgpool(x0)   
        x0 = x0.view(x0.size(0), -1)
        x0 = self.dropout(x0)   
        x0 = x0.detach()         
        x0 = self.bottleneck(x0)
        
        xt = self.avgpool(xt)   
        xt = xt.view(xt.size(0), -1)
        xt = self.dropout(xt)           
        xt = self.bottleneck(xt)
        # pdb.set_trace()
        xt = xt.view((int(xt.shape[0] / n_views), n_views, xt.shape[-1]))
        xt = xt.detach()
        Xb = torch.max(xt, 1)[0].view(xt.shape[0], -1)
        
        ya = self.fc(Xa)
        yb = self.fc(Xb)
        y0 = self.fc(x0)
        
        return y0, ya, yb, x0, Xa, Xb   
    
    else:
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.bottleneck(x)
        
        if pooling:
            x = x.view((int(x.shape[0] / n_views), n_views, x.shape[-1]))
            x = torch.max(x, 1)[0].view(x.shape[0], -1)
        y = self.fc(x)
        
        return y, x
    
    
    '''
    for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
        if sagnet and i + 1 == self.style_stage:
            x = self.style_randomization(x, xt)
        if sagnet:
            xt = layer(xt)
        x = layer(x)
    
    x = self.avgpool(x)   
    x = x.view(x.size(0), -1)
    x = self.dropout(x)
        
    Xa = self.bottleneck(x) #128
    # Za = Soft_Quantization(Xa, self.C, 8, 5.0)
    ya = self.fc(Xa)
        

    if sagnet:
        xt = self.avgpool(xt)   
        xt = xt.view(xt.size(0), -1)
        xt = self.dropout(xt)
            
        Xb = self.bottleneck(xt) #128
        
        n_views = 4
        Xb = Xb.view((int(Xb.shape[0] / n_views), n_views, Xb.shape[-1]))   #[18,4,128]
        Xb = torch.max(Xb, 1)[0].view(Xb.shape[0], -1)   #[18,128]
        
        # Zb = Soft_Quantization(Xb, self.C, 8, 5.0)
        yb = self.fc(Xb)
        
            
        return ya, yb, Xa, Xb
    
    else:
        x = self.avgpool(x)   
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x_vp = x    
        x = self.bottleneck(x)
        y = self.fc(x)
    
        return x, y, x_vp

    
    elif spq:       
        # out = nn.functional.relu(self.fc_out(x)) #512
        # X = self.fc_spq(out) #128
        # Z = Soft_Quantization(X, self.C, 8, 5.0)
        
        # x = self.dropout(x)    
        # x = self.bottleneck(x)
        # y = self.fc_t(x)
        
        return ya, Xa, Za'''
       
    

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}

class VGGFc(nn.Module):
  def __init__(self, vgg_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bridge = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.bridge.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_vgg.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    z = self.bridge(x)
    return x, y, z

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.bridge.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

# For SVHN dataset
class DTN(nn.Module):
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout()
                )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500


    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    y = self.ad_layer2(x)
    y = self.relu2(y)
    y = self.dropout2(y)
    y = self.ad_layer3(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
