import torch
import torch.nn as nn
import pdb
import jittor as jt

def varvar(x,mean):
    temp1 = x*(x-mean)
    temp2 = temp1.mean(dim=-1, keepdims=True)

    epsilon = 1e-8
    temp2 = torch.clamp(temp2, min=epsilon)  # Clamp input to prevent negative values
    var = torch.sqrt(temp2)
    # var = temp2.sqrt()
    return var

class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        # x,y = x.chunk(2,dim=0)
        # a=[]
        # for i in range(0,y.size()[0],4):
        #     a.append(i)
        # y = y[a,:,:,:]
        
        N, C, H, W = x.size()
        # pdb.set_trace()

        if self.training:
            x = x.view(N, C, -1)
            #[30,512,784,]
            mean = x.mean(-1, keepdims=True)
            var = varvar(x,mean)
            
           
            
            y = y.view(N, C, -1)
            #[30,512,1176,]
            meany = y.mean(-1, keepdims=True)
            vary = varvar(y,meany)
            
            
            x = (x - mean) / (var + self.eps).sqrt()

            mean = jt.Var(mean.data.cpu().numpy())
            var = jt.Var(var.data.cpu().numpy())
            meany = jt.Var(meany.data.cpu().numpy())
            vary = jt.Var(vary.data.cpu().numpy())
           
            
            idx_swap = jt.randperm(N)
            alpha = jt.rand(N, 1, 1)
            # if x.is_cuda:
            # alpha = alpha.cuda()
            one_minus_alpha = 1.0 - alpha
            meany_idx = jt.gather(meany, 0, idx_swap)
            vary_idx = jt.gather(vary, 0, idx_swap)
            
            # x = torch.from_numpy(x.data).cuda() 

            mean = alpha * mean + (one_minus_alpha) * meany_idx
            var = alpha * var + (one_minus_alpha) * vary_idx
            # print(type(mean))
            # input()
            mean = torch.from_numpy(mean.data).cuda()
            var = torch.from_numpy(var.data).cuda()

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class ContentRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x,y = x.chunk(2,dim=0)
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            y = y.view(N, C, -1)
            meany = y.mean(-1, keepdim=True)
            vary = y.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            y = (y - meany) / (vary + self.eps).sqrt()
            
            # idx_swap = torch.randperm(N)
            # y = y[idx_swap].detach()

            x = y * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x
