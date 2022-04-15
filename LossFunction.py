# define for different loss functions
import torch
from torch import nn
from utils import tensor_to_onehot


def get_tp_fp_fn_tn(net_output, gt_onehot, axes=None):
    tp = net_output * gt_onehot  # [b,c,x,y,z]
    fp = net_output * (1 - gt_onehot)
    fn = (1 - net_output) * gt_onehot
    tn = (1 - net_output) * (1 - gt_onehot)

    if len(axes) > 0:
        tp = torch.sum(tp, axes, keepdim=False)  # [b,c]
        fp = torch.sum(fp, axes, keepdim=False)
        fn = torch.sum(fn, axes, keepdim=False)
        tn = torch.sum(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class CrossEntropyNDim(nn.CrossEntropyLoss):
    # output:[b,c,x,y,z]
    # target [b,x,y,z]
    def forward(self, output, target):
        # target=torch.argmax(target,dim=1,keepdim=False)#[b,x,y,z]
        loss = super(CrossEntropyNDim, self).forward(output, target)
        return loss


class DiceForSample(object):
    def __init__(self, smooth=1e-5):
        super(DiceForSample, self).__init__()
        self.smooth = smooth
        self.dice_for_per_class, self.dice_for_region, self.dice_loss = None, None, None
        self.tp, self.fp, self.fn = None, None, None
    def forward(self, output, target):
        # output:[b,c,x,y,z]
        # target [b,x,y,z]
        output = torch.softmax(output, dim=1)
        target = tensor_to_onehot(target, 4)
        target = target.to(torch.float32)
        self.tp, self.fp, self.fn, _ = get_tp_fp_fn_tn(output, target, axes=(2, 3, 4))  # [b,n]
    def forward_for_test(self, output, target):
        # output:[b,c,x,y,z]
        # target [b,x,y,z]
        output = tensor_to_onehot(output, 4)
        output = output.to(torch.float32)
        target = tensor_to_onehot(target, 4)
        target = target.to(torch.float32)
        self.tp, self.fp, self.fn, _ = get_tp_fp_fn_tn(output, target, axes=(2, 3, 4))  # [b,n]

    def get_dice_per_class(self,batch=True):
        dice = (2 * self.tp + self.smooth) / (2 * self.tp + self.fp + self.fn + self.smooth)
        if batch:
            dice=torch.mean(dice,dim=0,keepdim=False)
        return dice

    def get_dice_region(self,batch=True):
        # whole
        tp_whole = self.tp[:, 1] + self.tp[:, 2] + self.tp[:, 3]
        fp_whole = self.fp[:, 1] + self.fp[:, 2] + self.fp[:, 3]
        fn_whole = self.fn[:, 1] + self.fn[:, 2] + self.fn[:, 3]

        tp_core = self.tp[:, 1] + self.tp[:, 3]
        fp_core = self.fp[:, 1] + self.fp[:, 3]
        fn_core = self.fn[:, 1] + self.fn[:, 3]

        dice_whole = (2 * tp_whole + self.smooth) / (2 * tp_whole + fp_whole + fn_whole + self.smooth)
        dice_core = (2 * tp_core + self.smooth) / (2 * tp_core + fp_core + fn_core + self.smooth)
        # dice_enhance = (2 * self.tp[3] + self.smooth) / (2 * self.tp[3] + self.fp[3] + self.fn[3] + self.smooth)
        if batch:
            dice_whole=torch.mean(dice_whole,dim=0,keepdim=False)
            dice_core=torch.mean(dice_core,dim=0,keepdim=False)

        return torch.stack([dice_whole, dice_core], dim=0)

    def get_dice_loss(self,background=False):
        dice = (2 * self.tp + self.smooth) / (2 * self.tp + self.fp + self.fn + self.smooth)
        if not background:
            dice=dice[:,1:]
        loss = torch.tensor(1) - torch.mean(dice)
        return loss


class GenerizedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5,background=False):
        super(GenerizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.background=background

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        target = tensor_to_onehot(target, 4)
        target = target.to(torch.float32)
        tp, fp, fn, _ = get_tp_fp_fn_tn(output, target, axes=(2, 3, 4))  # [b,c]
        volume = torch.sum(target, dim=(2, 3, 4)) + 1e-6  # [b,c]
        tp = tp / volume
        fp = fp / volume
        fn = fn / volume


        generalized_dic = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        if not self.background:
            generalized_dic=generalized_dic[:,1:]
        loss = - torch.mean(generalized_dic)
        return loss


# class CrossEntropySoftDiceLoss(nn.Module):
#     def __init__(self):
#         super(CrossEntropySoftDiceLoss, self).__init__()
#         self.cross_entropy = CrossEntropyNDim()
#         self.softdice = SoftDiceLoss()
# 
#     def forward(self, output, target):
#         return self.cross_entropy(output, target) + self.softdice(output, target)


class CrossEntropyGeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyGeneralizedDiceLoss, self).__init__()
        self.cross_entropy = CrossEntropyNDim()
        self.generalized_dice = GenerizedDiceLoss()

    def forward(self, output, target):
        dice = self.generalized_dice(output, target)
        entropy = self.cross_entropy(output, target)
        return dice + entropy


class DeepSupervisedCrossEntropyGeneralizedDiceLoss(nn.Module):
    def __init__(self, loss_function,weights=(1/4,1/2,1)):
        super(DeepSupervisedCrossEntropyGeneralizedDiceLoss, self).__init__()
        self.loss_function = loss_function
        self.weights=weights

    def forward(self, output_list, target):
        assert isinstance(output_list, (tuple, list))
        loss_list = [self.loss_function(output_list[i], target) for i in range(len(output_list))]
        loss = loss_list[0]*self.weights[0]+loss_list[1]*self.weights[1]+loss_list[2]
        return loss
