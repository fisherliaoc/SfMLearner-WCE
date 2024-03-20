import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn.init import xavier_uniform_, xavier_normal_, zeros_, ones_, kaiming_normal_, kaiming_uniform_, uniform_
from gumbel_softmax import *

from utils import euler2mat
from inverse_warp import mat2euler

from mobilevit import *

def conv(in_planes, out_planes, kernel_size=3, padding=None, bias=False):
    if padding is None:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2, bias=bias),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=2, bias=bias),
            nn.ReLU(inplace=True)
        )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def predict_exp(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
    )

def truncated_normal_(tensor,mean=0):
    with torch.no_grad():
        stdd = 0.05*math.sqrt(min((global_n_iter2.n_iter+1)/40000., 1))
        size = tensor.shape
        tmp = tensor.new_empty(size).normal_(mean=mean, std=stdd)
        while True:
            cond = torch.logical_or(tmp < mean - 2*stdd, tmp > mean + 2*stdd)
            if not torch.sum(cond):
                break
            tmp = torch.where(cond, tensor.new_empty(size).normal_(mean=mean, std=stdd), tmp)
        return tmp

class LnWithGN_N2(nn.Module):
    def __init__(self, channelsize, eps=1e-3, affine=True):
        super(LnWithGN_N2, self).__init__()
        self.channelsize = channelsize
        self.affine = affine
        self.eps = eps
        self.momentum_adp = 1.

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(self.channelsize))
            self.beta = nn.Parameter(torch.Tensor(self.channelsize))

    def forward(self, x):
        # print(torch.max(x))

        mean = torch.mean(x,dim=(2, 3), keepdim=True)
        var = torch.var(x,dim=(2, 3), keepdim=True)

        if self.training:
            mean = mean *(1.0 + truncated_normal_(mean))
            var = var *(1.0 + truncated_normal_(var))

        y = (x - mean.expand( x.size() )) / torch.sqrt(var.expand( x.size() ) + self.eps)

        if self.affine:
            shape = [1] + [self.channelsize] + [1] * (x.dim() - 2) # [1, C, 1, 1]

            y = self.gamma.view(*shape).expand( x.size() )*y + \
                self.beta.view(*shape).expand( x.size() )
        return y


class PoseExpNet_WCE(nn.Module):

    def __init__(self, input_image_c=4, learn_pose_scale=True, r_scale=0.01, tl_scale=0.01, intri_scale=1, output_exp=False, is_unitycam=False, dist_truth=False, **kwargs):
        super(PoseExpNet_WCE, self).__init__()
        self.input_image_c = input_image_c
        self.intri_scale = intri_scale
        self.learn_pose_scale = learn_pose_scale
        self.r_scale = r_scale
        self.tl_scale = tl_scale

        self.random_rot = random_rot
        self.img_hw = 256

        self.is_unitycam = is_unitycam
        self.dist_truth = dist_truth
        self.vit = mobilevit_s(image_size=(self.img_hw, self.img_hw), num_classes=11, input_image_c=self.input_image_c)

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if self.learn_pose_scale:
            self.trans_scale = nn.Parameter(tl_scale*torch.ones(1, requires_grad=True))
            self.rots_scale = nn.Parameter(r_scale*torch.ones(1, requires_grad=True))

    def sub_forward(self, target_image, ref_img, mask):
        image_height = ref_img.size(2)
        image_width = ref_img.size(3)

        assert(image_height == self.img_hw)
        assert(image_width == self.img_hw)

        if self.input_image_c == 4:
            t_i = target_image
            r_i = ref_img
        elif self.input_image_c == 3:
            t_i = (target_image[:,0:3]-0.5)/0.5
            r_i = (ref_img[:,0:3]-0.5)/0.5

        bottleneck, x3, x2, x1 = self.vit(t_i, r_i)

        if self.learn_pose_scale:
            self.rots_scale = nn.Parameter(self.rots_scale.clamp(min=1e-4))
            rots_scale = self.rots_scale
            self.trans_scale = nn.Parameter(self.trans_scale.clamp(min=1e-4))
            trans_scale = self.trans_scale
        else:
            rots_scale = self.r_scale
            trans_scale = self.tl_scale

        angles_estimated = rots_scale*bottleneck[:,:3]
        rotation_estimated = euler2mat(angles_estimated)
        translation_estimated = trans_scale*(bottleneck[:,3:6]).unsqueeze(1)
        pose_mat = torch.cat((rotation_estimated, translation_estimated), 1).unsqueeze(1)

        scales_tensor = torch.tensor([[image_width, image_height]], requires_grad=False).to(bottleneck.device)

        focal_lengths_ratio = F.softplus(bottleneck[:,6:8])

        focal_lengths = focal_lengths_ratio * scales_tensor.expand(focal_lengths_ratio.size(0),-1)
        foci = torch.diag_embed(focal_lengths)

        if intri_index is not None:
            offsets = self.intrinsics_u[intri_index.squeeze(1),2:]*0.5
        else:
            offsets = torch.sigmoid(bottleneck[:,8:10])
        offsets = offsets * scales_tensor.expand(offsets.size(0),-1)
        intrinsic_mat = torch.cat((foci, offsets.unsqueeze(-1)), 2)
        if self.dist_truth:
            distort_coeffs = 0.8*math.pi/6*torch.ones_like(bottleneck[:,10:11])
        elif self.is_unitycam:
            distort_coeffs = 0.7*math.pi*torch.sigmoid(bottleneck[:,10:11])
        else:
            distort_ratio = torch.sigmoid(bottleneck[:,10:11])
            maxx = torch.maximum(intrinsic_mat[:,0,2], scales_tensor[0,0].expand(intrinsic_mat.size(0))-intrinsic_mat[:,0,2])
            maxy = torch.maximum(intrinsic_mat[:,1,2], scales_tensor[0,1].expand(intrinsic_mat.size(0))-intrinsic_mat[:,1,2])
            maxx /= intrinsic_mat[:,0,0]
            maxy /= intrinsic_mat[:,1,1]
            maxrr = maxx * maxx + maxy * maxy
            min_distortion = - 0.7 / maxrr
            distort_coeffs = min_distortion.view(-1,1).detach()*distort_ratio

        last_row = torch.tensor([[[0.0, 0.0, 1.0]]], requires_grad=False).to(bottleneck.device)
        intrinsic_mat = torch.cat((intrinsic_mat, last_row.expand(bottleneck.size(0), -1, -1)), 1)

        with torch.no_grad():
            euler_pose = mat2euler(rotation_estimated)
            pose = torch.cat((translation_estimated, euler_pose.unsqueeze(1)), 2)

        exp_mask3 = torch.cat((gumbel_softmax(x3[:,:2])[:,:1], gumbel_softmax(x3[:,2:])[:,:1]), 1)
        exp_mask2 = torch.cat((gumbel_softmax(x2[:,:2])[:,:1], gumbel_softmax(x2[:,2:])[:,:1]), 1)
        exp_mask1 = torch.cat((gumbel_softmax(x1[:,:2])[:,:1], gumbel_softmax(x1[:,2:])[:,:1]), 1)

        return intrinsic_mat, distort_coeffs, exp_mask1[:,:2], exp_mask2[:,:2], exp_mask3[:,:2], pose_mat, pose


    def forward(self, target_image, ref_imgs):

        b = target_image.size(0)

        intrinsic_mat_f, distort_coeffs_f, exp_mask1_f, exp_mask2_f, exp_mask3_f, pose_mat_f, pose_f = self.sub_forward(torch.cat((target_image,ref_imgs[0]),0), torch.cat((ref_imgs[0],target_image),0))
        exp_mask1, exp_mask1t = torch.chunk(exp_mask1_f, 2, 0)
        exp_mask2, exp_mask2t = torch.chunk(exp_mask2_f, 2, 0)
        exp_mask3, exp_mask3t = torch.chunk(exp_mask3_f, 2, 0)
        pose_mat, re_pose_mat = torch.chunk(pose_mat_f, 2, 0)
        pose, re_pose = torch.chunk(pose_f, 2, 0)
        
        intrinsic_mat = intrinsic_mat_f.view(2,b,3,3).mean(0)
        distort_coeffs = distort_coeffs_f.view(2,b,1).mean(0)

        return [exp_mask1, exp_mask2, exp_mask3], [exp_mask1t, exp_mask2t, exp_mask3t], pose, re_pose, pose_mat, re_pose_mat, intrinsic_mat, distort_coeffs