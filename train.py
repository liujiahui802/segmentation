import torch
from torch.nn.modules import Module

import torch.nn as nn
import numpy as np
import pickle
import deeplab_resnet 
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import *
import random
from docopt import docopt
import timeit
import os.path as osp
from get_label import get_label_batch


start = timeit.timeit()
docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file [default: data/list/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>          Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return j

def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
 return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def resize_conf_batch(conf, size):
    conf_resized = np.zeros((size,size,2,conf.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    h,w,chun = conf[:,:,0,:].shape
    labelVar = Variable(torch.from_numpy(conf[:,:,0,:].reshape((h,w,1,chun)).transpose( 3, 2, 0, 1)))
    conf_resized[:, :, 0, :] = interp(labelVar).data.numpy().transpose(2,3,1,0).reshape((size,size,conf.shape[3]))
    labelVar = Variable(torch.from_numpy(conf[:,:,1,:].reshape((h,w,1,chun)).transpose( 3, 2, 0, 1)))
    conf_resized[:, :, 1, :] = interp(labelVar).data.numpy().transpose(2,3,1,0).reshape((size,size,conf.shape[3]))   
    
    return conf_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)
   
def get_data_from_chunk_v2(chunk):
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']

    scale = random.uniform(0.5, 1.3) 
    #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me
    dim = int(scale*321)
    images = np.zeros((dim,dim,3,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    
    confidence = np.zeros((dim, dim, 2, len(chunk)))
    file_names = []
    for i,piece in enumerate(chunk):
        
        flip_p = random.uniform(0, 1)
        piece = (piece.split('/')[-1]).split('.')[0]
        file_names.append(piece)
        img_temp = cv2.imread(os.path.join(img_path,piece+'.jpg')).astype(float)
        img_temp = cv2.resize(img_temp,(321,321)).astype(float)
        img_temp = scale_im(img_temp,scale)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images[:,:,:,i] = img_temp

#        gt_temp = cv2.imread(os.path.join(gt_path,piece+'.png'))[:,:,0]
        gt_temp = np.array(Image.open(os.path.join(gt_path,piece+'.png')).convert(mode = 'P'), dtype = np.uint8)
        gt_temp[gt_temp == 255] = 0
        gt_temp = cv2.resize(gt_temp,(321,321) , interpolation = cv2.INTER_NEAREST)
        gt_temp = scale_gt(gt_temp,scale)
        gt_temp = flip(gt_temp,flip_p)
        gt[:,:,0,i] = gt_temp
        a = outS(321*scale)#41            ??????????????????????????
        b = outS((321*0.5)*scale+1)#21      ??????????????????????
        
#        con = np.load(osp.join('/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/Baseline/new_loss_train_data2map/SE5S1CRF_m1'  , '{}.npy'.format(piece)))
#        conf = np.zeros((2,321,321))
#        conf[0] = cv2.resize(con[0], (321, 321) , interpolation = cv2.INTER_NEAREST)
#        conf[1] = cv2.resize(con[1], (321, 321) , interpolation = cv2.INTER_NEAREST)
#        
#        confidence[:,:,0,i] = flip(scale_gt(conf[0] , scale), flip_p)
#        confidence[:,:,1,i] = flip(scale_gt(conf[1] , scale), flip_p)
#       
#    confidences = [resize_conf_batch(confidence , i) for i in [a,a,b,a]]
    labels = [resize_label_batch(gt,i) for i in [a,a,b,a]]               
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    
    return images, labels, file_names#, confidences



def loss_calc(out, label , gpu0):  #, confidence, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    
    confidence : batchsize * 2 * h * w
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()                       # log + soft max
    criterion = nn.NLLLoss2d()                     # -log
    out = m(out)

    return criterion(out,label)
    
#    label = label[:,:,0,:].transpose(2,0,1)
#    label = torch.from_numpy(label).long()
#    label = Variable(label).cuda(gpu0)
#
#    # for calculating difference metric
#    confidence = confidence.transpose(3,2,0,1)
#    batch , _ , h , w =  confidence.shape
#    diff = np.zeros((batch , h , w))
#    for h_i in range(h):
#        for w_i in range(w):
#            batch_fg_score = confidence[:, 0 , h_i, w_i]
#            batch_bg_score = confidence[:, 1 , h_i, w_i]
#            for batch_i in range(batch):
#                # idff function , need to be modified
#                if max(batch_fg_score[batch_i] ,batch_bg_score[batch_i]) ==0 :
#                    b_diff = float(np.abs(batch_fg_score[batch_i] - batch_bg_score[batch_i]))
#                else:
#                    b_diff = float(np.abs(batch_fg_score[batch_i] - batch_bg_score[batch_i])) / float(abs(max(batch_fg_score[batch_i] ,batch_bg_score[batch_i])))
#                #  < 0.4 we consider not high confidece as fg or bg enough , thus give a threshold and not learning in this case 
#                diff[batch_i , h_i, w_i] = b_diff
#    diff[diff <0.4] = 0.000
#    diff = torch.from_numpy(diff).long()
#    diff = Variable(diff).cuda(gpu0)
#    
#    criterion = Conf2dLoss()
#    loss = criterion(out,label,diff)
#
#
#    return loss


'''
inputs : batch_size x channels x h x w
labels : batch_size x 1 x h x w
'''
class Conf2dLoss(Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(Conf2dLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target ,diff):
        batch , h, w = diff.shape
        loss = float(0)
        for batch_i in range(batch):
            loss = loss + get_each_batch_loss(input[batch_i], target[batch_i], diff[batch_i])
        loss = Variable(loss.float(), requires_grad = True).cuda(gpu0)
        return loss

def get_each_batch_loss(input , target , diff):  # input shape 21*23*23 ,   target shape 23*23  , diff.shape 23 *23
    h , w = diff.shape
    fg_area = torch.sum(target != 0)
    bg_area = torch.sum(target == 0)
    huc = np.zeros(diff.shape)  
    for h_i in range(h):
        for w_i in range(w):
            huc[h_i][w_i] = input[target[h_i , w_i] , h_i , w_i]
    huc = np.log(huc)       
    huc = Variable(torch.from_numpy(huc).long()).cuda(gpu0)
    loss_1 = -(torch.sum(diff * huc * (label != 0))) / float(fg_area)
    loss_2 = -(torch.sum(diff * huc * (label == 0))) / float(bg_area)
    loss = loss_1 + loss_2

    return loss


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')

#CUDA_LAUNCH_BLOCKING=1
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))

saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
#saved_state_dict = torch.load('data/snapshots2/VOC12_scenes_20000.pth')

if int(args['--NoLabels'])!=21:
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if i_parts[1]=='layer5':
            saved_state_dict[i] = model.state_dict()[i]

model.load_state_dict(saved_state_dict)

max_iter = int(args['--maxIter']) 
batch_size = 1
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.eval() # use_global_stats = True

img_list = read_file(args['--LISTpath'])

data_list = []
for i in range(10):  # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
    np.random.shuffle(img_list)
    data_list.extend(img_list)

model.cuda(gpu0)
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

optimizer.zero_grad()
data_gen = chunker(data_list, batch_size)
print(batch_size)

criterion = nn.BCEWithLogitsLoss(size_average = True) 
criterion.cuda()

for iter in range(max_iter+1):
    chunk = data_gen.next()

    images, label, file_names= get_data_from_chunk_v2(chunk)
    images = Variable(images).cuda(gpu0)
#    print(images.data.size())
    
    labels = get_label_batch(file_names)
    labels = torch.from_numpy(np.ascontiguousarray(labels)).float()            #.long() for different lossfunction
    labels = Variable(labels.cuda())

    out = model(images)
#    print('---------- before pooling------------')
#    print(out[0].data.size())
#    print(out[1].data.size())
#    print(out[2].data.size())
#    print(out[3].data.size())
#    out[0] = nn.AvgPool2d(out[0].data.size[2:4])             #   GAP ,   GMP
#    out[1] = nn.AvgPool2d(out[1].data.size[2:4])
#    out[2] = nn.AvgPool2d(out[2].data.size[2:4])
#    out[3] = nn.AvgPool2d(out[3].data.size[2:4])
    
#            x.contiguous()
#        x = x.view(x.size(0) , -1)
#    print('===================  after pooling ==========')
#    print(out[0].data.size())
#    print(out[1].data.size())
#    print(out[2].data.size())
#    print(out[3].data.size())
#    print(label.data.shape)
    
    
    
#    loss = criterion(out[0] , labels)
    loss = loss_calc(out[0], label[0] ,gpu0)                   #loss
    iter_size = int(args['--iterSize']) 
    
    
    for i in range(len(out)-1):
        loss = loss + loss_calc(out[i+1],label[i+1]  ,gpu0)        #loss
#        loss = loss + criterion(out[i+1] , labels)
        
         
    loss = loss/iter_size
    loss.backward()

    if iter %1 == 0:
        print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

    if iter % iter_size  == 0:
        optimizer.step()
        lr_ = lr_poly(base_lr,iter,max_iter,0.9)
        print '(poly lr policy) learning rate',lr_
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        optimizer.zero_grad()

    if iter % 1000 == 0 and iter!=0:
        print 'taking snapshot ...'
        torch.save(model.state_dict(),'experiment/snapshots_seed/VOC12_scenes_'+str(iter)+'.pth')
end = timeit.timeit()
print 'time taken ', end-start
