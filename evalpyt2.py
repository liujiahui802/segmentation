import scipy
from scipy import ndimage
import cv2
import numpy as np
import os.path as osp
import sys
#sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
#import caffe
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet 
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image

from docopt import docopt
from util import make_palette
from crf import dense_crf
from get_label import get_label_batch

import scipy.ndimage as ndi
from skimage import measure,color
import matplotlib.pyplot as plt
from skimage import morphology

from gather import statis, cluster
from update import sal_seed1 , seed_dense_crf

from pydensecrf.utils import unary_from_labels
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""
palette = make_palette(20).reshape(-1)
args = docopt(docstr, version='v0.1')
print args

max_label = int(args['--NoLabels'])-1                           # labels from 0,1, ... 20(for VOC) 
def fast_hist(a, b, n):                                         # a for gt , b for predicted output.flatten()
    k = (a >= 0) & (a < n)                                      # pixel numbers for all class in gt, return array with True and False , n = 21 , leave out 255 ,  k is with size img_h * img_width
    
#    print('----------------------------------------')
#    print(k.shape)
#    print(a[k].shape)
#    print((n * a[k].astype(int) + b[k]).max())
#    print((n * a[k].astype(int) + b[k]).shape)  
#    print(b[k].max())
#    print((n*a[k]).max())
#    aa = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)   
#    # minimum bin number is  21*21  = 441, np.bincount to calculate the number of element in 0 - (bin-1)
#    # why 
#    aa = aa.reshape(n,n)   # shape (441,) to (21,21)
#    print(aa.shape)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)       # in a[k] , a and k are all array, chose element in a when corresponding place in k is True , to construct a new array

def error_analyse(hist):
    fg_error_list = [] 
    h , w = hist.shape
    for i in range(h):
        if i != 0:
            fg_error = []
            ei0 = float(hist[i][0]) / float(np.sum(hist[i]))
            eie = float(np.sum(hist[i]) - hist[i][0] - hist[i][i]) / float(np.sum(hist[i]))
            fg_error.append([ei0 , eie])
            fg_error_list.append(fg_error) 
    
    return fg_error_list

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label+1,))                                          # 21 class for iou rate
    for j in range(max_label+1):                                          # for each class in 21 , GT_idx_j may be 0
        x = np.where(pred==j)                                                  # for place axis x, y
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))    
        # x[0] is the x axis , y  is the y axis ,  set return unrepeated all x,y tuple axis list
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)                          # set calculation for intersevtion and union with class pixel axises
         
        if len(GT_idx_j)!=0:                                  # only  calculate for classes has in gt
            count[j] = float(len(n_jj))/float(len(u_jj))
    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 
    
    return Aiou                                                   # average iou for all  classes whitch is in gt

# remove m1m2m3crf combination's small area , which can remove some invalid samll class area ,and valid class small area , for last modify
def remove_small(mask ,  minarea = 150):
    mask_cls = np.unique(mask)
    new = np.zeros((mask.shape[0] , mask.shape[1]))

    for cls in mask_cls:
        if cls != 0:
            labels = measure.label(mask==cls , connectivity = 2)
            dst = morphology.remove_small_objects(labels , min_size = 150 ,  connectivity = 2)
            labels = measure.label(dst, connectivity = 2)   
            prop = measure.regionprops(labels)

            for idx, idx_region in enumerate(prop):         # whether including bg for label 0, no
                for pixel_coord in idx_region.coords:
                    new[pixel_coord[0]][pixel_coord[1]] = cls
    return new.astype(np.uint8)

def add_valid(output , m1_path , gt_cls):
    m1 = np.array(Image.open(m1_path).convert(mode = 'P'), dtype = np.uint8)
    out_cls = np.unique(output)
    for cls in out_cls:
        if cls not in gt_cls:
            output[np.logical_and(output==0 , m1==cls)] = cls
    return output
    


def m2_m3(m2_path , m3_path , m2_crf_path , m3_crf_path , seed_path):
    m2 =  np.array(Image.open(m2_path).convert(mode = 'P'), dtype = np.uint8)
    m3 =  np.array(Image.open(m3_path).convert(mode = 'P'), dtype = np.uint8)
    m2_crf = np.array(Image.open(m2_crf_path).convert(mode = 'P'), dtype = np.uint8)
    m3_crf = np.array(Image.open(m3_crf_path).convert(mode = 'P'), dtype = np.uint8)
    output = np.zeros((4 , m2.shape[0] , m2.shape[1]))
    output[0] = m2
    output[1] = m3
    output[2] = m2_crf
    output[3] = m3_crf
    mask = np.zeros( (m2.shape[0] , m2.shape[1]))
    
    for x in range(m2.shape[0]):
        for y in range(m2.shape[1]):
            m2m3_arr = np.zeros((2))            
            value_arr = np.zeros((4))
            for i in range(4):
                value_arr[i] = output[i][x][y]
            for i in range(2):
                m2m3_arr[i] = output[i][x][y]
            m2m3_uni = np.unique(m2m3_arr)
            cls_uni = np.unique(value_arr) 
            
            # for algorithme of union , no conflict
            if len(cls_uni)==1:
                mask[x][y] = cls_uni[0]

            if len(cls_uni) == 2 and 0 in cls_uni:
                if cls_uni[0]==0:
                    mask[x][y] = cls_uni[1]
                else:
                    mask[x][y] = cls_uni[0]
            # for conflict situation
            else:
                if len(m2m3_uni) == 1:
                    mask[x][y] = m2m3_uni[0]
                else:
                    mask[x][y] = 0
    # overwrite mask with high confidence seed3 , small area right onto the object               
    if seed_path  != None:
        seed = np.array(Image.open(seed_path).convert(mode = 'P'), dtype = np.uint8)
        seed_cls = np.unique(seed)
        for cls in seed_cls:
            if cls != 0:
                mask[seed == cls] = cls
    return mask.astype(np.uint8)
            
def m1_m2_m3(m1_path, m2_path , m3_path , m2_crf_path , m3_crf_path , seed_path):
    m1 = np.array(Image.open(m1_path).convert(mode = 'P'), dtype = np.uint8)
    m2 =  np.array(Image.open(m2_path).convert(mode = 'P'), dtype = np.uint8)
    m3 =  np.array(Image.open(m3_path).convert(mode = 'P'), dtype = np.uint8)
    m2_crf = np.array(Image.open(m2_crf_path).convert(mode = 'P'), dtype = np.uint8)
    m3_crf = np.array(Image.open(m3_crf_path).convert(mode = 'P'), dtype = np.uint8)
    output = np.zeros((5 , m2.shape[0] , m2.shape[1]))
    output[0] = m2
    output[1] = m3
    output[2] = m2_crf
    output[3] = m3_crf
    output[4] = m1
    mask = np.zeros( (m2.shape[0] , m2.shape[1]))
    
    for x in range(m2.shape[0]):
        for y in range(m2.shape[1]):
            m2m3_arr = np.zeros((2))            
            value_arr = np.zeros((5))
            for i in range(5):
                value_arr[i] = output[i][x][y]
            for i in range(2):
                m2m3_arr[i] = output[i][x][y]
            m2m3_uni = np.unique(m2m3_arr)
            cls_uni = np.unique(value_arr)
 
            # for algorithme of union , no conflict
            if len(cls_uni)==1:
                mask[x][y] = cls_uni[0]

            if len(cls_uni) == 2 and 0 in cls_uni:
                if cls_uni[0]==0:
                    mask[x][y] = cls_uni[1]
                else:
                    mask[x][y] = cls_uni[0]
            # for conflict situation
            else:
                if len(m2m3_uni) == 1:
                    mask[x][y] = m2m3_uni[0]
                else:
                    mask[x][y] = 0
                    
    # overwrite mask with high confidence seed3 , small area right onto the object               
    if seed_path  != None:
        seed = np.array(Image.open(seed_path).convert(mode = 'P'), dtype = np.uint8)
        seed_cls = np.unique(seed)
        for cls in seed_cls:
            if cls != 0:
                mask[seed == cls] = cls
                
    return mask.astype(np.uint8)               

def read_initiaL_sal(file_name):
    file_name = file_name + '.jpg'
    data = np.array(Image.open(osp.join('/home/hhhhhhhhhh/desktop/project_seg/new_saliency/test_output',file_name)), dtype = np.uint8) 
    h , w = data.shape
    
    for h_idx in range(h):
    	for w_idx in range(w):
    		if data[h_idx][w_idx] >= 100:
    			data[h_idx][w_idx] = 1
    		else:
    			data[h_idx][w_idx] = 0
    #print(data.max(0).max(0))
    return data


    
torch.cuda.set_device(0)

gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapPrefix = args['--snapPrefix'] 
gt_path = args['--testGTpath']
img_list = open('data/list/val.txt').readlines()

for iter in range(20,21):                            #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
    print('use model if {}000.pth iteration times.......'.format(iter))    
    saved_state_dict = torch.load(os.path.join('/home/hhhhhhhhhh/desktop/pytorch-deeplab-resnet-master/experiment/snapshots_seed',snapPrefix+str(iter)+'000.pth'))                      
    if counter==0:
	print snapPrefix
    counter+=1
    model.load_state_dict(saved_state_dict)

    hist = np.zeros((max_label+1, max_label+1))
    pytorch_list = []
    
    nocrf_pytorch_list = []
    nocrf_hist = np.zeros((max_label+1, max_label+1))
# for dataset clustering    
    crf_mask_dict = {}
    input4cluster = []
    class_labels = []
    for idx , i in enumerate(img_list):
        print('processing...  {}  images'.format(len(img_list)))
        print(idx)
        print(i[:-1])
#        print(os.path.join(im_path,i[:-1]+'.jpg'))
        idx_label = get_label_batch([i[:-1]])[0]          # size is (1,20)-> (20, )  no bg
        gt_cls = (np.where(idx_label == 1)[0] +1 ).tolist()
        gt_cls.append(0)
        
        
        img_path = osp.join('/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/JPEGImages',i[:-1]+'.jpg')
        img_arr = np.array(torch.from_numpy(np.ascontiguousarray((np.array(Image.open(img_path), dtype = np.uint8)).transpose(2,0,1))).float(), dtype = np.uint8)
        img = np.zeros((513,513,3));
        
        #############################################   for m2 m3 seed combination ########################################
#        seed_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/seed3/{}.png'.format(i[:-1])
#        m1_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/seed5_sal1/{}.png'.format(i[:-1])
#        m2_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/SE3S1CLSm2/{}.png'.format(i[:-1])
#        m3_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/SE3S1CLSm3/{}.png'.format(i[:-1])
#        m2_crf_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/SE3S1CLSm2_crf/{}.png'.format(i[:-1])
#        m3_crf_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/SE3S1CLSm3_crf/{}.png'.format(i[:-1])
##        
#        output = m2_m3(m2_path , m3_path , m2_crf_path ,m3_crf_path, seed_path = None)
#        output = m1_m2_m3(m1_path ,m2_path , m3_path , m2_crf_path ,m3_crf_path , seed_path = None)
##        
#        # remove small area in last
#        output = remove_small(output ,  minarea = 150)
        
        # use m1 seed_sal when there is no invalid class
#        output = add_valid(output , m1_path , gt_cls)
        ##############################   img and gt input     #########################################################
#        print(os.path.join(im_path,i[:-1]+'.jpg'))
        img_temp = cv2.imread(os.path.join(im_path,i[:-1]+'.jpg')).astype(float)    #  img_size * 3   BGR, img_size < 513
        img_ = img_temp.astype(np.uint8)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp    # put image into a larger numpy which has shape of 513*513
        gt = np.array(Image.open(os.path.join(gt_path,i[:-1]+'.png')).convert(mode = 'P'), dtype = np.uint8)
##        #gt[gt==255] = 0
#        print('--------------  before-----------------------')
#        print(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0).data.size())   # 1 3 513 513
        output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))   # np.newaxis add a batch_size = 1 axis , and inout img size is 1*513*513 *3  - >  1 * 3 * 513 * 513 ,  # output is a list with each element shape is 1*21*65*65 , 1*21*65*65 , 1*21*33*33 , 1*21*65*65
#        print('===================== after========================')        
#        print(output[3][0].data.size()) 
#        print(output[3][0].max(0))               #argmax max and class
#        print(output[3][0].max(1)[0].max(1)[0])  # # max per map
#
#        print(output[3][0])
        
        interp = nn.UpsamplingBilinear2d(size=(513, 513))            # upsample to 513 * 513 size , and only chose the last element in original output list  ??????? in net       
        output = interp(output[3]).cpu().data[0].numpy()
        output = output[:,:img_temp.shape[0],:img_temp.shape[1]]    # 21 * img.shape , # chose the original image size part in a larger image size of 513 * 513 , from top ans left ,  #  21 * img_size is the output shape , !!!!!!!!!!!!!!!!!!!!! and value are not normalized yet
        
#        ######################################       crf    ###################################################

        
##########################################################################################################################################################
        
# normalized output for threshold
#        nor_output = np.zeros(output.shape)
#        for idx in range(output.shape[0]):
#            max_per = output[idx].max()
#            if max_per <= 0 :
#                max_per = 1
#            nor_output[idx] = output[idx] / float(max_per)
        
            
 #        print(output.max(1).max(1)) # noramalize all map according to the max value for all map   , max() for all max , .max(1).max(1) for each map max  # and relative bigger and smaller value hasn't changed            
        max_all = output.max()   
        for idx in range(output.shape[0]): 
            output[idx] = output[idx] / float(max_all)
        nocrf_output = output
        output = dense_crf(img_, output)
        crf_mask =  np.argmax(output,axis = 0)

# for gathering algorithm
#        mask = statis(nocrf_output, crf_mask, i[:-1])
#        mask = cluster(nocrf_output, crf_mask , i[:-1] , mode = 'kmeans')
        
        
#        nocrf_output = mask

#        img_arr = np.array(Image.open(img_path) , dtype = np.uint8)        
#        output = seed_dense_crf(img_arr , mask)        
        

#        ##########################   no_remove   ######################
#        thre = 0.3
        output_heat = output.transpose(1,2,0)    # img_size * 21
        output = np.argmax(output_heat,axis = 2)   # img_size= mask           ??????????????????????????   when making mask , can use image label or not
# for threshold        
#        for h_i in range(output.shape[0]):
#            for w_i in range(output.shape[1]):
#                cls = output[h_i][w_i]
#                if cls != 0:
#                    if nor_output[cls][h_i][w_i] < thre:
#                        output[h_i][w_i] = 0
#        sal = read_initiaL_sal(i[:-1])
#        no_bg_gt_cls = np.unique(output).tolist()
#        no_bg_gt_cls.remove(0)
#        output = sal_seed1(sal, output , img_arr ,output_heat.transpose(2,0,1) , no_bg_gt_cls )
#        output = seed_dense_crf(img_arr.transpose(1,2,0).copy(order='C') , output)
        ##########################################################################
        nocrf_output_heat = nocrf_output.transpose(1,2,0)    # img_size * 21
        nocrf_output = np.argmax(nocrf_output_heat,axis = 2)
        
        
# for threshold        
#        for h_i in range(nocrf_output.shape[0]):
#            for w_i in range(nocrf_output.shape[1]):
#                cls = nocrf_output[h_i][w_i]
#                if cls != 0:
#                    if nor_output[cls][h_i][w_i] < thre:
#                        nocrf_output[h_i][w_i] = 0        
#        nocrf_output = sal_seed1(sal, nocrf_output, img_arr, nocrf_output_heat.transpose(2,0,1) , no_bg_gt_cls)
#        nocrf_output = seed_dense_crf(img_arr.transpose(1,2,0).copy(order='C') , nocrf_output)
        
         #######################  remove invalid classes  ########################

        
        # remove invalid class in output , turn out most valid area are beated by bg value
#        all_cls = np.unique(output)
#        for cls in all_cls:
#            if cls not in gt_cls:
#                output[output == cls] = 0
#
#         ##  no crf       
#        all_cls = np.unique(nocrf_output)
#        for cls in all_cls:
#            if cls not in gt_cls:
#                nocrf_output[nocrf_output == cls] = 0



############################################# show eval resluts   #########################################################
        if args['--visualize']:
            img = Image.open(os.path.join(im_path,i[:-1]+'.jpg'))
#            img.putalpha(255)
            
#            gt_show = Image.fromarray(gt)       # save image with palatte and visualize
#            gt_show.putpalette(palette)
#            gt_show.putalpha(255)
#            gt_arr = np.array(gt_show, dtype = np.uint8)
#            
            fake_mask = Image.fromarray(np.uint8(nocrf_output))
            fake_mask.putpalette(palette)
#            
            crf_fake_mask = Image.fromarray(np.uint8(output))
            crf_fake_mask.putpalette(palette)
#            
##            fake_mask.putalpha(255)
#            
##            img_fake_mask = Image.blend(img , fake_mask , 0.625)
##            img_fake_mask_arr = np.array(img_fake_mask, dtype = np.uint8)
##            
##            gt_seed_img_arr = np.concatenate([gt_arr,img_fake_mask_arr ], 1)
##            gt_seed_img = Image.fromarray(gt_seed_img_arr)
#            
            save_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/Baseline/m2/seed_only'  
##            crf_save_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/evaluate/test_crf11'
##            gt_seed_img.save(os.path.join(save_path, '{}.png').format(i[:-1])) 
            fake_mask.save(os.path.join(save_path, '{}.png').format(i[:-1]))
            crf_fake_mask.save(os.path.join(save_path, '{}-crf.png').format(i[:-1]))
             

        iou_pytorch = get_iou(output,gt)    # average iou for each image on labels in gt , not all 21 labels      
        pytorch_list.append(iou_pytorch)
        hist += fast_hist(gt.flatten(),output.flatten(),max_label+1) 
#        
        nocrf_iou_pytorch = get_iou(nocrf_output,gt)
        nocrf_pytorch_list.append(nocrf_iou_pytorch)
        nocrf_hist += fast_hist(gt.flatten(),nocrf_output.flatten(),max_label+1) 
#        
    # hist is shape of 21 * 21
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('has crf results ------------------------------------------------  ')
    print('class iou ',iter , "---" , miou)      
    print 'evalpyt2-pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    print 'evalpyt1-pytorch',iter, np.sum(np.asarray(pytorch_list))/len(pytorch_list)   # average 
#    error_list = error_analyse(hist)
#
#    print(error_list)  
    
#    # nocrf output
    miou = np.diag(nocrf_hist) / (nocrf_hist.sum(1) + nocrf_hist.sum(0) - np.diag(nocrf_hist))
    print('no crf results-------------------------------------------------')    
    print('class iou ',iter , "---" , miou)      
    print 'evalpyt2-pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    print 'evalpyt1-pytorch',iter, np.sum(np.asarray(nocrf_pytorch_list))/len(nocrf_pytorch_list)   # average 
    
#    error_list = error_analyse(nocrf_hist)    
#    print(error_list)
