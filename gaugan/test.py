from re import X
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import time
import datetime
import sys
import cv2
import time

from network.generator import SPADEGenerator
from network.models import *
from datasets import *
from network.loss import *
# from tensorboardX import SummaryWriter
from Diffaug import * 

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1
import random
seed = random.randint(1, 100000)
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", type=int, default=146, help="epoch to start training from")
# parser.add_argument("--input_path", type=str, default="/data/lfq/gaugan/datasets/val")
parser.add_argument("--input_path", type=str, default="./Bbang")
parser.add_argument("--output_path", type=str, default="./b_results")
parser.add_argument("--model_path", type=str, default="./final_generator")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--label_c", type=int, default=30, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64)


opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img


# Configure dataloaders
transform_label = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.NEAREST),
    transform.ToTensor(),
    # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

transform_img = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

val_dataloader = TestImageDataset(opt.input_path, transform_label=transform_label, transform_img=transform_img).set_attrs(
    batch_size=8,
    shuffle=False,
    num_workers=1,
)

def save_image(img, path, nrow):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

def get_inst(t):
    edge = jt.init.zero([t.shape[0], 1, t.shape[2], t.shape[3]], int)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float32()

def change_label(real_A):
    real_A = real_A[:,0,:,:].unsqueeze(1)
    inst_A = get_inst(real_A)
    nc = 29
    input_label = jt.init.zero([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    real_A = real_A.int8()
    temp = jt.init.one([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    input_label = input_label.scatter_(1, real_A, temp)
    # input_label[:, [0, 14, 15, 18, 26]] *= 2
    input_label = jt.concat([input_label, inst_A], 1)
    return input_label

@jt.single_process_scope()
def eval():
    generator = SPADEGenerator(opt)
    # generator.load(f"{opt.model_path}/generator_139.pkl")
    generator.load(f"{opt.model_path}/generator_999.pkl")
    cnt = 1
    os.makedirs(f"{opt.output_path}", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        temp = change_label(real_A)
        fake_B = generator(temp)    
        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1
        
        
jt.sync_all(True)

eval()