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

from network.encoder import ConvEncoder
import json
from network.generator import SPADEGenerator
from network.models import *
from datasets import *
from network.loss import *
from pretrained.deeplab_jittor.deeplab import DeepLab
# from tensorboardX import SummaryWriter
from Diffaug import * 

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=139, help="number of epochs of training")
parser.add_argument("--input_path", type=str, default="./datasets/")
parser.add_argument("--output_path", type=str, default="./results_train")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--eval_batch_size", type=int, default=6, help="size of the evaluation batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--label_c", type=int, default=30, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--add_test", action='store_true')
parser.add_argument("--lambda_pixel", type=float, default=5)
parser.add_argument("--lambda_kld", type=float, default=0.05)
parser.add_argument("--lambda_vgg", type=float, default=10.0)
parser.add_argument("--lambda_feat", type=float, default=10.0)
parser.add_argument("--lambda_seg", type=float, default=40.0)
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
opt.output_path += 'add_test' if opt.add_test else ''
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


os.makedirs(f"{opt.output_path}/images/", exist_ok=True)
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

# Loss functions
Gan_loss_g = Ganloss('g')
Gan_loss_d = Ganloss('d')
criterion_pixelwise = nn.L1Loss()
KLD_loss = KLDLoss()
VGG_loss = VGGLoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = opt.lambda_pixel
lambda_kld = opt.lambda_kld
lambda_vgg = opt.lambda_vgg
lambda_feat = opt.lambda_feat
lambda_seg = opt.lambda_seg

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
encoder = ConvEncoder(opt)
generator = SPADEGenerator(opt)
discriminator = Discriminator(opt)
# encoder = ConvEncoder(opt)
deeplab = DeepLab(output_stride=16, num_classes=29)
deeplab.eval()

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/saved_models/discriminator_{opt.epoch}.pkl")
    encoder.load(f"{opt.output_path}/saved_models/encoder_{opt.epoch}.pkl")
    # encoder.load(f"{opt.output_path}/saved_models/encoder_{opt.epoch}.pkl")

deeplab.load("./pretrained/Epoch_40.pkl")


# Optimizers
optimizer_G = jt.optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=opt.lr / 2, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr * 2, betas=(opt.b1, opt.b2))

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

dataloader = ImageDataset(opt.input_path, transform_label=transform_label, transform_img=transform_img, mode="train", add_test=opt.add_test).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = ImageDataset(opt.input_path, transform_label=transform_label, transform_img=transform_img, mode="val").set_attrs(
    batch_size=opt.eval_batch_size,
    shuffle=False,
    num_workers=1,
)


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
    real_A = jt.round(real_A).int8()
    temp = jt.init.one([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    input_label = input_label.scatter_(1, real_A, temp)
    input_label = jt.concat([input_label, inst_A], 1)
    return input_label, real_A[:,0,:,:]

@jt.single_process_scope()
def eval(epoch):
    generator.eval()
    encoder.eval()
    cnt = 1
    os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch + 1}", exist_ok=True)
    fake_B_1 = ""
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        with open(f"{opt.input_path}label_to_img.json")as f:
            label_to_img = json.load(f)
        imgs_name = [label_to_img[single_photo_id + '.png'] for single_photo_id in photo_id]
        input_image = jt.randn(len(photo_id), 3, opt.img_height, opt.img_width)
        for img_count, img_name in enumerate(imgs_name):
            img_B = Image.open(f"{opt.input_path}train/imgs/" + img_name)
            temp_transform_img = transform.Compose(transform_img)
            img_B = temp_transform_img(img_B)
            input_image[img_count] = img_B

        latent_z, mu, logvar = encoder.encode_z(input_image)
        temp, _ = change_label(real_A)
        fake_B = generator(temp, z=latent_z)
        if i == 0:
            fake_B_1 = fake_B
        if i == 1:
            # visual image result
            img_sample = np.concatenate([fake_B_1.data, fake_B.data], -2)
            img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch + 1}_sample.png", nrow=3)
        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch + 1}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1
        generator.train()
        encoder.train()

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

def divide_pred(pred):
    fake = []
    real = []
    for p in pred:
        fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
        real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    return fake, real

def cal_discriminator(real_A, fake_B, real_B, mode):
    if mode == 'discriminator':
        fake_concat = jt.concat([real_A, fake_B], dim=1).detach()
    elif mode == 'generator':
        fake_concat = jt.concat([real_A, fake_B], dim=1)
    real_concat = jt.concat([real_A, real_B], dim=1)
    fake_and_real = jt.concat([fake_concat, real_concat], dim=0)
    discriminator_out = discriminator(fake_and_real)
    return divide_pred(discriminator_out)
# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    total_seg_loss = 0
    lambda_seg = 40.0 if epoch >= 130 else 0
    for i, (real_B, real_A, _) in enumerate(dataloader):    # real_B为真实图片， real_A为真实标签
        # Adversarial ground truths
        valid = jt.ones([real_A.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_A.shape[0], 1]).stop_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(discriminator)
        policy = "color"   
        real_A, target_label = change_label(real_A)
        latent_z, mu, logvar = encoder.encode_z(real_B)
        fake_B = generator(real_A, z=latent_z)      # fake_B为真实标签生成的假图片
        pred_fake, pred_real = cal_discriminator(real_A, DiffAugment(fake_B, policy=policy), DiffAugment(real_B, policy=policy), 'discriminator')

        loss_D_fake = Gan_loss_d(pred_fake, False)
        loss_D_real = Gan_loss_d(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        optimizer_D.step(loss_D)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(discriminator)        
        fake_and_real = jt.concat([real_B, fake_B], dim=0)      
        fake_and_real = DiffAugment(fake_and_real, policy=policy, sync=True)
        real_B1, fake_B1 = jt.chunk(fake_and_real, 2, dim=0)
        pred_fake, pred_real = cal_discriminator(real_A, fake_B1, real_B1, 'generator')
        
        loss_G_GAN = Gan_loss_g(pred_fake, True)
        
        num_D = len(pred_fake)
        GAN_Feat_loss = 0
        
        for ii in range(num_D):
            # for each discriminator, last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[ii]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = criterion_pixelwise(pred_fake[ii][j], pred_real[ii][j].detach())
                GAN_Feat_loss += unweighted_loss * lambda_feat / num_D
        if lambda_pixel > 0:
            loss_G_L1 = criterion_pixelwise(fake_B, real_B)
        perc_loss = VGG_loss(fake_B, real_B)
        if lambda_seg > 0:
            seg_label = deeplab(fake_B) # batchsize, 29, 384, 512
            seg_loss = nn.cross_entropy_loss(seg_label, target_label, ignore_index=255)
        
        # total_seg_loss += seg_loss
        # real_seg_loss = nn.cross_entropy_loss(real_seg_label, target_label, ignore_index=255)
        # print(seg_loss, real_seg_loss)
        # loss_G = (loss_G_GAN + GAN_Feat_loss + lambda_vgg * perc_loss + lambda_kld * kld_loss) / 4
        kld_loss = KLD_loss(mu, logvar)
        loss_G = loss_G_GAN + GAN_Feat_loss + lambda_vgg * perc_loss + lambda_kld * kld_loss

        seg_loss = lambda_seg * seg_loss if lambda_seg > 0 else jt.float64(0)
        if seg_loss > 0:
            loss_G += seg_loss
        loss_G_L1 = lambda_pixel * loss_G_L1 if lambda_pixel > 0 else jt.float64(0)
        if loss_G_L1 > 0:
            loss_G += loss_G_L1
        loss_G = loss_G / 4
        # loss_G = (loss_G_GAN + GAN_Feat_loss + lambda_pixel * loss_G_L1) / 3 
        optimizer_G.step(loss_G)
        # writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, perc: %f, KLD: %f, FM: %f, adv: %f, seg: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_L1.numpy()[0],
                        perc_loss.numpy()[0],
                        kld_loss.numpy()[0],   
                        GAN_Feat_loss.numpy()[0],
                        loss_G_GAN.numpy()[0],
                        seg_loss.numpy()[0] * lambda_seg,
                        time_left,
                    )   
                )
    # total_seg_loss = total_seg_loss / len(dataloader)
    # print(total_seg_loss)

    if jt.rank == 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        eval(epoch)
    #     # Save model checkpoints
        generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch + 1}.pkl"))
        discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch + 1}.pkl"))
        encoder.save(os.path.join(f"{opt.output_path}/saved_models/encoder_{epoch + 1}.pkl"))