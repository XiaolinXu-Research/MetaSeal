import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import os
import sys
sys.path.append('./')
sys.path.append('../')
from scripts.model.model import *
import scripts.config as c
from scripts.dataset import get_data_loaders
import modules.Unet_common as common
from tensorboardX import SummaryWriter
from transform import *
from skimage.metrics import structural_similarity as ssim
import numpy as np
from jpeg import JpegSS, JpegTest
from pyzbar.pyzbar import decode
from torchmetrics.image import StructuralSimilarityIndexMeasure
from calculate_PSNR_SSIM import ssim


# # Function to compute SSIM
def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min(),  win_size=11, channel_axis=0)



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
writer = None



def get_writer():
    global writer
    if writer is None:
        writer = SummaryWriter(comment='hinet', filename_suffix="steg")
    return writer



def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict, strict=False)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def qr_reconstruction_accuracy(original, reconstructed, threshold=128):
    # Binarize (if not already binary)
    original_bin = (original > threshold).astype(int)
    reconstructed_bin = (reconstructed > threshold).astype(int)
    
    # Calculate correct pixels
    correct_pixels = np.sum(original_bin == reconstructed_bin)
    total_pixels = original_bin.size
    
    # Compute accuracy
    accuracy = (correct_pixels / total_pixels) * 100
    return accuracy

def image_ssim(preds, targets, data_range=2.0):
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    return ssim(preds, targets)

def gauss_noise(shape):

    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise

def binary_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        # Generate binary noise with values 0 or 1
        noise[i] = torch.randint(0, 2, noise[i].shape).float().to(device)

    return noise

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.to(device)
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

test_noise_layer = JpegTest(85)

load(c.MODEL_PATH + c.suffix)

net.eval()



def backward_only(steg_img_path, net, device):
    """Run only the backward pass on a stego image"""
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    
    with torch.no_grad():
        # Load and preprocess stego image
        steg_img = Image.open(steg_img_path).convert('RGB')
        transform = torchvision.transforms.ToTensor()
        steg_img = transform(steg_img).unsqueeze(0).to(device)
        
        # Generate noise for backward pass
        backward_z = gauss_noise(dwt(steg_img).shape)
        
        # Backward pass
        output_rev = torch.cat((dwt(steg_img), backward_z), 1)
        backward_img = net(output_rev, rev=True)
        
        # Extract recovered secret
        secret_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        
        # Save recovered secret
        os.makedirs(c.IMAGE_PATH_secret_rev, exist_ok=True)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + 'recovered_secret.png')
        
        return secret_rev

class DWT_MODI(nn.Module):
    def __init__(self):
        super(DWT_MODI, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        # Create zero tensors with same shape as x_lh
        zeros = torch.zeros_like(x_LL)
        # Concatenate LL with zeros for other bands
        return torch.cat((x_LL, zeros, zeros, zeros), 1)



def inference(testloader, c, transformation, scale=1, angle=0, crop_ratio=0):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    
    
    with torch.no_grad():
        psnr_s = []
        psnr_c = []
        ssim_cover = []
        cnt = 0
        rec_acc = []
        total = 0
      
        for i, data in enumerate(testloader):
            

            cover, secret = data
            # print('cover: ', cover.shape)
            cover = cover.to(device)
            secret = secret.to(device)
            
            
                
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            total += len(cover)
            # print('cover_input: ', cover_input.shape)
            # print('secret_input: ', secret_input.shape)
            input_img = torch.cat((cover_input, secret_input),1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_steg[:, :c.channels_in, :, :] = cover_input[:, :c.channels_in, :, :]
            # output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg = iwt(output_steg)
            

            steg_img = steg.to(device)
            #################
            #  add perturbation #
            #################
            batch_size, channels, height, width = steg_img.shape
            # Generate Gaussian noise (mean=0, std=1) with the same shape as steg_img
            

            
            if transformation == 'noise':
                noise = torch.randn(batch_size, channels, height, width)
                # Adjust noise intensity by scaling (optional)
                noise_intensity = 0.05  # Set the noise intensity (you can adjust this)
                noise = noise * noise_intensity

                # Apply the noise to steg_img by adding it
                steg_img = steg_img + noise.to(device)

            if transformation == 'brightness':
                steg_img = apply_selected_transformations(
                    steg_img, device, 
                    brightness=True
                )
            
            if transformation == 'contrast':
                steg_img = apply_selected_transformations(
                    steg_img, device, 
                    contrast=True
                )
            
            if transformation == 'blur':
                steg_img = apply_selected_transformations(
                    steg_img, device, 
                    blur=True
                )
            
            if transformation == 'resize':
                steg_img = apply_selected_transformations(
                    steg_img, device, scale = scale,
                    resize=True
                )

            if transformation == 'rotate':
                steg_img = apply_selected_transformations(
                    steg_img, device, angle=angle,
                    rotate=True
                )

            if transformation == 'center_crop':
                steg_img = apply_selected_transformations(
                    steg_img, device, crop_ratio=crop_ratio,
                    center_crop=True
                )

            if transformation == 'flip':
                steg_img_tmp = apply_selected_transformations(
                    steg_img, device, 
                    flip=True
                )
            

                steg_img = F.hflip(steg_img_tmp)

            if transformation == 'jpeg':
                steg_img = test_noise_layer(steg_img.clone()).to(device)
                


            # backward_z = binary_noise(dwt(steg_img).shape)
            backward_z = gauss_noise(dwt(steg_img).shape)

         
            #################
            #   backward:   #
            #################
            # output_rev = torch.cat((output_steg, backward_z), 1)
            output_rev = torch.cat((dwt(steg_img).to(device), backward_z.to(device)), 1)
            bacward_img = net(output_rev, rev=True)
            secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
            
            secret_rev = iwt(secret_rev)
            cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
            
            cover_rev = iwt(cover_rev)
            # resi_cover = (steg_img.to(device) - cover.to(device)) * 10
            # resi_secret = (secret_rev.to(device) - secret.to(device)) * 10

            # secret =ss_layer(secret.to(device))
            # secret_rev =ss_layer(secret_rev.to(device))
            # secret_rev = torch.sigmoid(secret_rev.to(device))

          

            os.makedirs(c.IMAGE_PATH_cover, exist_ok=True)
            os.makedirs(c.IMAGE_PATH_secret, exist_ok=True)
            os.makedirs(c.IMAGE_PATH_steg, exist_ok=True)
            os.makedirs(c.IMAGE_PATH_secret_rev, exist_ok=True)
            os.makedirs(c.IMAGE_PATH_resi_cover, exist_ok=True)
            os.makedirs(c.IMAGE_PATH_resi_secret, exist_ok=True)

            steg = steg_img
            # steg_img = (steg_img>0.5).float()

            if i % 1 == 0:
        
                torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + 'cover_'+ '%d.png' % i)
                torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + 'secret_'+'%d.png' % i)
                if transformation == 'flip':
                    torchvision.utils.save_image(steg_img_tmp, c.IMAGE_PATH_steg + 'steg_' + '%d.png' % i)
                else:
                    torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + 'steg_' + '%d.png' % i)
                torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + 'secret_rev_' + '%d.png' % i)
                # torchvision.utils.save_image(resi_cover, c.IMAGE_PATH_resi_cover + 'resi_cover_'+'%d.png' % i)
                # torchvision.utils.save_image(resi_secret, c.IMAGE_PATH_resi_secret + 'resi_secret_'+ '%d.png' % i)
                torchvision.utils.save_image(cover_rev, c.IMAGE_PATH_cover + 'cover_rev_' + '%d.png' % i)
                
            # Convert tensor to PIL Image in memory
            # Save secret_rev temporarily
            torchvision.utils.save_image(secret_rev, 'temp_secret_rev.png')
            
            # Open saved image for decoding
            qr_image = Image.open('temp_secret_rev.png')
            decoded_objects = decode(qr_image)
            
            # # Clean up temporary file
            # os.remove('temp_secret_rev.png')
            
            if decoded_objects:
                cnt += 1
            
            ssim_tmp_cover = image_ssim(cover.cpu(), steg.cpu())
            ssim_cover.append(ssim_tmp_cover)

            secret_rev = secret_rev.cpu().numpy().squeeze() * 255
            np.clip(secret_rev, 0, 255)
            secret = secret.cpu().numpy().squeeze() * 255
            np.clip(secret, 0, 255)
            cover = cover.cpu().numpy().squeeze() * 255
            np.clip(cover, 0, 255)
            steg = steg.cpu().numpy().squeeze() * 255
            np.clip(steg, 0, 255)
            psnr_temp = computePSNR(secret_rev, secret)
            psnr_s.append(psnr_temp)
            psnr_temp_c = computePSNR(cover, steg)
            psnr_c.append(psnr_temp_c)

            # ssim_secret = compute_ssim(secret_rev, secret)
            # print('cover: ', cover.shape)
            # ssim_tmp_cover = compute_ssim(cover, steg)
            # ssim_cover.append(ssim_tmp_cover)

            # print(f"SSIM (secret): {ssim_secret}")
            # print(f"SSIM (cover): {ssim_cover}")
            accuracy = qr_reconstruction_accuracy(secret_rev, secret)
            rec_acc.append(accuracy)

            if i == 10:
                break

            
       
        # w = get_writer()
        # if w is not None:
        #     w.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s), "variance": np.std(psnr_s)})
        #     w.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c), "variance": np.std(psnr_c)})
        #     w.add_scalars("SSIM_Cover", {"average ssim": np.mean(ssim_cover), "variance": np.std(ssim_cover)})
        #     w.add_scalars("Rec_Acc", {"average rec acc": np.mean(rec_acc), "variance": np.std(rec_acc)})

        print('cnt: ', cnt)
        print('total: ', total)
        print('success rate: ', cnt / total)
        # print('rec acc: ', np.mean(rec_acc))
        # print('variance', np.std(psnr_c), np.std(ssim_cover))

        return np.mean(psnr_s), np.mean(psnr_c), np.mean(ssim_cover)

if __name__ == '__main__':
    try:
        testloader = get_data_loaders(c, 1, c.cropsize_val, "val")
        transformations = ['none']
        # transformations = ['none','noise', 'brightness', 'contrast', 'blur', 'flip', 'jpeg']
        for transformation in transformations:
            if transformation == 'resize':
                scale_list = [2]
                for scale in scale_list:
                    print('Scale: ', scale)
                    psnr_s, psnr_c, ssim = inference(testloader, c, transformation, scale=scale)
                    print('PSNR_S: ', psnr_s)
                    print('PSNR_C: ', psnr_c)
                    print('SSIM_Cover: ', ssim)
            elif transformation == 'rotate':
                rotate_list = [15,30,45,60,75,90]
                for angle in rotate_list:
                    print('angle: ', angle)
                    psnr_s, psnr_c, ssim = inference(testloader, c, transformation, angle=angle)
                    print('PSNR_S: ', psnr_s)
                    print('PSNR_C: ', psnr_c)
                    print('SSIM_Cover: ', ssim)
            elif transformation == 'center_crop':
                crop_ratio_list = [0.05,0.1,0.15,0.2, 0.25]
                for crop_ratio in crop_ratio_list:
                    print('crop_ratio: ', crop_ratio)
                    psnr_s, psnr_c, ssim = inference(testloader, c, transformation, crop_ratio=crop_ratio)
                    print('PSNR_S: ', psnr_s)
                    print('PSNR_C: ', psnr_c)
                    print('SSIM_Cover: ', ssim)
            else:
                psnr_s, psnr_c, ssim = inference(testloader, c, transformation)
                print('Transformation: ', transformation)
                print('PSNR_S: ', psnr_s)
                print('PSNR_C: ', psnr_c)
                print('SSIM_Cover: ', ssim)
    finally:
        if writer is not None:
            writer.close()
