# import argparse

# parser = argparse.ArgumentParser(description="Inference Post processing step for VAE")
# parser.add_argument('-i', '--input', type=str, required=True, help='input, full file name with file path')
# parser.add_argument('-o', '--output', type=str, required=True, help='output, full file name with file path')

# args = parser.parse_args()
#mat_input = "phase.mat"
#mat_output = "phase1.mat"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorflow.keras.utils import to_categorical
import scipy.io
from scipy.io import loadmat, savemat
from sklearn.neighbors import KernelDensity
import joblib
#NEWS LIB
from utils.dataload import apply_normalization
from utils.data_augmentation import common_test_augmentation
import albumentations
from utils.training import reshape_masks


def direct(valor,x):
    pdf_name = 'anatomical.sav'
    pdf = joblib.load(pdf_name)
    tam=0;
     
    n_latent = 32
    class Anatomical_VAE(nn.Module):
        def __init__(self,n_latent,pdf):
            super(Anatomical_VAE, self).__init__()
            self.pdf = pdf
            self.n_latent = n_latent
            self.drop = nn.Dropout(0.2)
            self.conv1 = nn.Conv3d(4, 32, 3, padding=1, padding_mode='zeros')
            self.conv2 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
            self.pool1 = nn.MaxPool3d((2,2,2))
            self.conv3 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')
            self.conv4 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
            self.pool2 = nn.MaxPool3d((2,2,2))
            self.conv5 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')
            self.conv6 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
            self.pool3 = nn.MaxPool3d((2,2,2))
            self.conv7 = nn.Conv3d(128, 256, 3, padding=1, padding_mode='zeros')
            self.conv8 = nn.Conv3d(256, 256, 3, padding=1, padding_mode='zeros')
            self.pool4 = nn.MaxPool3d((2,2,2))
            self.conv9 = nn.Conv3d(256, 512, 3, padding=1, padding_mode='zeros')
            self.conv10 = nn.Conv3d(512, 512, 3, padding=1, padding_mode='zeros')
            self.z_mean = nn.Linear(512 * 24**2, n_latent)
            self.z_var = nn.Linear(512 * 24**2, n_latent)
            self.z_develop = nn.Linear(n_latent, 512 * 24**2)
            self.trans1 = nn.ConvTranspose3d(512, 256, 2, stride=2)
            self.conv11 = nn.Conv3d(512, 256, 3, padding=1, padding_mode='zeros')
            self.conv12 = nn.Conv3d(256, 256, 3, padding=1, padding_mode='zeros')
            self.trans2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
            self.conv13 = nn.Conv3d(256, 128, 3, padding=1, padding_mode='zeros')
            self.conv14 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
            self.trans3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
            self.conv15 = nn.Conv3d(128, 64, 3, padding=1, padding_mode='zeros')
            self.conv16 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
            self.trans4 = nn.ConvTranspose3d(64, 32, 2, stride=2)
            self.conv17 = nn.Conv3d(64, 32, 3, padding=1, padding_mode='zeros')
            self.conv18 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
            self.conv19 = nn.Conv3d(32, 4, 1)
        def sample_z(self, mean, logvar):
            stddev = torch.exp(0.5 * logvar).to(device)
            noise = torch.randn(stddev.size()).to(device)
            result = ((noise * stddev) + mean).to(device)
            return result
    
        def latent_vector_transformation(self, z):
            zn1 = self.pdf.sample(1)
            zn1 = zn1.reshape(z.shape)
            zn1 = torch.from_numpy(zn1.astype(np.float32)).to(device)
            z_prime = torch.sub(zn1,z)
            return z_prime
    
        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = self.drop(x)
            down1 = F.relu(self.conv2(x))
            x = self.pool1(down1)
            x = F.relu(self.conv3(x))
            x = self.drop(x)
            down2 = F.relu(self.conv4(x))
            x = self.pool2(down2)
            x = F.relu(self.conv5(x))
            x = self.drop(x)
            down3 = F.relu(self.conv6(x))
            x = self.pool3(down3)
            x = F.relu(self.conv7(x))
            x = self.drop(x)
            down4 = F.relu(self.conv8(x))
            x = self.pool4(down4)
            x = F.relu(self.conv9(x))
            x = self.drop(x)
            x = F.relu(self.conv10(x))
            x = x.view(x.size(0), -1)
    
            mean = self.z_mean(x)
            logvar = self.z_var(x)
    
            z = self.sample_z(mean, logvar)
            z = self.latent_vector_transformation(z)
            x = self.z_develop(z)
            x = x.view(x.size(0), 512,1, 24, 24)
            x = F.relu(self.trans1(x))
            x = torch.cat((x,down4), 1)
            x = F.relu(self.conv11(x))
            x = self.drop(x)
            x = F.relu(self.conv12(x))
            x = F.relu(self.trans2(x))
            x = torch.cat((x,down3), 1)
            x = F.relu(self.conv13(x))
            x = self.drop(x)
            x = F.relu(self.conv14(x))
            x = F.relu(self.trans3(x))
            x = torch.cat((x,down2), 1)
            x = F.relu(self.conv15(x))
            x = self.drop(x)
            x = F.relu(self.conv16(x))
            x = F.relu(self.trans4(x))
            x = torch.cat((x,down1), 1)
            x = F.relu(self.conv17(x))
            x = self.drop(x)
            x = F.relu(self.conv18(x))
            x = F.relu(self.conv19(x))
    
            return z, x, mean, logvar
    
    scan_shape = (16,384,384)
    anatomical_vae = Anatomical_VAE(n_latent,pdf).to(device)
    anatomical_vae.load_state_dict(torch.load('anatomical_best.pt'))
    print('model loaded!')
    
    
    if x==1:
        mat_array = loadmat(valor)
        mat_array = mat_array.get('phase')
        result=np.empty_like(mat_array);
        a,b,c=mat_array.shape
        auxM=np.empty((384,384,c));
        if a>=385 or b>=385:
            tam=1;
            for i in range(c):
                image=mat_array[:,:,i];
                common_reshape = common_test_augmentation(384)
                image1 = albumentations.Compose(common_reshape)(image=image)["image"]
                auxM[:,:,i]=image1; 
        else:
            auxM=mat_array;
        
        mat_array = mat_array.T
        w,l,h = mat_array.shape
        
        if w>=17:
            mat_array=mat_array[:16,:,:];
        
        padded_images = np.zeros(scan_shape, dtype=np.float32)
        padded_images[:w,:l,:h] = mat_array
        mat_array = to_categorical(padded_images,4)
        mat_array = np.expand_dims(mat_array, axis=0)
        mat_array = np.swapaxes(mat_array,0,4)
        mat_array = np.squeeze(mat_array,axis=4)
        mat_array = np.expand_dims(mat_array, axis=0)
        mat_tensor = torch.from_numpy(mat_array).to(device)
        print('processing done!')
        with torch.no_grad():
            _,output,_,_ = anatomical_vae(mat_tensor)
        output = torch.squeeze(torch.argmax(output,1))
        output = output.cpu().numpy().T
        output_array = output[:h,:l,:w]
        result1=output_array
        
        if tam==1:
            for j in range (c):
                image2 = reshape_masks(result1[:,:,j], (a,b))
                result[:,:,j]=image2;
            tam=0;
        else:
            result=result1;
                
            
        scipy.io.savemat('auto.mat',{'auto': result})
        #savemat(mat_output, {'phase':output_array})
        print('results saved at {}'.format('auto.mat'))
        
    if x==2:
        mat_array = loadmat(valor)
        mat_array = mat_array.get('resultado')
        aux=mat_array;
        d=np.size(aux,3)
        a,b,c,d=mat_array.shape
        auxM=np.empty((384,384,c));
        result=np.empty_like(aux);
        output_array1=np.empty((a,b,c));
        for i in range (d):
            mat_array=aux[:,:,:,i];
            if a>=385 or b>=385:
                tam=1;
                for k in range(c):
                    image=mat_array[:,:,k];
                    common_reshape = common_test_augmentation(384)
                    image1 = albumentations.Compose(common_reshape)(image=image)["image"]
                    auxM[:,:,k]=image1;
            else:
                auxM=mat_array;
                        
               
            mat_array = auxM.T
            mat_array[np.isnan(mat_array)]=0; 
            
            w,l,h = mat_array.shape
            
            if w>=17:
                mat_array=mat_array[:16,:,:];
            
            padded_images = np.zeros(scan_shape, dtype=np.float32)
            padded_images[:w,:l,:h] = mat_array            
            mat_array = to_categorical(padded_images,num_classes=4)
            mat_array = np.expand_dims(mat_array, axis=0)
            mat_array = np.swapaxes(mat_array,0,4)
            mat_array = np.squeeze(mat_array,axis=4)
            mat_array = np.expand_dims(mat_array, axis=0)
            mat_tensor = torch.from_numpy(mat_array).to(device)
            print('processing done!')
            with torch.no_grad():
                _,output,_,_ = anatomical_vae(mat_tensor)
            output = torch.squeeze(torch.argmax(output,1))
            output = output.cpu().numpy().T
            output_array = output[:h,:l,:w]
            
            if tam==1:
                for j in range (c):
                    image2 = reshape_masks(output_array[:,:,j], (a,b))
                    output_array1[:,:,j]=image2;
                tam=0;
                print(output_array1.shape)
                print(result.shape)
                
            result[:,:,:,i]=output_array1
            
            
        scipy.io.savemat('auto.mat',{'auto': result})
        #savemat(mat_output, {'phase':output_array})
        print('results saved at {}'.format('auto.mat'))
