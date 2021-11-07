# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:47:35 2021

@author: alejo
"""
import cv2
import scipy.io
from torch.utils.data import DataLoader
import albumentations
import numpy as np
import os
script_dir = os.path.dirname(__file__)
# ---- My utils ----
from models import *
from utils.arguments import *

from utils.data_augmentation import common_test_augmentation
from utils.dataload import apply_normalization
from utils.dataload import load_nii

from utils.training import convert_multiclass_mask
from utils.training import reshape_masks

abs_file_path = os.path.join(script_dir, args.model_checkpoint)

def url(valor,num,phase):
    print(valor)
    print(num)
    print(phase)
    np.set_printoptions(precision=4)
      
    model = model_selector(args.model_name)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if args.model_checkpoint != "":
        print("Load from pretrained checkpoint: {}".format(abs_file_path))
        model.load_state_dict(torch.load(abs_file_path,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    else:
        model_checkpoint = args.output_dir + "/model_" + args.model_name + "_best_iou.pt"
        if not os.path.exists(model_checkpoint):
            assert False, "Evaluating without model checkpoint?!"
        else:
            print("No checkpoint provided! Loading from best checkpoint: {}".format(model_checkpoint))
            model.load_state_dict(torch.load(model_checkpoint))
           
    if valor=='':
        imag_org=np.empty(0);
    else:
        img_path = valor;
        mat_contents = scipy.io.loadmat(img_path)
        imag_org=mat_contents[('original')];
        #imag_org=load_nii(img_path)[0][..., :, :];
            #Arreglar (abajoquitar)
            #imag_org=np.squeeze(imag_org, axis=3);
        a1,b1,c1,d1=imag_org.shape;
        result=np.empty_like(imag_org);

    #TODAS LAS FASES y SLICES
    if num==1:
        for i in range (d1):
            for j in range (c1):
                
                #image = load_nii(img_path)[0][...,j,i]
                #Arreglar(Arriba add abajo eliminar)
                image=imag_org[:,:,j,i];
                #
                
                common_reshape = common_test_augmentation(224)
                image1 = albumentations.Compose(common_reshape)(image=image)["image"]
                image1 = apply_normalization(image1,'standardize')
                
                image1=np.expand_dims(image1,axis=0)
                image1=torch.from_numpy(image1)
                
                image1=np.expand_dims(image1,axis=0)
                dat = DataLoader(image1, batch_size=1, shuffle=False, drop_last=False)
                
                model.eval();    
                
                with torch.no_grad():
                    for ind, img in enumerate (dat):
                        img = img.type(torch.float).cpu()
                        prob_pred = model(img)

                for indx, single_pred in enumerate(prob_pred):
                    pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                    pred_mask = reshape_masks(pred_mask.squeeze(0), image.shape)
                result[:,:,j,i]=pred_mask;
                print(i)
                print(j)
                print('-')
        scipy.io.savemat('resultado.mat',{'resultado': result})         
                
    #SOLO SLICE MEDIO
    if num==2:
        result=result.transpose(0,1,3,2);
        j=round((c1/2),ndigits=None);
        result=result[:,:,:,j];
        for i in range (d1):
            image=imag_org;
            #image = load_nii(img_path)[0][...,j,i]
            #Arreglar(Arriba add abajo eliminar)
            image=imag_org[:,:,j,i];
            #
            common_reshape = common_test_augmentation(224)
            image1 = albumentations.Compose(common_reshape)(image=image)["image"]
            image1 = apply_normalization(image1,'standardize')
        
            image1=np.expand_dims(image1,axis=0)
            image1=torch.from_numpy(image1)
            
            image1=np.expand_dims(image1,axis=0)
            dat = DataLoader(image1, batch_size=1, shuffle=False, drop_last=False)
            
            model.eval();    
            
            with torch.no_grad():
                for ind, img in enumerate (dat):
                    img = img.type(torch.float).cpu()
                    prob_pred = model(img)

            for indx, single_pred in enumerate(prob_pred):
                pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                pred_mask = reshape_masks(pred_mask.squeeze(0), image.shape)
            result[:,:,i]=pred_mask;
            print(i)
            print(j)
            print('-')
        scipy.io.savemat('middle.mat',{'middle': result})     
    
    #SOLO PHASE NECESARIA AUMENTAR 3dimension el slice
    if num==3:
        i=int(phase-1);
        result=result[:,:,:,i];
        for j in range (c1):
            image=imag_org;    
            #image = load_nii(img_path)[0][...,j,i]
            #Arreglar(Arriba add abajo eliminar)
            image=imag_org[:,:,j,i];
            #
            common_reshape = common_test_augmentation(224)
            image1 = albumentations.Compose(common_reshape)(image=image)["image"]
            image1 = apply_normalization(image1,'standardize')
                
            image1=np.expand_dims(image1,axis=0)
            image1=torch.from_numpy(image1)
                
            image1=np.expand_dims(image1,axis=0)
            dat = DataLoader(image1, batch_size=1, shuffle=False, drop_last=False)
                
            model.eval();    
                
            with torch.no_grad():
                for ind, img in enumerate (dat):
                    img = img.type(torch.float).cpu()
                    prob_pred = model(img)

            for indx, single_pred in enumerate(prob_pred):
                pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                pred_mask = reshape_masks(pred_mask.squeeze(0), image.shape)
                result[:,:,j]=pred_mask;
                print(i)
                print(j)
                print('-')
        scipy.io.savemat('phase.mat',{'phase': result}) 
    
    
    scipy.io.savemat('original.mat',{'original': imag_org})
       

