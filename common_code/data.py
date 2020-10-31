import numpy as np
import csv
import os
import os.path
from PIL import Image
import glob
import torch.utils.data as udata
import json
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage import io, color
import transforms_struct as tfs_s
import torchvision.transforms as torch_transforms
import torch
from train_aux import _worker_init_fn_

def preprocess_data_2017_structures(root_dir, out_dir, seg_dir='Train_Lesion', step_image=1):
    print('pre-processing data ...\n')
    # training data
    # Lesion images
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    # Lesion/Skin mask
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    # Superpixels
    melanoma_sp  = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.png')); melanoma_sp.sort()
    nevus_sp     = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.png')); nevus_sp.sort()
    sk_sp        = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.png')); sk_sp.sort()
    # Structure information
    melanoma_st  = glob.glob(os.path.join(root_dir, 'Train_Dermo', 'melanoma', '*.json')); melanoma_st.sort()
    nevus_st     = glob.glob(os.path.join(root_dir, 'Train_Dermo', 'nevus', '*.json')); nevus_st.sort()
    sk_st        = glob.glob(os.path.join(root_dir, 'Train_Dermo', 'seborrheic_keratosis', '*.json')); sk_st.sort()
    
    train_csv = os.path.join(out_dir,'train.csv')
    with open(train_csv, 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in np.arange(0,len(melanoma),step_image):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            filename_sp = melanoma_sp[k]
            filename_st = melanoma_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['1'])
        for k in np.arange(0,len(nevus),step_image):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            filename_sp = nevus_sp[k]
            filename_st = nevus_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])
        for k in np.arange(0,len(sk),step_image):
            filename = sk[k]
            filename_seg = sk_seg[k]
            filename_sp = sk_sp[k]
            filename_st = sk_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])

    # training data oversample    
    train_oversample_csv = os.path.join(out_dir,'train_oversample.csv')
    with open(train_oversample_csv, 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for k in np.arange(0,len(melanoma),step_image):
                filename = melanoma[k]
                filename_seg = melanoma_seg[k]
                filename_sp = melanoma_sp[k]
                filename_st = melanoma_st[k]
                writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['1'])
        for k in np.arange(0,len(nevus),step_image):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            filename_sp = nevus_sp[k]
            filename_st = nevus_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])
        for k in np.arange(0,len(sk),step_image):
            filename = sk[k]
            filename_seg = sk_seg[k]
            filename_sp = sk_sp[k]
            filename_st = sk_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])

    # val data
    if step_image != 1:
        step_image = 4
    # Lesion images
    melanoma = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    #### segmentation of val data is not used! ######
    # Lesion/Skin mask
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Val_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Val_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Val_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    # Superpixels
    melanoma_sp  = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.png')); melanoma_sp.sort()
    nevus_sp     = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.png')); nevus_sp.sort()
    sk_sp        = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.png')); sk_sp.sort()
    # Structure information
    melanoma_st  = glob.glob(os.path.join(root_dir, 'Val_Dermo', 'melanoma', '*.json')); melanoma_st.sort()
    nevus_st     = glob.glob(os.path.join(root_dir, 'Val_Dermo', 'nevus', '*.json')); nevus_st.sort()
    sk_st        = glob.glob(os.path.join(root_dir, 'Val_Dermo', 'seborrheic_keratosis', '*.json')); sk_st.sort()
    
    val_csv = os.path.join(out_dir,'val.csv')
    with open(val_csv, 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in np.arange(0,len(melanoma),step_image):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            filename_sp = melanoma_sp[k]
            filename_st = melanoma_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['1'])
        for k in np.arange(0,len(nevus),step_image):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            filename_sp = nevus_sp[k]
            filename_st = nevus_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])
        for k in np.arange(0,len(sk),step_image):
            filename = sk[k]
            filename_seg = sk_seg[k]
            filename_sp = sk_sp[k]
            filename_st = sk_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])
    # test data
    # Lesion images
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    # Lesion/Skin mask
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    # Superpixels
    melanoma_sp  = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.png')); melanoma_sp.sort()
    nevus_sp     = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.png')); nevus_sp.sort()
    sk_sp        = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.png')); sk_sp.sort()
    # Structure information
    melanoma_st  = glob.glob(os.path.join(root_dir, 'Test_Dermo', 'melanoma', '*.json')); melanoma_st.sort()
    nevus_st     = glob.glob(os.path.join(root_dir, 'Test_Dermo', 'nevus', '*.json')); nevus_st.sort()
    sk_st        = glob.glob(os.path.join(root_dir, 'Test_Dermo', 'seborrheic_keratosis', '*.json')); sk_st.sort()
    test_csv = os.path.join(out_dir,'test.csv')
    with open(test_csv, 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in np.arange(0,len(melanoma),step_image):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            filename_sp = melanoma_sp[k]
            filename_st = melanoma_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['1'])
        for k in np.arange(0,len(nevus),step_image):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            filename_sp = nevus_sp[k]
            filename_st = nevus_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])
        for k in np.arange(0,len(sk),step_image):
            filename = sk[k]
            filename_seg = sk_seg[k]
            filename_sp = sk_sp[k]
            filename_st = sk_st[k]
            writer.writerow([filename] + [filename_seg] + [filename_sp] + [filename_st] + ['0'])

def data_statistics(root_dir):
    step_image = 1
    print('pre-processing data ...\n')
    # training data
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()

    # Validation data
    melanoma_v = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma_v.sort()
    nevus_v    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus_v.sort()
    sk_v       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk_v.sort()
    
    R, G, B = 0, 0, 0
    L, a, b = 0, 0, 0
    N = 0
    for k in np.arange(0,len(melanoma),step_image):
        image = np.array(Image.open(melanoma[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))

        N += 1
    for k in np.arange(0,len(nevus),step_image):
        image = np.array(Image.open(nevus[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))
        N += 1
    for k in np.arange(0,len(sk),step_image):
        image = np.array(Image.open(sk[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))
        N += 1

    for k in np.arange(0,len(melanoma_v),step_image):
        image = np.array(Image.open(melanoma_v[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))
        N += 1
    for k in np.arange(0,len(nevus_v),step_image):
        image = np.array(Image.open(nevus_v[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))
        N += 1
    for k in np.arange(0,len(sk_v),step_image):
        image = np.array(Image.open(sk_v[k]))
        imagelab = color.rgb2lab(image)
        R += np.sum(np.sum(image[0]))
        G += np.sum(np.sum(image[1]))
        B += np.sum(np.sum(image[2]))
        L += np.sum(np.sum(imagelab[0]))
        a += np.sum(np.sum(imagelab[1]))
        b += np.sum(np.sum(imagelab[2]))
        N += 1
    
    R, G, B = R/N, G/N, B/N
    L, a, b = L/N, a/N, b/N

    # std dev
    for k in np.arange(0,len(melanoma),step_image):
        image = np.array(Image.open(melanoma[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))

    for k in np.arange(0,len(nevus),step_image):
        image = np.array(Image.open(nevus[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))
    for k in np.arange(0,len(sk),step_image):
        image = np.array(Image.open(sk[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))

    for k in np.arange(0,len(melanoma_v),step_image):
        image = np.array(Image.open(melanoma_v[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))
    for k in np.arange(0,len(nevus_v),step_image):
        image = np.array(Image.open(nevus_v[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))
    for k in np.arange(0,len(sk_v),step_image):
        image = np.array(Image.open(sk_v[k]))
        imagelab = color.rgb2lab(image)
        sR += np.sum(np.sum((image[0]-R)**2))
        sG += np.sum(np.sum((image[1]-G)**2))
        sB += np.sum(np.sum((image[2]-B)**2))
        sL += np.sum(np.sum((imagelab[0]-L)**2))
        sa += np.sum(np.sum((imagelab[1]-a)**2))
        sb += np.sum(np.sum((imagelab[2]-b)**2))

    sR, sG, sB = np.sqrt(sR/N), np.sqrt(sG/N), np.sqrt(sB/N)
    sL, sa, sa = np.sqrt(sL/N), np.sqrt(sa/N), np.sqrt(sb/N)

    return [R,G,B],[sR,sG,sB],[L,a,b],[sL,sa,sb]

def structure_processing(superpixels_path, structures_path):
    feat = json.load(open(structures_path))
    sp = 255*img.imread(superpixels_path)
    list_structures = []

    # Para convertir superpíxeles definidos en 3D a 2D (utilizados R, G, B para 
    # contar las vueltas al rango [0,255] y B todo a 0)
    sp = (sp[:,:,0] + (2**8 - 1)*sp[:,:,1] + (2**16 - 1)*sp[:,:,2]).astype(int) # número_etiqueta + etiqueta_máxima*vuelta

    if 'pigment_network' in feat:
        struct = np.array(feat['pigment_network'])[sp] 
        sum_struct = struct
        list_structures.append(Image.fromarray(255*struct.astype('uint8'), 'L'))
        
    if 'negative_network' in feat:
        struct = np.array(feat['negative_network'])[sp] 
        sum_struct = np.logical_or(sum_struct,struct)
        list_structures.append(Image.fromarray(255*struct.astype('uint8'), 'L'))

    if 'milia_like_cyst' in feat:
        struct = np.array(feat['milia_like_cyst'])[sp] 
        sum_struct = np.logical_or(sum_struct,struct)
        list_structures.append(Image.fromarray(255*struct.astype('uint8'), 'L'))

    if 'streaks' in feat:
        struct = np.array(feat['streaks'])[sp] 
        sum_struct = np.logical_or(sum_struct,struct)
        list_structures.append(Image.fromarray(255*struct.astype('uint8'), 'L'))

    return list_structures, sum_struct

class ISIC_structures(udata.Dataset):
    def __init__(self, csv_file, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        ima = Image.open(pair[0])
        img_seg = Image.open(pair[1])
        label = int(pair[4])

        # Structure processing (from one .json to 4 images)
        struct_list, sum_struct = structure_processing(superpixels_path=pair[2], structures_path=pair[3])
        skin = Image.fromarray((255*np.logical_not(np.logical_or(np.array(img_seg).copy(), sum_struct))).astype('uint8'), 'L')
        img_seg = Image.fromarray(255*np.logical_not(np.logical_or(np.array(skin).copy(), sum_struct)).astype('uint8'), 'L')

        # construct one sample
        sample = {'image': ima, 'skin': skin, 'image_seg': img_seg, 'str_1': struct_list[0], 'str_2': struct_list[1], 'str_3': struct_list[2], 'str_4': struct_list[3], 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

# Selecting database
def choose_db(global_path, db):
    # Reduced database
    if db == 0:
        path = global_path + '/reduced'
    elif db == 1:
        path = global_path

    return path

def select_data (selected_images, db_dir, path_db):
    train_csv = os.path.join(db_dir,'train.csv')
    all_csv = os.path.join(path_db,'all_images.csv')
    num_images = 0
    with open(train_csv, 'wt', newline='') as write_file:
        with open(all_csv, newline='') as read_file:
            writer = csv.writer(write_file, delimiter=',')
            reader = csv.reader(read_file)
            for row in reader:
                if int(row[0]) in selected_images:
                    writer.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                    num_images = num_images + 1
                    if num_images == len(selected_images):
                        break
    print('Created .csv\n')

def select_data_min2structs (opt, db_dir, path_db):
    train_csv = os.path.join(db_dir,'train.csv')
    val_csv = os.path.join(db_dir,'val.csv')
    all_csv = os.path.join(path_db,'all_images.csv')
    total_tr = 20
    total_val = 6
    str1, str2, str3, str4 = 0, 0, 0, 0
    tr_images, val_images = 0, 0
    with open(train_csv, 'wt', newline='') as write_file_tr:
        with open(val_csv, 'wt', newline='') as write_file_val:
            with open(all_csv, newline='') as read_file:
                writer_tr = csv.writer(write_file_tr, delimiter=',')
                writer_val = csv.writer(write_file_val, delimiter=',')
                reader = csv.reader(read_file)
                for row in reader:
                    save = 0
                    if tr_images < total_tr/2:
                        if (float(row[2]) > 0) and ((float(row[3]) + float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[3]) > 0) and ((float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[4]) > 0) and (float(row[5]) > 0):
                            save = 1
                        if save == 1:
                            print('Saved '+ str(row[0]))
                            writer_tr.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                            tr_images += 1

                    elif val_images < total_val/2:
                        if (float(row[2]) > 0) and ((float(row[3]) + float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[3]) > 0) and ((float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[4]) > 0) and (float(row[5]) > 0):
                            save = 1
                        if save == 1:
                            print('Saved '+ str(row[0]))
                            writer_val.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                            val_images += 1
                    elif tr_images < total_tr and float(row[10])==0:
                        if (float(row[2]) > 0) and ((float(row[3]) + float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[3]) > 0) and ((float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[4]) > 0) and (float(row[5]) > 0):
                            save = 1
                        if save == 1:
                            print('Saved '+ str(row[0]))
                            writer_tr.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                            tr_images += 1
                    elif val_images < total_val and float(row[10])==0:
                        if (float(row[2]) > 0) and ((float(row[3]) + float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[3]) > 0) and ((float(row[4]) + float(row[5])) > 0):
                            save = 1
                        elif (float(row[4]) > 0) and (float(row[5]) > 0):
                            save = 1
                        if save == 1:
                            print('Saved '+ str(row[0]))
                            writer_val.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                            val_images += 1
                            if val_images == 4:
                                break

    print('Created .csv\n')

def num_min_str(db_dir, path_db, min_tr=30):
    train_csv = os.path.join(db_dir,'train.csv')
    val_csv = os.path.join(db_dir,'val.csv')
    all_csv = os.path.join(path_db,'all_images.csv')
    min_val = (min_tr * 2)//10 # 20% of train images approximately
    num_str = 4
    strs = np.zeros([num_str,])
    strs_val = np.zeros([num_str,])
    strs_class = np.arange(0,num_str)
    tr_images, val_images = 0, 0
    end_melanomas = 520
    N = 0
    train_val = 0 # fill train images, then, activate validation
    with open(train_csv, 'wt', newline='') as write_file_tr:
        with open(val_csv, 'wt', newline='') as write_file_val:
            with open(all_csv, newline='') as read_file:
                writer_tr = csv.writer(write_file_tr, delimiter=',')
                writer_val = csv.writer(write_file_val, delimiter=',')
                reader = csv.reader(read_file)
                for row in reader:
                    present_strs_aux = np.array([0,float(row[3]),float(row[4]),float(row[5])])>0
                    present_strs = np.array([float(row[2]),float(row[3]),float(row[4]),float(row[5])])>0
                    present_strs_class = strs_class[present_strs_aux==1]
                    if train_val == 0:
                        for k in present_strs_class:
                            if strs[k] < (min_tr//2):
                                writer_tr.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                                strs += present_strs
                                N += 1
                                break

                        if np.sum(strs<(min_tr//2)) == 0:
                            train_val = 1
                            # strs = np.arange(0,num_str)
                    elif train_val == 1:
                        for k in present_strs_class:
                            if strs_val[k] < (min_val//2):
                                writer_val.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                                strs_val += present_strs
                                break

                        if np.sum(strs_val<(min_val//2)) == 0:
                            train_val = 2
                            # strs = np.arange(0,num_str)
                    elif train_val == 2 and int(row[0])>end_melanomas:
                        for k in present_strs_class:
                            if strs[k] < min_tr:
                                writer_tr.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                                strs += present_strs
                                N += 1
                                break

                        if np.sum(strs<min_tr) == 0:
                            train_val = 3
                            # strs = np.arange(0,num_str)

                    elif train_val == 3 and int(row[0])>end_melanomas:
                        for k in present_strs_class:
                            if strs_val[k] < min_val:
                                writer_val.writerow([row[6]] + [row[7]] + [row[8]] + [row[9]] + [row[10]])
                                strs_val += present_strs
                                break

                        if np.sum(strs_val<min_val) == 0:
                            break
                            
    print(strs)
    print(N)
    print('Created .csv\n')

def get_normalization():
    normalization = tfs_s.Normalize((0.6990122935081374, 0.5560426973085759, 0.5121212559185988), 
                                    (0.1576619536490988, 0.15625517092142197, 0.17061035673581912))
    # Obtenido: ([0.6990122935081374, 0.5560426973085759, 0.5121212559185988], [0.1576619536490988, 0.15625517092142197, 0.17061035673581912])
    # artículo:(0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
    # imagenet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    return normalization

def preprocess_data_train(opt, path, data_loading):
    # ISIC2017 and model > 2
    db_dir = os.path.join(path,'data_2017')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    if data_loading == 0:
        print('\nSelect data min structs...')
        path_db = os.path.join(choose_db(opt.global_path,1),'data_2017')
        num_min_str(db_dir, path_db, min_tr=opt.num_min_str)
    if data_loading == 1:
        print('\nSelect data...')
        path_db = os.path.join(choose_db(opt.global_path,1),'data_2017')
        select_data(opt.selected_images, db_dir, path_db)
    elif data_loading == 2:
        print('\nSelect data min2str...')
        path_db = os.path.join(choose_db(opt.global_path,1),'data_2017')
        select_data_min2structs(opt, db_dir, path_db)
    elif data_loading == 3:
        print('\nSelect structured data...')
        path_db = os.path.join(choose_db(opt.global_path,opt.db),'data_2017')
        if opt.preprocess == True:
            print('Preprocessing with structures\n')
            preprocess_data_2017_structures(root_dir=path_db, out_dir=db_dir, seg_dir='Train_Lesion', step_image=opt.step_image)

    normalize = get_normalization()
        
    if opt.over_sample:
        print('data is offline oversampled ...')
        train_file = os.path.join(db_dir,'train_oversample.csv')
    else:
        print('no offline oversampling ...')
        train_file = os.path.join(db_dir,'train.csv')

    val_file = os.path.join(db_dir,'val.csv')

    print('\nLoading dataset with structures...')

    transform_train = torch_transforms.Compose([
        tfs_s.RatioCenterCrop(0.8),
        tfs_s.Resize((256,256)),
        tfs_s.RandomCrop((224,224)),
        tfs_s.RandomRotate(),
        tfs_s.RandomHorizontalFlip(),
        tfs_s.RandomVerticalFlip(),
        tfs_s.ToTensor(),
        normalize
    ])
    
    transform_val = torch_transforms.Compose([
        tfs_s.RatioCenterCrop(0.8),
        tfs_s.Resize((256,256)),
        tfs_s.CenterCrop((224,224)),
        tfs_s.ToTensor(),
        normalize
    ])

    trainset = ISIC_structures(csv_file=train_file, transform=transform_train)
    valset = ISIC_structures(csv_file=val_file, transform=transform_val)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
        num_workers=8, worker_init_fn=_worker_init_fn_(), drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=8) # 64=batch_size

    return db_dir, train_file, trainloader, val_file, valloader

def preprocess_data_test(global_path, preprocess, batch_size, path):
    # ISIC2017 and model > 2
    db_dir = os.path.join(path,'data_2017')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    test_file = None
    test_results_file = None
    testloader = None
    
    print('\nSelect structured data...')
    path_db = os.path.join(choose_db(global_path,1),'data_2017')
    if preprocess == True:
        print('Preprocessing with structures\n')
        preprocess_data_2017_structures(root_dir=path_db, out_dir=db_dir, seg_dir='Train_Lesion', step_image=1)
        
    test_file = os.path.join(db_dir,'test.csv')
    test_results_file = os.path.join(db_dir,'test_results.csv')

    normalize = get_normalization()
        
    print('\nLoading dataset with structures...')

    transform_test = torch_transforms.Compose([
        tfs_s.RatioCenterCrop(0.8),
        tfs_s.Resize((256,256)),
        tfs_s.CenterCrop((224,224)),
        tfs_s.ToTensor(),
        normalize
    ])

    testset = ISIC_structures(csv_file=test_file, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return db_dir, test_file, test_results_file, testloader