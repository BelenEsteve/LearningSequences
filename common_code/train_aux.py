from loss import *
from utilities import *
from data import *
from visualization import *
from networks import *
from networks2 import *
import math
import scipy
import torch
import numpy as np
import random
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32
    random.seed(torch_seed)
    np.random.seed(np_seed)

def choose_criterion(option):
    # Aquí se podría mirar el meter la función aproximada a AUC pero diferenciable 
    if option == 0:
        print('use focal loss ...')
        criterion = FocalLoss(gama=2., size_average=True, weight=None)
    else:
        print('use cross entropy loss ...')
        criterion = nn.CrossEntropyLoss()
    print('done\n')

    return criterion

def define_GT(num_structs, y_mask, skin, str1, str2, str3, str4, device):
    # masks - [batch, opt.num_structs, dim, dim]
    if num_structs == 1:
        masks = torch.cat([y_mask], dim = 1).to(device) # lesion
    elif num_structs == 2:
        masks = torch.cat([y_mask, skin], dim = 1).to(device) # lesion and skin
    elif num_structs == 4:
        masks = torch.cat([str1, str2, str3, str4], dim = 1).to(device) # structures
    elif num_structs == 5:
        masks = torch.cat([y_mask, str1, str2, str3, str4], dim = 1).to(device) # lesion and structures
    elif num_structs == 6:
        masks = torch.cat([y_mask, skin, str1, str2, str3, str4], dim = 1).to(device) # lesion, skin and structures

    # structure classes
    real_classes = torch.zeros([masks.size(0),masks.size(1)], dtype=torch.int64).to(device)
    real_classes[:,:] = torch.tensor(np.arange(0,masks.size(1)).astype(np.int64)).to(device) # torch.Size([b_s, opt.t])

    return masks, real_classes

def train_class(opt):
    torch.cuda.empty_cache()

    ##### Create the arquitecture and load directories #####
    path, out_files, net = get_dirs_and_net(opt, device)

    ##### Load data #####
    print('\nloading the dataset ...')
    db_dir, train_file, trainloader, val_file, valloader, __, __, __= preprocess_data(opt, path, opt.data_loading)

    print('done\n')

    ############### Load models ###############
    ##### Choose the objective function or criterion #####
    criterion = choose_criterion(0)#opt.criterion)
    criterion.to(device)

    ##### Load checkpoints #####
    if opt.first_epoch != 0:
        checkpoint_lists, checkpoint = load_checkpoints(opt,out_files)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint_lists, checkpoint = None, None
        
    ##### Moving models to GPU #####
    print('\nmoving models to GPU ...')
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    print('done\n')

    ##### Load or create optimizer #####
    optimizer, scheduler_2 = load_or_create_optimizer_and_scheduler(opt, model, checkpoint)

    ##### Load scores and losses #####
    if opt.first_epoch != 0:
        # Load lists
        loss_train, loss_val_list = checkpoint_lists['loss_train'][:opt.first_epoch], checkpoint_lists['loss_val_list'][:opt.first_epoch]
        loss_train_class, loss_val_class = checkpoint_lists['loss_train_class'][:opt.first_epoch], checkpoint_lists['loss_val_class'][:opt.first_epoch]
        loss_train_str, loss_val_str = checkpoint_lists['loss_train_str'][:opt.first_epoch], checkpoint_lists['loss_val_str'][:opt.first_epoch]
        step_list = checkpoint_lists['step_list'][:opt.first_epoch]
        
        step = step_list[-1]

        AUC_train, AP_train = checkpoint_lists['AUC_train'][:opt.first_epoch], checkpoint_lists['AP_train'][:opt.first_epoch]
        AUC_val, AP_val = checkpoint_lists['AUC_val'][:opt.first_epoch], checkpoint_lists['AP_val'][:opt.first_epoch]

        # train lists
        precision_mel_train_list = checkpoint_lists['precision_mel_train_list'][:opt.first_epoch]
        recall_mean_train_list = checkpoint_lists['recall_mean_train_list'][:opt.first_epoch]
        recall_mel_train_list = checkpoint_lists['recall_mel_train_list'][:opt.first_epoch]
        precision_mean_train_list = checkpoint_lists['precision_mean_train_list'][:opt.first_epoch]
        # val lists
        precision_mel_val_list = checkpoint_lists['precision_mel_val_list'][:opt.first_epoch]
        recall_mean_val_list = checkpoint_lists['recall_mean_val_list'][:opt.first_epoch]
        recall_mel_val_list = checkpoint_lists['recall_mel_val_list'][:opt.first_epoch]
        precision_mean_val_list = checkpoint_lists['precision_mean_val_list'][:opt.first_epoch]

    else:
        step = 0
        loss_train, loss_val_list = [], []
        loss_train_class, loss_val_class = [], []
        loss_train_str, loss_val_str = [], []
        step_list = []
        
        AUC_train, AUC_val = [], []
        AP_train, AP_val = [], []

        precision_mel_train_list, precision_mel_val_list = [], []
        recall_mean_train_list, recall_mean_val_list = [], []
        recall_mel_train_list, recall_mel_val_list = [], []
        precision_mean_train_list, precision_mean_val_list = [], []

    # Losses through every iteration
    loss_it_train_class, loss_it_train_str = [], []
    loss_it_train_str_class, loss_it_train_str_mask = [], []
    loss_it_val_class, loss_it_val_str = [], []
    loss_it_val_str_class, loss_it_val_str_mask = [], []
    
    train_images = len(trainloader) * opt.batch_size
    val_images = len(valloader)* opt.batch_size

    ##### Training #####
    print('\nstart training ...\n')
    for epoch in np.arange(opt.first_epoch,opt.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print("\nepoch %d learning rate %f\n" % (epoch, current_lr))
        running_loss, running_loss_class, running_loss_str = 0, 0, 0
        
        # run for one epoch
        for i, data in enumerate(trainloader, 0):
            torch.cuda.empty_cache()
            # warm up
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            loss_structures, loss_classification = torch.zeros([]), torch.zeros([])

            # load inputs and GT
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
                
            # forward
            pred, At, out_masks, out_classes, activation_post_attmod, activation_pre_attmod = model(inputs)

            loss_classification = criterion(pred, labels)

            loss = opt.str_mul * loss_structures + opt.class_mul * loss_classification

            loss_it_train_class.append(opt.class_mul * loss_classification)
            loss_it_train_str.append(opt.str_mul * loss_structures)
            loss_it_train_str_class.append(0)
            loss_it_train_str_mask.append(0)

            running_loss += loss.cpu().detach().numpy()
            running_loss_class += loss_classification.cpu().detach().numpy()
            running_loss_str += loss_structures.cpu().detach().numpy()

            loss.backward()
            optimizer.step()
            # display results
            if i % opt.iters_visualization == 0:
                print("[epoch %d][iter %d/%d]" % (epoch, i+1, len(trainloader)))

                if i != 0:
                    paint_it_losses(loss_it_train_class, loss_it_train_str, loss_it_train_str_class, loss_it_train_str_mask)
                
                predict = torch.argmax(pred, 1)
                total = labels.size(0)
                correct = torch.eq(predict, labels).sum().double().item()
                accuracy = correct / total
                print("loss %.4f accuracy %.2f%%"
                    % (loss.item(), (100*accuracy)))

        step += 1
            
        # adjust learning rate
        if opt.enable_scheduler == 1:
            scheduler_2.step()

        ##### (every epoch) Save train results (just classification) ##### 
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            results = os.path.join(db_dir,'train_results.csv')
            file2 = os.path.join(db_dir,'train_aux.csv')
            with open(results, 'wt', newline='') as csv_file:
                with open(file2, 'wt', newline='') as csv_file2:
                    csv_writer, csv_writer2 = csv.writer(csv_file, delimiter=','), csv.writer(csv_file2, delimiter=',')
                    for i, data in enumerate(trainloader, 0):
                        images, labels = data['image'], data['label']
                        images, labels = images.to(device), labels.to(device)

                        pred, __, __, __, __, __ = model(images)

                        predict = torch.argmax(pred, 1)
                        total += labels.size(0)
                        correct += torch.eq(predict, labels).sum().double().item()

                        # record prediction
                        responses = F.softmax(pred, dim=1).squeeze().cpu().numpy()
                        responses = [responses[i] for i in range(responses.shape[0])]
                        csv_writer.writerows(responses)

                        # record real labels
                        responses = labels.squeeze().cpu().numpy()                        
                        responses = [['x','x',responses[i]] for i in range(responses.shape[0])]
                        csv_writer2.writerows(responses)
                              
            AP_t, AUC_t, precision_mean_t, precision_mel_t, recall_mean_t, recall_mel_t = compute_metrics(results, file2)
        
        print("\n[epoch %d] train result:" % (epoch))
        print("\nAP %.4f AUC %.4f" % (AP_t, AUC_t))
        print("accuracy %.2f%%" % (100*correct/total))
        print("mean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
                (100*precision_mean_t, 100*recall_mean_t, 100*precision_mel_t, 100*recall_mel_t))
        print("\n")

        ##### (every epoch) Validation results #####
        print('\nValidation...')
        with torch.no_grad():
            total, correct = 0, 0
            running_loss_val, running_loss_val_class, running_loss_val_str = 0, 0, 0
            results = os.path.join(db_dir,'val_results.csv')
            file2 = os.path.join(db_dir,'val_aux.csv')
            with open(results, 'wt', newline='') as csv_file:
                with open(file2, 'wt', newline='') as csv_file2:
                    csv_writer, csv_writer2 = csv.writer(csv_file, delimiter=','), csv.writer(csv_file2, delimiter=',')
                    for i, data in enumerate(valloader, 0):
                        loss_structures, loss_classification = torch.zeros([]), torch.zeros([])

                        images, labels = data['image'], data['label']
                        images, labels = images.to(device), labels.to(device)
                        
                        pred, At, out_masks, out_classes, activation_post_attmod, activation_pre_attmod = model(images)

                        loss_classification = criterion(pred, labels)

                        predict = torch.argmax(pred, 1)
                        total += labels.size(0)
                        correct += torch.eq(predict, labels).sum().double().item()
                    
                        # record prediction
                        responses = F.softmax(pred, dim=1).squeeze().cpu().numpy()
                        responses = [responses[i] for i in range(responses.shape[0])]
                        csv_writer.writerows(responses)

                        # record real labels
                        responses = labels.squeeze().cpu().numpy()                        
                        responses = [['x','x',responses[i]] for i in range(responses.shape[0])]
                        csv_writer2.writerows(responses)

                        # Validation loss
                        loss_val = opt.str_mul * loss_structures + opt.class_mul * loss_classification
                        loss_it_val_class.append(opt.class_mul * loss_classification)
                        loss_it_val_str.append(opt.str_mul * loss_structures)
                        loss_it_val_str_class.append(0)
                        loss_it_val_str_mask.append(0)
                        
                        running_loss_val += loss_val.cpu().detach().numpy()
                        running_loss_val_class += loss_classification.cpu().detach().numpy()
                        running_loss_val_str += loss_structures.cpu().detach().numpy()

                        if i % 1 == 0:
                            print("[epoch %d][iter %d/%d]" % (epoch, i+1, len(valloader)))

                            if i != 0:
                                paint_it_losses(loss_it_val_class, loss_it_val_str,loss_it_val_str_class,loss_it_val_str_mask)

                            predict = torch.argmax(pred, 1)
                            total = labels.size(0)
                            correct = torch.eq(predict, labels).sum().double().item()
                            accuracy = correct / total
                            print("loss %.4f accuracy %.2f%%"
                                % (loss.item(), (100*accuracy)))

            AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics(results, file2)

        ## SAVE CHECKPOINTS
        print('\nsaving checkpoints ...\n')
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'lr': current_lr
        }
        checkpoint_name = 'checkpoint_' + opt.ch_name + str(epoch) + '.pth'
        torch.save(checkpoint, os.path.join(out_files, checkpoint_name))

        print('\nsaved checkpoint ' +  checkpoint_name + '...\n')

        # Checkpoint for global variables:
        loss_train.append(running_loss / train_images)
        loss_train_class.append(running_loss_class / train_images)
        loss_train_str.append(running_loss_str / train_images)
        loss_val_list.append(running_loss_val / val_images)
        loss_val_class.append(running_loss_val_class / val_images)
        loss_val_str.append(running_loss_val_str / val_images)
        step_list.append(step)

        AUC_train.append(AUC_t)
        AP_train.append(AP_t)
        precision_mel_train_list.append(precision_mel_t)
        recall_mean_train_list.append(recall_mean_t)
        recall_mel_train_list.append(recall_mel_t)
        precision_mean_train_list.append(precision_mean_t)

        AUC_val.append(AUC)
        AP_val.append(AP)
        precision_mel_val_list.append(precision_mel)
        recall_mean_val_list.append(recall_mean)
        recall_mel_val_list.append(recall_mel)
        precision_mean_val_list.append(precision_mean)

        checkpoint = {
            # Common variables
            'loss_train': loss_train,
            'loss_val_list': loss_val_list,
            'step_list': step_list,
            'loss_train_class': loss_train_class,
            'loss_train_str': loss_train_str,
            'loss_val_class': loss_val_class,
            'loss_val_str': loss_val_str,
            # Classifier
            'AUC_val': AUC_val,
            'AUC_train': AUC_train,
            'AP_val': AP_val,
            'AP_train': AP_train,
            'precision_mel_val_list': precision_mel_val_list,
            'recall_mean_val_list': recall_mean_val_list, 
            'recall_mel_val_list': recall_mel_val_list,
            'precision_mean_val_list': precision_mean_val_list,
            'precision_mel_train_list': precision_mel_train_list,
            'recall_mean_train_list': recall_mean_train_list,
            'recall_mel_train_list': recall_mel_train_list,
            'precision_mean_train_list': precision_mean_train_list,
            # Decoder
            'jaccard_scores_train': jaccard_scores_train,
            'jaccard_scores_val': jaccard_scores_val,
            'avg_prec_scores_train': avg_prec_scores_train,
            'avg_prec_scores_val': avg_prec_scores_val,
            'prec_scores_train': prec_scores_train,
            'prec_scores_val': prec_scores_val
        }

        checkpoint_name = 'lists'+ opt.ch_name + '.pth'
        torch.save(checkpoint, os.path.join(out_files, checkpoint_name))

        print('\nsaved checkpoint ' +  checkpoint_name + '...\n')

        print("\n[epoch %d] val result:" % (epoch))
        print("\nAP %.4f AUC %.4f" % (AP, AUC))
        print("accuracy %.2f%%" % (100*correct/total))
        print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
                (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))

    print('\nEnd training ' + str(opt.epochs) + '\n')
    

def train_deco(opt):
    torch.cuda.empty_cache()

    ##### Create the arquitecture and load directories #####
    path, out_files, net = get_dirs_and_net(opt, device)

    ##### Load data #####
    print('\nloading the dataset ...')
    db_dir, train_file, trainloader, val_file, valloader, __, __, __= preprocess_data(opt, path, opt.data_loading)

    print('done\n')

    ############### Load models ###############
    ##### Choose the objective function or criterion #####
    mask_siou = softIoULoss().to(device)                                               # Segmentation loss
    class_crit = MaskedNLLLoss(balance_weight=None).to(device)                         # Structure classification loss (class_xentropy)

    ##### Load checkpoints #####
    if opt.first_epoch != 0:
        checkpoint_lists, checkpoint = load_checkpoints(opt,out_files)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint_lists, checkpoint = None, None
        
    ##### Moving models to GPU #####
    print('\nmoving models to GPU ...')
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    print('done\n')

    ##### Load or create optimizer #####
    optimizer, scheduler_2 = load_or_create_optimizer_and_scheduler(opt, model, checkpoint)

    ##### Load scores and losses #####
    if opt.first_epoch != 0:
        # Load lists
        loss_train, loss_val_list = checkpoint_lists['loss_train'][:opt.first_epoch], checkpoint_lists['loss_val_list'][:opt.first_epoch]
        loss_train_class, loss_val_class = checkpoint_lists['loss_train_class'][:opt.first_epoch], checkpoint_lists['loss_val_class'][:opt.first_epoch]
        loss_train_str, loss_val_str = checkpoint_lists['loss_train_str'][:opt.first_epoch], checkpoint_lists['loss_val_str'][:opt.first_epoch]
        step_list = checkpoint_lists['step_list'][:opt.first_epoch]
        
        step = step_list[-1]

        jaccard_scores_train = torch.zeros([opt.epochs, opt.num_structs])
        jaccard_scores_train[:opt.first_epoch,:] = checkpoint_lists['jaccard_scores_train'][:opt.first_epoch,:]
        jaccard_scores_val = torch.zeros([opt.epochs, opt.num_structs])
        jaccard_scores_val[:opt.first_epoch,:] = checkpoint_lists['jaccard_scores_val'][:opt.first_epoch,:]
        avg_prec_scores_train = torch.zeros([opt.epochs])
        avg_prec_scores_train[:opt.first_epoch] = checkpoint_lists['avg_prec_scores_train'][:opt.first_epoch]
        avg_prec_scores_val = torch.zeros([opt.epochs])
        avg_prec_scores_val[:opt.first_epoch] = checkpoint_lists['avg_prec_scores_val'][:opt.first_epoch]
        prec_scores_train = torch.zeros([opt.epochs, len(opt.th_scores)])
        prec_scores_train[:opt.first_epoch,:] = checkpoint_lists['prec_scores_train'][:opt.first_epoch,:]
        prec_scores_val = torch.zeros([opt.epochs, len(opt.th_scores)])
        prec_scores_val[:opt.first_epoch,:] = checkpoint_lists['prec_scores_val'][:opt.first_epoch,:]

    else:
        step = 0
        loss_train, loss_val_list = [], []
        loss_train_class, loss_val_class = [], []
        loss_train_str, loss_val_str = [], []
        step_list = []

        jaccard_scores_train = torch.zeros([opt.epochs, opt.num_structs])
        jaccard_scores_val = torch.zeros([opt.epochs, opt.num_structs])
        avg_prec_scores_train = torch.zeros([opt.epochs])
        avg_prec_scores_val = torch.zeros([opt.epochs])
        prec_scores_train = torch.zeros([opt.epochs, len(opt.th_scores)])
        prec_scores_val = torch.zeros([opt.epochs, len(opt.th_scores)])

    # Losses through every iteration
    loss_it_train_class, loss_it_train_str = [], []
    loss_it_train_str_class, loss_it_train_str_mask = [], []
    loss_it_val_class, loss_it_val_str = [], []
    loss_it_val_str_class, loss_it_val_str_mask = [], []
    
    train_images = len(trainloader) * opt.batch_size
    val_images = len(valloader)* opt.batch_size

    ##### Training #####
    print('\nstart training ...\n')
    for epoch in np.arange(opt.first_epoch,opt.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print("\nepoch %d learning rate %f\n" % (epoch, current_lr))
        running_loss, running_loss_class, running_loss_str = 0, 0, 0
        jaccard_aux, avg_prec_aux, prec_aux = [], [], []
        
        # run for one epoch
        for i, data in enumerate(trainloader, 0):
            torch.cuda.empty_cache()
            # warm up
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            loss_structures, loss_classification = torch.zeros([]), torch.zeros([])

            # load inputs and GT
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            y_mask, skin, str1, str2, str3, str4 = data['image_seg'], data['skin'], data['str_1'], data['str_2'], data['str_3'], data['str_4']
            y_mask, skin, str1, str2, str3, str4 = y_mask.to(device), skin.to(device), str1.to(device), str2.to(device), str3.to(device), str4.to(device)

            masks, real_classes = define_GT(opt.num_structs, y_mask, skin, str1, str2, str3, str4, device)
                
            # forward
            pred, At, out_masks, out_classes, activation_post_attmod, activation_pre_attmod = model(inputs)

            loss_structures, loss_mask_iou, loss_class, y_mask_perm, y_class_perm, out_masks2, scores = compute_loss_structures_train(opt, masks, out_masks, real_classes, out_classes, 
                                                                                                                                mask_siou, class_crit, device)
            jc, out_masks_th = compute_jaccard(y_mask_perm, out_masks2, y_class_perm, opt.num_structs, opt.threshold)
            jaccard_aux.append(jc)

            y_mask_perm_list = []
            out_masks_list = []
            for batch in range(y_mask_perm.size(0)):
                for el in range(y_mask_perm.size(1)):
                    y_mask_perm_list.append(y_mask_perm[batch,el,:].cpu().detach().numpy())
                    out_masks_list.append(out_masks2[batch,el,:].cpu().detach().numpy())
            avg_prec, precision = calculate_average_precision(y_mask_perm_list, out_masks_th.cpu().numpy(), thresholds=opt.th_scores)
            avg_prec_aux.append(avg_prec)
            prec_aux.append(precision)

            loss = opt.str_mul * loss_structures 

            loss_it_train_class.append(0)
            loss_it_train_str.append(opt.str_mul * loss_structures)
            loss_it_train_str_class.append(opt.class_weight*loss_class)
            loss_it_train_str_mask.append(opt.iou_weight * loss_mask_iou)

            running_loss += loss.cpu().detach().numpy()
            running_loss_class += loss_classification.cpu().detach().numpy()
            running_loss_str += loss_structures.cpu().detach().numpy()

            loss.backward()
            optimizer.step()
            # display results
            if i % opt.iters_visualization == 0:
                print("[epoch %d][iter %d/%d]" % (epoch, i+1, len(trainloader)))

                if i != 0:
                    paint_it_losses(loss_it_train_class, loss_it_train_str, loss_it_train_str_class, loss_it_train_str_mask)

                if opt.model == 5:
                    visualize_all_attn(inputs, At, idx=0)
                visualize_all_masks(inputs, masks, y_mask_perm, out_masks, y_class_perm, opt.threshold, opt.num_structs, idx=0)
                
                # Mean jc and avg_prec along training
                jc = torch.mean(torch.cat(jaccard_aux,dim=0),dim=0)
                avg_prec = np.mean(np.array(avg_prec_aux))
                prec = np.mean(np.array(prec_aux),axis=0).tolist()

                print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)

        step += 1
            
        # adjust learning rate
        if opt.enable_scheduler == 1:
            scheduler_2.step()
        
        ##### (every epoch) Save train results (Jaccard index) ##### 
        jaccard_scores_train[epoch,:] = torch.tensor(np.mean(np.array(jaccard_aux), axis=1).squeeze())
        avg_prec_scores_train[epoch] = torch.tensor(np.mean(np.array(avg_prec_aux)))
        prec_scores_train[epoch,:] = torch.tensor(np.mean(np.array(prec_aux), axis=0))

        print("\n[epoch %d] train result:" % (epoch))
        jc = jaccard_scores_train[epoch,:]
        avg_prec = avg_prec_scores_train[epoch]
        prec = prec_scores_train[epoch,:]
        print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)
                
        ##### (every epoch) Validation results #####
        print('\nValidation...')
        with torch.no_grad():
            total, correct = 0, 0
            running_loss_val, running_loss_val_class, running_loss_val_str = 0, 0, 0
            jaccard_aux, avg_prec_aux, prec_aux = [], [], []
            for i, data in enumerate(valloader, 0):
                loss_structures, loss_classification = torch.zeros([]), torch.zeros([])

                images, labels = data['image'], data['label']
                images, labels = images.to(device), labels.to(device)
                # Structure order in dataloader: 'pigment_network','negative_network','milia_like_cyst','streaks' 
                y_mask, skin, str1, str2, str3, str4 = data['image_seg'], data['skin'], data['str_1'], data['str_2'], data['str_3'], data['str_4']
                y_mask, skin, str1, str2, str3, str4 = y_mask.to(device), skin.to(device), str1.to(device), str2.to(device), str3.to(device), str4.to(device)

                # masks - [batch, opt.num_structs, dim, dim]
                masks, real_classes = define_GT(opt.num_structs, y_mask, skin, str1, str2, str3, str4, device)
                
                pred, At, out_masks, out_classes, activation_post_attmod, activation_pre_attmod = model(images)

                loss_structures, loss_mask_iou, loss_class, y_mask_perm, y_class_perm, out_masks2 = compute_loss_structures_eval(opt, masks, out_masks, real_classes, out_classes, 
                                                                                                                                    mask_siou, class_crit, device)
                jc, out_masks_th = compute_jaccard(y_mask_perm, out_masks2, y_class_perm, opt.num_structs, opt.threshold)
                jaccard_aux.append(jc)

                y_mask_perm_list = []
                out_masks_list = []
                for batch in range(y_mask_perm.size(0)):
                    for el in range(y_mask_perm.size(1)):
                        y_mask_perm_list.append(y_mask_perm[batch,el,:].cpu().detach().numpy())
                        out_masks_list.append(out_masks2[batch,el,:].cpu().detach().numpy())
                avg_prec, precision = calculate_average_precision(y_mask_perm_list, out_masks_th.cpu().numpy(), thresholds=opt.th_scores)
                avg_prec_aux.append(avg_prec)
                prec_aux.append(precision)

                # Validation loss
                loss_val = opt.str_mul * loss_structures 
                loss_it_val_class.append(0)
                loss_it_val_str.append(opt.str_mul * loss_structures)
                loss_it_val_str_class.append(opt.class_weight*loss_class)
                loss_it_val_str_mask.append(opt.iou_weight * loss_mask_iou)
                
                running_loss_val += loss_val.cpu().detach().numpy()
                running_loss_val_class += loss_classification.cpu().detach().numpy()
                running_loss_val_str += loss_structures.cpu().detach().numpy()

                if i % 1 == 0:
                    print("[epoch %d][iter %d/%d]" % (epoch, i+1, len(valloader)))

                    if i != 0:
                        paint_it_losses(loss_it_val_class, loss_it_val_str,loss_it_val_str_class,loss_it_val_str_mask)

                    if opt.model == 5:
                        visualize_all_attn(images, At, idx=0)
                    visualize_all_masks(images, masks, y_mask_perm, out_masks, y_class_perm, opt.threshold, opt.num_structs, idx=0)
                    
                    # Mean jc and avg_prec along training
                    jc = torch.mean(torch.cat(jaccard_aux,dim=0),dim=0)
                    avg_prec = np.mean(np.array(avg_prec_aux))
                    prec = np.mean(np.array(prec_aux),axis=0).tolist()
                    
                    print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)

            jaccard_scores_val[epoch,:] = torch.tensor(np.mean(np.array(jaccard_aux), axis=1).squeeze())
            avg_prec_scores_val[epoch] = torch.tensor(np.mean(np.array(avg_prec_aux)))
            prec_scores_val[epoch,:] = torch.tensor(np.mean(np.array(prec_aux), axis=0))

        ## SAVE CHECKPOINTS
        print('\nsaving checkpoints ...\n')
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'lr': current_lr
        }
        checkpoint_name = 'checkpoint_' + opt.ch_name + str(epoch) + '.pth'
        torch.save(checkpoint, os.path.join(out_files, checkpoint_name))

        print('\nsaved checkpoint ' +  checkpoint_name + '...\n')

        # Checkpoint for global variables:
        loss_train.append(running_loss / train_images)
        loss_train_class.append(running_loss_class / train_images)
        loss_train_str.append(running_loss_str / train_images)
        loss_val_list.append(running_loss_val / val_images)
        loss_val_class.append(running_loss_val_class / val_images)
        loss_val_str.append(running_loss_val_str / val_images)
        step_list.append(step)
       
        checkpoint = {
            # Common variables
            'loss_train': loss_train,
            'loss_val_list': loss_val_list,
            'step_list': step_list,
            'loss_train_class': loss_train_class,
            'loss_train_str': loss_train_str,
            'loss_val_class': loss_val_class,
            'loss_val_str': loss_val_str,
            # Decoder
            'jaccard_scores_train': jaccard_scores_train,
            'jaccard_scores_val': jaccard_scores_val,
            'avg_prec_scores_train': avg_prec_scores_train,
            'avg_prec_scores_val': avg_prec_scores_val,
            'prec_scores_train': prec_scores_train,
            'prec_scores_val': prec_scores_val
        }                

        checkpoint_name = 'lists'+ opt.ch_name + '.pth'
        torch.save(checkpoint, os.path.join(out_files, checkpoint_name))

        print('\nsaved checkpoint ' +  checkpoint_name + '...\n')

        print("\n[epoch %d] val result:" % (epoch))

        jc = jaccard_scores_val[epoch,:]
        avg_prec = avg_prec_scores_val[epoch]
        prec = prec_scores_val[epoch,:]
        print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)

    print('\nEnd training ' + str(opt.epochs) + '\n')

def test_fun(opt):

    ##### Create the arquitecture and load directories #####
    path, out_files, net = get_dirs_and_net(opt, device)

    ##### Load data #####
    print('\nloading the dataset ...')
    db_dir, __, __, __, __, test_file, test_results_file, testloader = preprocess_data(opt, path, opt.data_loading)
    
    print('done\n')

    ##### Load model #####
    print('\nloading the model ...')
    
    checkpoint_name = 'checkpoint_' + opt.ch_name + str(opt.checkpoint) + '.pth'
    checkpoint = torch.load(os.path.join(out_files, opt.checkpoint_name))
    net.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()
    print('done\n')

    ##### Choose the objective function/s #####
    if opt.enable_classifier == 1:
        criterion = choose_criterion(opt)
        criterion.to(device)
    
    if opt.enable_decoder == 1:
        # Objective functions
        mask_siou = softIoULoss().to(device)                           # Segmentation loss
        class_crit = MaskedNLLLoss(balance_weight=None).to(device)     # Structure classification loss (class_xentropy)

    ##### Testing #####
    print('\nstart testing ...\n')
    total, correct = 0, 0
    running_loss_test = 0
    running_loss_test_class, running_loss_test_str = 0, 0
    jaccard_aux, avg_prec_aux, prec_aux = [], []
    with torch.no_grad():
        loss_it_class, loss_it_str = [], []
        loss_it_str_class, loss_it_str_mask = [], []
        test_file2 = os.path.join(db_dir,'test_aux.csv')
        with open(test_results_file, 'wt', newline='') as csv_file:
            with open(test_file2, 'wt', newline='') as csv_file2:
                csv_writer, csv_writer2 = csv.writer(csv_file, delimiter=','), csv.writer(csv_file2, delimiter=',')
                for i, data in enumerate(testloader, 0):
                    images, labels = data['image'], data['label']
                    images, labels = images.to(device), labels.to(device)

                    loss_structures, loss_classification = torch.zeros([]), torch.zeros([])

                    # Structure order in dataloader: 'pigment_network','negative_network','milia_like_cyst','streaks' 
                    y_mask, skin, str1, str2, str3, str4 = data['image_seg'], data['skin'], data['str_1'], data['str_2'], data['str_3'], data['str_4']
                    y_mask, skin, str1, str2, str3, str4 = y_mask.to(device), skin.to(device), str1.to(device), str2.to(device), str3.to(device), str4.to(device)

                    # masks - [batch, opt.t, dim, dim]
                    masks, real_classes = define_GT(opt.num_structs, y_mask, skin, str1, str2, str3, str4, device)

                    if opt.model == 5:
                        pred, At, out_masks, out_masks_sft, out_classes, activation_post_attmod, activation_pre_attmod = model(images)
                    else:
                        pred, At, out_masks, out_masks_sft, out_classes = model(images)

                    if opt.enable_decoder == 1:
                        loss_structures, loss_mask_iou, loss_class, y_mask_perm, y_class_perm, out_masks2 = compute_loss_structures_eval(opt, masks, out_masks, real_classes, out_classes, 
                                                                                                                                            mask_siou, class_crit, device)
                        jc, out_masks_th = compute_jaccard(y_mask_perm, out_masks2, y_class_perm, opt.num_structs, opt.threshold)
                        jaccard_aux.append(jc)

                        y_mask_perm_list = []
                        out_masks_list = []
                        for batch in range(y_mask_perm.size(0)):
                            for el in range(y_mask_perm.size(1)):
                                y_mask_perm_list.append(y_mask_perm[batch,el,:].cpu().detach().numpy())
                                out_masks_list.append(out_masks2[batch,el,:].cpu().detach().numpy())
                        avg_prec, precision = calculate_average_precision(y_mask_perm_list, out_masks_th.cpu().numpy(), thresholds=opt.th_scores)
                        avg_prec_aux.append(avg_prec)
                        prec_aux.append(precision)

                    if opt.enable_classifier == 1:
                        loss_classification = criterion(pred, labels)

                        predict = torch.argmax(pred, 1)
                        total += labels.size(0)
                        correct += torch.eq(predict, labels).sum().double().item()
                    
                        # record prediction
                        responses = F.softmax(pred, dim=1).squeeze().cpu().numpy()
                        responses = [responses[i] for i in range(responses.shape[0])]
                        csv_writer.writerows(responses)

                        # record real labels
                        responses = labels.squeeze().cpu().numpy()                        
                        responses = [['x','x',responses[i]] for i in range(responses.shape[0])]
                        csv_writer2.writerows(responses)

                    # Validation loss
                    loss_test = opt.str_mul * loss_structures + opt.class_mul * loss_classification
                    loss_it_class.append(opt.class_mul * loss_classification)
                    loss_it_str.append(opt.str_mul * loss_structures)
                    loss_it_str_class.append(opt.class_weight * loss_class)
                    loss_it_str_mask.append(opt.iou_weight * loss_mask_iou)
                    
                    running_loss_test += loss_test.cpu().detach().numpy()
                    running_loss_test_class += loss_classification.cpu().detach().numpy()
                    running_loss_test_str += loss_structures.cpu().detach().numpy()

                    if i % 1 == 0:
                        print("[iter %d/%d]" % (i+1, len(valloader)))

                        if i != 0:
                            paint_it_losses(loss_it_val_class, loss_it_val_str)

                        if opt.enable_classifier == 1:
                            predict = torch.argmax(pred, 1)
                            total = labels.size(0)
                            correct = torch.eq(predict, labels).sum().double().item()
                            accuracy = correct / total
                            print("loss %.4f accuracy %.2f%%"
                                % (loss.item(), (100*accuracy)))

                        if opt.enable_decoder == 1:
                            if opt.model == 5:
                                visualize_all_attn(inputs, At, idx=0)
                            visualize_all_masks(inputs, masks, y_mask_perm, out_masks, y_class_perm, opt.threshold, opt.num_structs, idx=0)
                            
                            # Mean jc and avg_prec along training
                            jc = torch.mean(torch.cat(jaccard_aux,dim=0),dim=0)
                            avg_prec = np.mean(np.array(avg_prec_aux))
                            prec = np.mean(np.array(prec_aux),axis=0).tolist()
                            
                            print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)

    AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics(test_results_file, test_file2)

    print("\ntest result:")
    if opt.enable_classifier == 1:
        print("accuracy %.2f%%" % (100*correct/total))
        print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
                (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
    if opt.enable_decoder == 1:  
        # Mean jc and avg_prec along training
        jc = torch.mean(torch.cat(jaccard_aux,dim=0),dim=0)
        avg_prec = np.mean(np.array(avg_prec_aux))
        prec = torch.mean(torch.cat(prec_aux,dim=0),dim=0)
        print_seg_scores(opt.num_structs, jc, avg_prec, prec, opt.th_scores)

    print('\nEnd testing')