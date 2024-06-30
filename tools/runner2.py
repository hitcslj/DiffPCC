import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json 
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from torch.nn import BCEWithLogitsLoss 
from torch.nn import CrossEntropyLoss
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import wandb 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") 
def run_net(args, config):
    logger = get_logger(args.log_name)
    # Build Dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = 1.0
    metrics = None 
     
    # builder.load_model(base_model, config.model.ckpt, logger = logger)

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        
    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)  
    # Criterion
    # BCELoss = BCEWithLogitsLoss()  
    CrossLoss = CrossEntropyLoss() 
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)  
    # Training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):  
        # metrics = validate(base_model, test_dataloader, epoch, CrossLoss, args, config, logger=logger)
        if args.distributed:    
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time() 
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # losses = AverageMeter(['BceLoss', 'Acc', 'Precision', 'Recall'])
        losses = AverageMeter(['CrossLoss', 'Acc'])

        num_iter = 0 
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader) 
        for idx, (taxonomy_ids, model_ids, data, sparse_voxel, _, _) in enumerate(train_dataloader): 
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI coarse_voxeltune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda() # B, N, 3
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                sparse_voxel = sparse_voxel.cuda()
                
                #normalize to [0,1]
                min_coords = torch.min(gt, dim=1, keepdim=True)[0] # B, 1, 3
                max_coords = torch.max(gt, dim=1, keepdim=True)[0] # B, 1, 3
                range_coords = max_coords - min_coords
                gt = (gt - min_coords) / (range_coords + 1e-3)
                partial = (partial - min_coords) / (range_coords + 1e-3) 
                
                # partial = misc.fps(gt, 2048)  
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1 
           
            coarse_voxel = base_model(partial)   
            crossloss = CrossLoss(coarse_voxel.reshape(-1, 9), sparse_voxel.reshape(-1,1).squeeze()) 
            _loss = crossloss  
              
            _loss.backward()  
            
            coarse_voxel = torch.softmax(coarse_voxel, dim=-1)
            _, coarse_voxel = torch.max(coarse_voxel, -1)
            accuracy = (coarse_voxel == sparse_voxel).float().mean()    
           

            # Forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed: 
                crossloss = dist_utils.reduce_tensor(crossloss, args) 
                accuracy = dist_utils.reduce_tensor(accuracy, args)
                losses.update([crossloss.item(), accuracy.item()])
            else: 
                losses.update([crossloss.item(), accuracy.item()]) 

            if args.distributed:
                torch.cuda.synchronize() 

            n_itr = epoch * n_batches + idx 
                
        

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()  
            
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            # metrics = validate(base_model, test_dataloader, epoch, BCELoss, args, config, logger=logger)
            metrics = validate(base_model, test_dataloader, epoch, CrossLoss, args, config, logger=logger)

            # Save checkpoints
            if  metrics < best_metrics:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if epoch > 0 and epoch % 10 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger) 
            
    run.finish()
    
def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    occnet = builder.model_builder(config.model) 
    CrossLoss = CrossEntropyLoss() 
    # load checkpoints
    builder.load_model(occnet, config.model.ckpt, logger = logger)  
    if args.use_gpu:
        occnet = occnet.to(args.local_rank)  
    #  DDP    
    if args.distributed:
        raise NotImplementedError() 
    validate(occnet, test_dataloader, -1, CrossLoss, args, config, logger=logger)
     
def validate(base_model, test_dataloader, epoch, CrossLoss, args, config, logger = None):  
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode 
    # test_losses = AverageMeter(['BceLoss', 'Acc', 'Precision', 'Recall'])  
    test_losses = AverageMeter(['CrossLoss', 'Acc'])
    n_samples = len(test_dataloader) # bs is 1 
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, sparse_voxel, _, _) in enumerate(test_dataloader): 
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0] 
            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                sparse_voxel = sparse_voxel.cuda() 
                
                #normalize to [0,1]
                min_coords = torch.min(gt, dim=1, keepdim=True)[0] # B, 1, 3
                max_coords = torch.max(gt, dim=1, keepdim=True)[0] # B, 1, 3
                range_coords = max_coords - min_coords
                gt = (gt - min_coords) / (range_coords + 1e-3)
                partial = (partial - min_coords) / (range_coords + 1e-3) 
                
                # partial = misc.fps(gt, 2048)  
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')  
           
            coarse_voxel = base_model(partial)   
            crossloss = CrossLoss(coarse_voxel.reshape(-1, 9), sparse_voxel.reshape(-1,1).squeeze())     
              
            coarse_voxel = torch.softmax(coarse_voxel, dim=-1)# bs, v, v, v, 9
            accuracy = (torch.max(coarse_voxel, -1)[1] == sparse_voxel).float().mean()  
            coarse_voxel1 = torch.sum(torch.arange(9).cuda().reshape(1, 1, 1, 1, 9) * coarse_voxel, dim=-1) # bs, v, v, v   
            coarse_voxel2 = torch.argmax(coarse_voxel, dim=-1) # bs, v, v, v
            coarse_voxel = (coarse_voxel1 + coarse_voxel2) / 2.0 
             
            if args.distributed:  
                crossloss = dist_utils.reduce_tensor(crossloss, args) 
                accuracy = dist_utils.reduce_tensor(accuracy, args) 
            
            test_losses.update([crossloss.item(), accuracy.item()])  
            
            if (idx + 1) % args.val_interval == 0: 
                gt_pc = gt.squeeze().detach().cpu().numpy()
                input_pc = partial.squeeze().detach().cpu().numpy()  
                sparse_voxel = sparse_voxel.squeeze().cpu().numpy() 
                coarse_voxel = coarse_voxel.squeeze().cpu().numpy() 
                vis_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(vis_path) and args.local_rank == 0:
                    os.mkdir(vis_path)
                img_path = vis_path + f'/{args.local_rank}_{idx}.gif'  
                misc.get_voxel_img(img_path, input_pc, sparse_voxel, coarse_voxel) 
                
            if idx % args.val_interval == 0:  
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s' %
                            (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()]), logger=logger) 
         
        print_log('[Validation] EPOCH: %d = %s' % (epoch, ['%.4f' % m for m in test_losses.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize() 
     
   
    return test_losses.avg(0)  
    
