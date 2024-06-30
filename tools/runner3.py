import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import json
from copy import deepcopy 
import wandb 
import warnings
from utils.logger import *
from tools import builder
from utils import misc, dist_utils
import time
from collections import OrderedDict
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
import wandb 
warnings.filterwarnings("ignore")
import math
'''
some utils
'''
@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name.startswith('model.module'):
            name = name.replace('model.module.', 'model.')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag 
         
    
def run_net(args, config):

    logger = get_logger(args.log_name)
    # Build Dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # Build Model
    model = builder.model_builder(config.model) 
    
    occnet_train = builder.model_builder(config.occnet)
    occnet_eval = builder.model_builder(config.occnet)

    if args.use_gpu:
        model.to(args.local_rank) 
        occnet_train.to(args.local_rank)
        occnet_eval.to(args.local_rank)
    
    builder.load_model(occnet_train, config.occnet.train_ckpt, logger = logger)
    builder.load_model(occnet_eval, config.occnet.eval_ckpt, logger = logger)
    occnet_train.eval()
    occnet_eval.eval() 
    # model.load_state_dict(torch.load(config.model.ckpt, map_location='cpu')['model_state']) 

    # Parameter Setting 
    start_epoch = 0 

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(model, args.start_ckpts, logger = logger) 
    

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn: 
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        model = nn.parallel.DistributedDataParallel(model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        model = nn.DataParallel(model).cuda()
        
    # Optimizer & Scheduler
    optimizer, _ = builder.build_opti_sche(model, config)  
    # run = wandb.init(
    #             project='diffpc', 
    #             name='train',
    #         )
    # wandb.watch(model)  

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)   
    
    # Note that parameter initialization is done within the DiT constructor
    if config.use_ema:
        ema = deepcopy(model).to(args.local_rank)  # Create an EMA of the model for use after training
        requires_grad(ema, False)   

    # Prepare models for training:
    if config.use_ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
    
    # Training
    model.zero_grad() 
    for epoch in range(start_epoch, config.max_epoch + 1):  
        if args.distributed: 
            train_sampler.set_epoch(epoch)
            
        epoch_start_time = time.time() 
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['MSELoss'])
        testlosses = AverageMeter(['MSELoss'])
        model.train()  
        num_iter = 0
        n_batches = len(train_dataloader) 
        for idx, (taxonomy_ids, model_ids, data, sparse_voxel, dense_voxel, y) in enumerate(train_dataloader): 
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
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
                
            x = dense_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3).cuda()
            noises_batch = torch.randn(x.shape).cuda() 
            y = y.cuda()
            
            num_iter += 1 
            '''  
            train diffusion
            '''
            with torch.no_grad(): 
                coarse_voxel  = occnet_train(partial)     
                coarse_voxel = torch.softmax(coarse_voxel, dim=-1)# bs, v, v, v, 9 
                coarse_voxel1 = torch.sum(torch.arange(9).cuda().reshape(1, 1, 1, 1, 9) * coarse_voxel, dim=-1) # bs, v, v, v   
                coarse_voxel2 = torch.argmax(coarse_voxel, dim=-1) # bs, v, v, v
                coarse_voxel = (coarse_voxel1 + coarse_voxel2) / 2.0 
                
                #normalize to [-1,1] 
                coarse_voxel = (coarse_voxel - 4.0) / 4.0  
                
                avg = torch.mean(coarse_voxel.reshape(coarse_voxel.shape[0], -1), dim=-1, keepdim=True) # bs, 1
                avg = avg.reshape(coarse_voxel.shape[0], 1, 1, 1, 1) # bs, 1, 1, 1, 1
                
                coarse_voxel = coarse_voxel.reshape(coarse_voxel.shape[0], 12, 12, 12)  
                coarse_voxel = coarse_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)      
             
            mask = (coarse_voxel > (-0.075 + avg)).float() # bs, 1, 12, 12, 12     
            mask_dense = F.interpolate(mask, scale_factor=6, mode='nearest') 
            loss, coarse_embedder_global, coarse_embedder = model.module.training_loss(x, coarse_voxel)
            loss = (loss * mask_dense).mean() # bs, 1, 48, 48, 48    
            
            loss.backward()  
            # Forward 
            if num_iter == config.step_per_update:
                num_iter = 0
                # netpNorm, netgradNorm = getGradNorm(model)
                # if config.grad_clip is not None:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                if config.use_ema:
                    update_ema(ema, model)
                optimizer.zero_grad()  
                    
            if args.distributed:
                mseloss = dist_utils.reduce_tensor(loss, args)   
                losses.update([mseloss.item()])
            else:
                mseloss = loss
                losses.update([mseloss.item()]) 

            if args.distributed:
                torch.cuda.synchronize()  
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.6f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger) 
        
        epoch_end_time = time.time()  
     
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.6f' % l for l in losses.avg()]), logger = logger) 

        if (epoch + 1) % config.saveIter == 0: 
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                if config.use_ema:
                    save_dict.update({'ema': ema.state_dict()}) 

                torch.save(save_dict, '%s/epoch_%d.pth' % (args.experiment_path, epoch)) 
                
        if (epoch + 1) % args.val_freq == 0:
            model.eval()
            with torch.no_grad():
                for idx, (taxonomy_ids, model_ids, data, sparse_voxel, dense_voxel, y) in enumerate(test_dataloader): 
                    npoints = config.dataset.train._base_.N_POINTS
                    dataset_name = config.dataset.train._base_.NAME 
                    if dataset_name == 'ShapeNet':
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
                        
                    x = dense_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3).cuda() 
                    noises_batch = torch.randn(x.shape).cuda() 
                    y = y.cuda()  
                    
                    # coarse_voxel = sparse_voxel.float() 
                    coarse_voxel = occnet_eval(partial)     
                    coarse_voxel = torch.softmax(coarse_voxel, dim=-1)# bs, v, v, v, 9 
                    coarse_voxel1 = torch.sum(torch.arange(9).cuda().reshape(1, 1, 1, 1, 9) * coarse_voxel, dim=-1) # bs, v, v, v   
                    coarse_voxel2 = torch.argmax(coarse_voxel, dim=-1) # bs, v, v, v
                    coarse_voxel = (coarse_voxel1 + coarse_voxel2) / 2.0 
                        
                    #normalize to [-1,1]
                    coarse_voxel = (coarse_voxel - 4.0) / 4.0
                    
                    avg = torch.mean(coarse_voxel.reshape(coarse_voxel.shape[0], -1), dim=-1, keepdim=True) # bs, 1
                    avg = avg.reshape(coarse_voxel.shape[0], 1, 1, 1, 1) # bs, 1, 1, 1, 1
                    
                    coarse_voxel = coarse_voxel.reshape(coarse_voxel.shape[0], 12, 12, 12)  
                    coarse_voxel = coarse_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)   
                    
                    mask = (coarse_voxel > -0.075 + avg).float() # bs, 1, 12, 12, 12 
                    mask_dense = F.interpolate(mask, scale_factor=6, mode='nearest') 
                    loss = (model.module.training_loss(x, coarse_voxel)[0] * mask_dense).mean() # bs, 1, 48, 48, 48

                    if args.distributed: 
                        mseloss = dist_utils.reduce_tensor(loss, args)  
                        testlosses.update([mseloss.item()])
                    else:
                        mseloss = loss  
                        testlosses.update([mseloss.item()]) 
                    if idx % 20 == 0:
                        print_log('[Epoch %d/%d][Batch %d/%d] TestLosses = %s' % 
                            (epoch, config.max_epoch, idx + 1, n_batches, ['%.4f' % l for l in testlosses.val()]), logger = logger) 
                    if (idx + 1) % args.val_interval == 0:  
                        
                        # coarse_voxel = sparse_voxel.float()  
                        coarse_voxel = occnet_eval(partial)     
                        coarse_voxel = torch.softmax(coarse_voxel, dim=-1)# bs, v, v, v, 9 
                        coarse_voxel1 = torch.sum(torch.arange(9).cuda().reshape(1, 1, 1, 1, 9) * coarse_voxel, dim=-1) # bs, v, v, v   
                        coarse_voxel2 = torch.argmax(coarse_voxel, dim=-1) # bs, v, v, v
                        coarse_voxel = (coarse_voxel1 + coarse_voxel2) / 2.0 
                         
                        #normalize to [-1,1]
                        coarse_voxel = (coarse_voxel - 4.0) / 4.0
                        
                        avg = torch.mean(coarse_voxel.reshape(coarse_voxel.shape[0], -1), dim=-1, keepdim=True) # bs, 1
                        avg = avg.reshape(coarse_voxel.shape[0], 1, 1, 1, 1) # bs, 1, 1, 1, 1
                        
                        coarse_voxel = coarse_voxel.reshape(coarse_voxel.shape[0], 12, 12, 12)  
                        coarse_voxel = coarse_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)   
                         
                        mask = (coarse_voxel > -0.075 + avg).float() # bs, 1, 12, 12, 12 
                        mask_dense = F.interpolate(mask, scale_factor=6, mode='nearest')  
                        
                        dense_voxel = model.module.sample_with_voxel(coarse_voxel)
                        dense_voxel[mask_dense == 0] = -1000 
                         
                        gt_pc = gt.squeeze().detach().cpu().numpy()
                        input_pc = partial.squeeze().detach().cpu().numpy()  
                        sparse_voxel = sparse_voxel.squeeze().cpu().numpy() 
                        coarse_voxel = coarse_voxel.squeeze().cpu().numpy()  
                        dense_voxel = dense_voxel.squeeze().cpu().numpy() 
                        
                        flattened = dense_voxel.flatten() 
                        # 找到前K大值的索引
                        indices = np.argpartition(flattened, -4096)[-4096:]     
                        # 将展平后的索引转换回原始数组的索引
                        indices = np.unravel_index(indices, dense_voxel.shape) 
                        dense_pc = np.stack(indices, axis=-1) 
                        
                        voxel_size = 1.0 / 72  
                        dense_pc = dense_pc * voxel_size + 0.5 * voxel_size   
                        
                        vis_path = os.path.join(args.experiment_path, 'vis_result')
                        pcd_path = os.path.join(args.experiment_path, 'pcd_result')
                        if not (os.path.exists(vis_path) or os.path.exists(pcd_path)) and args.local_rank == 0:
                            os.mkdir(vis_path)
                            # os.mkdir(pcd_path)
                        img_path = vis_path + f'/{args.local_rank}_{idx}.gif' 
                        # np.savetxt(pcd_path + f'/{args.local_rank}_{idx}_gt.xyz', gt_pc, fmt='%.6f') 
                        # np.savetxt(pcd_path + f'/{args.local_rank}_{idx}_dense.xyz', dense_pc, fmt='%.6f') 
                        misc.get_pointcloud_img(img_path, input_pc, gt_pc, coarse_voxel, dense_pc) 
                print_log('[Training] EPOCH: %d TestLosses = %s' % (epoch,  ['%.4f' % l for l in testlosses.avg()]), logger = logger)     
 
    dist.destroy_process_group()   

 
crop_ratio = {
    'easy': 1/4,
    'median' :1/2,  
    'hard':3/4 
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    occnet = builder.model_builder(config.occnet) 
    model = builder.model_builder(config.model)
    
    # load checkpoints 
    builder.load_model(occnet, config.occnet.eval_ckpt, logger = logger)  
    model.load_state_dict(torch.load(config.model.ckpt, map_location='cpu')['model_state'])
    if args.use_gpu:
        occnet = occnet.to(args.local_rank) 
        model = model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()  

    test(occnet, model, test_dataloader, args, config, logger = logger)

def test(occnet, model, test_dataloader, args, config, logger = None):
    occnet = occnet.eval()  # set model to eval mode
    model.eval()  # set model to eval mode 
    test_metrics = AverageMeter(Metrics.names()) 
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1 
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, sparse_voxel, dense_voxel, y) in enumerate(test_dataloader): 
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item() 
            model_id = model_ids[0] 
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()  
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda() 
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                x = dense_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3).cuda()  
                noises_batch = torch.randn(x.shape).cuda() 
                y = y.cuda() 
                sparse_voxel = sparse_voxel.cuda()
                for item_idx, item in enumerate(choice):           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048) 
                    #normalize to [0,1]
                    min_coords = torch.min(gt, dim=1, keepdim=True)[0] # B, 1, 3
                    max_coords = torch.max(gt, dim=1, keepdim=True)[0] # B, 1, 3
                    range_coords = max_coords - min_coords
                    gt = (gt - min_coords) / (range_coords + 1e-3)
                    partial = (partial - min_coords) / (range_coords + 1e-3)    
                  
                    
                    # coarse_voxel = sparse_voxel.float() 
                    coarse_voxel  = occnet(partial)     
                    coarse_voxel = torch.softmax(coarse_voxel, dim=-1)# bs, v, v, v, 9 
                    coarse_voxel1 = torch.sum(torch.arange(9).cuda().reshape(1, 1, 1, 1, 9) * coarse_voxel, dim=-1) # bs, v, v, v   
                    coarse_voxel2 = torch.argmax(coarse_voxel, dim=-1) # bs, v, v, v
                    coarse_voxel = (coarse_voxel1 + coarse_voxel2) / 2.0 
                     
                    #normalize to [-1,1]
                    coarse_voxel = (coarse_voxel - 4.0) / 4.0 
                    
                    avg = torch.mean(coarse_voxel.reshape(coarse_voxel.shape[0], -1), dim=-1, keepdim=True) # bs, 1
                    avg = avg.reshape(coarse_voxel.shape[0], 1, 1, 1, 1) # bs, 1, 1, 1, 1
                    
                    coarse_voxel = coarse_voxel.reshape(coarse_voxel.shape[0], 12, 12, 12) 
                    coarse_voxel = coarse_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)   
                      
                    mask = (coarse_voxel > -0.075 + avg).float() # bs, 1, 12, 12, 12 
                    mask_dense = F.interpolate(mask, scale_factor=6, mode='nearest') 
                    loss = (model.training_loss(x, coarse_voxel)[0] * mask_dense).mean() # bs, 1, 48, 48, 48 
                    # loss = (model.module.training_loss(x, coarse_voxel)).mean() # bs, 1, 48, 48, 48 
                    dense_voxel = model.sample_with_voxel(coarse_voxel)  
                    dense_voxel[mask_dense == 0] = -1000     
                    
                    dense_voxel = dense_voxel.squeeze().cpu().numpy() 
                    
                    flattened = dense_voxel.flatten()  
                    # 找到前K大值的索引 
                    indices = np.argpartition(flattened, -4096)[-4096:]       
                    # 将展平后的索引转换回原始数组的索引
                    indices = np.unravel_index(indices, dense_voxel.shape)  
                    dense_pc = np.stack(indices, axis=-1)  
                    
                      
                    voxel_size = 1.0 / 72    
                    dense_pc = dense_pc * voxel_size + 0.5 * voxel_size 
                    
                    coarse_voxel = coarse_voxel.squeeze().cpu().numpy()  
                    
                    vis_path = os.path.join(args.experiment_path, 'vis_result')
                    pcd_path = os.path.join(args.experiment_path, 'pcd_result')
                    if not os.path.exists(vis_path) and args.local_rank == 0:
                        os.mkdir(vis_path)
                        os.mkdir(pcd_path)
                    img_path = vis_path + f'/{args.local_rank}_{idx}_{item_idx}.gif'
                    np.savetxt(pcd_path + f'/{args.local_rank}_{idx}_{item_idx}_gt.xyz', gt.squeeze().detach().cpu().numpy(), fmt='%.6f')
                    np.savetxt(pcd_path + f'/{args.local_rank}_{idx}_{item_idx}dense.xyz', dense_pc, fmt='%.6f')     
                    # misc.get_pointcloud_img(img_path, partial.squeeze().detach().cpu().numpy(), gt.squeeze().detach().cpu().numpy(), coarse_voxel, dense_pc)
                           
                    dense_pc = torch.from_numpy(dense_pc).float().cuda() 
                    dense_pc = dense_pc.unsqueeze(0) 
                    dense_pc = torch.cat([dense_pc, partial], dim=1) 
                    _metrics = Metrics.get(dense_pc ,gt)  
                    
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)  
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s loss %s Metrics = %s, Category_Metrics = %s' %
                            (idx, n_samples, taxonomy_id, model_id, loss.item(), ['%.4f' % m for m in _metrics], ['%.4f' % m for m in  category_metrics[taxonomy_id].avg()]), logger=logger)  
                    
            elif dataset_name == 'KITTI':
                partial = data.cuda() 
                # ret = base_model(partial)
                # dense_points = ret[1]
                # target_path = os.path.join(args.experiment_path, 'vis_result')
                # if not os.path.exists(target_path):
                #     os.mkdir(target_path)
                # misc.visualize_KITTI(
                #     os.path.join(target_path, f'{model_id}_{idx:03d}'),
                #     [partial[0].cpu(), dense_points[0].cpu()]
                # )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # Visualize 
                # Save output results
             
                
        # Compute testing results
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger) 
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return  