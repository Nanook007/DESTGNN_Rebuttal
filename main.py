#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pickle 
import pandas as pd
import random

import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from utils.train import *
from utils.load_data import *
from utils.log import get_logger
from models_tcn.losses import *
from models_tcn.trainer import trainer
import yaml

# DESTGNN
from models_tcn.model import DESTGNN

import torch
import torch.nn as nn
torch.set_num_threads(4)

import warnings
warnings.filterwarnings('ignore')

import time
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


def get_log_dir(kwargs):

    dataset = config['data_args']['dataset_name']
    batch_size = config['model_args']['batch_size']
    topk = config['model_args']['topk']

    gcn_depth = config['model_args']['gcn_depth']
    gcn_layers =  config['model_args']['gcn_layers']

    epochs = config['optim_args']['epochs']
    lr_decay_ways = config['optim_args']['lr_decay_ways']
    lr_schedule = config['optim_args']['lr_schedule']
    learning_rate = config['optim_args']['lrate']
    wdecay = config['optim_args']['wdecay']
    dropout =  config['model_args']['dropout']
    
    run_id = '%s_ep%s_bs%s_layers%s_lr%s_topk%s_%s/' % (
        dataset,
        str(epochs),
        str(batch_size),
        str(gcn_layers)+'_'+str(gcn_depth),
        str(learning_rate) +'_'+ str(wdecay) +'_'+ str(dropout) + str(lr_schedule) +'_'+ str(lr_decay_ways),
        str(topk),
        time.strftime('%m%d%H%M%S'))
    
    base_dir = './check_para/1104_rebuttal'
    
    log_dir = os.path.join(base_dir, run_id)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir


def main(runid):
    # set_config(seed=3407)
    # seed = 1234
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model_name      = config['start_up']['model_name']
    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['dataset_name']
    save_path       = 'output/' + model_name + "_" + dataset_name + ".pt"             # the best model
    save_path_resume= 'output/' + model_name + "_" + dataset_name + "_resume.pt"      # the resume model
    load_pkl        = config['start_up']['load_pkl']
    
    
    if load_pkl:
        dataloader  = pickle.load(open('output/dataloader/' + dataset_name + '.pkl', 'rb'))
    else:
        batch_size  = config['model_args']['batch_size']
        dataloader  = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)

    for k, v in dataloader.items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    scaler = dataloader['scaler']
    
    # traffic flow
    if dataset_name in ['PEMS03','PEMS04','PEMS07','PEMS08']:
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None

   
# ================================ Hyper Parameters ================================= #
    # model parameters
    model_args                  = config['model_args']
    model_args['device']        = device
    model_args['dataset']       = dataset_name

    # training strategy parametes
    optim_args                  = config['optim_args']
    optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])

# ============================= Model and Trainer ============================= #
   
    # init the model
    model = DESTGNN(**model_args).to(device)
    model = nn.DataParallel(model)

    # get a trainer
    engine  = trainer(scaler, model, **optim_args)

    # begin training:
    train_time  = []
    logger.info("Whole trainining iteration is " + str(len(dataloader['train_loader'])))

    # training init: resume model & load parameters
    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']

    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    else:
        if mode == 'resume':
            resume_epoch = config['start_up']['resume_epoch']
            model = load_model(model, save_path_resume)
        else:
            resume_epoch = 0
    
    batch_num = resume_epoch * len(dataloader['train_loader'])     
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

# =============================================================== Training ================================================================= #
    if mode != 'test':
        logger.info("start training...")
        his_loss, train_time  = [], []
        minl = 1e5
        epoch_best = -1
        tolerance = config['optim_args']['tolerance']
        count_lfx = 0
        # record train val test loss to observe while training
        df_train_loss, df_val_loss, df_epoch_loss = [],[],[]

        for epoch in range(resume_epoch , optim_args['epochs']):
            time_train_start = time.time()
            current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
            train_loss, train_mape, train_rmse = [],[],[]
            dataloader['train_loader'].shuffle()    

            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            time_train_end = time.time()
            train_time.append(time_train_end - time_train_start)

            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            his_loss.append(mvalid_loss)

            df_train_loss.append(mtrain_loss)
            df_val_loss.append(mvalid_loss)
            df_epoch_loss.append(epoch)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, LR: {}, Training Time: {:.4f}/epoch'
            logger.info(log.format(epoch, mtrain_loss, mvalid_loss, current_learning_rate, (time_train_end - time_train_start)))

            if mvalid_loss < minl:
                logger.info('val_mae decreased from {:.2f} to {:.2f}'.format(minl, mvalid_loss))
                torch.save(engine.model.state_dict(), os.path.join(log_dir,"exp" + "_" + str(runid) + ".pth"))
                minl = mvalid_loss
                epoch_best = epoch
                count_lfx = 0
            else:
                count_lfx += 1
                if count_lfx > tolerance:
                    break

            xticks = [ i for i in range(0,len(df_epoch_loss),10)]
            loss_df = pd.DataFrame({'train_loss':df_train_loss,'val':df_val_loss},index=df_epoch_loss)
            loss_df.index.name='Epoch'
            ax = loss_df.plot(xticks=xticks)
            fig = ax.get_figure()
            loss_path = os.path.join(log_dir, str(runid)+'loss.png')
            fig.savefig(loss_path)

        logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))

        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(os.path.join(log_dir,"exp" + "_" + str(runid) + ".pth"), map_location='cpu'))

        logger.info("Training finished")
        logger.info("The valid loss on best model is {}, epoch:{}".format(str(round(his_loss[bestid], 4)), epoch_best))

# =============================================================== Test ================================================================= #

        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)
        realy   = realy.transpose(1, 2)
        y_list  = []

        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx   = data_reshaper(x, device)
            testy   = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds   = model(testx)
            outputs.append(preds)
            y_list.append(testy)

        yhat    = torch.cat(outputs,dim=0)[:realy.size(0),...]
        y_list  = torch.cat(y_list, dim=0)[:realy.size(0),...]
        assert torch.where(y_list == realy)

        if _max is not None:  # traffic flow
            realy   = scaler(realy.squeeze(-1), _max[0, 0, 0, 0], _min[0, 0, 0, 0])
            yhat    = scaler(yhat.squeeze(-1), _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        else:
            realy   = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat    = scaler.inverse_transform(yhat)

        amae, amape, armse = [],[],[]
        
        for i in range(12):
            pred    = yhat[:,:,i]
            real    = realy[:,:,i]
            metrics = metric(pred,real)
            log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            logger.info(log.format(i+1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0])     
            amape.append(metrics[1])    
            armse.append(metrics[2])   

        log = '(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        logger.info(log.format(np.mean(amae),np.mean(armse),np.mean(amape) * 100))

        return amae, amape, armse


if __name__ == '__main__':

    print(torch.version.cuda, torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(str(device)))
    
    if str(device) != 'cuda':
        print("CUDA is not available. Exiting the program.")
        sys.exit("CUDA is not available. Exiting the program.")

    
    config_path = "configs/PEMS04.yaml"
    # config_path = "configs/PEMS08.yaml"
    # config_path = "configs/PEMS07.yaml"
    
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    log_dir = get_log_dir(config)
    logger  = get_logger(log_dir, __name__, 'info.log', level='INFO')
    logger.info(config)
    logger.info('device: {}'.format(str(device)))
    logger.info('current_device: {}'.format(torch.cuda.current_device()))
    logger.info('parallel_devices: {}'.format(torch.cuda.device_count()))

    mae, mape, rmse = [], [], []
    
    # count: Record the number of successful experiments 
    runs = 10
    success = 2
    count = 0
    for runid in range(1,runs+1):
        amae, amape, armse = main(runid)

        # if failed , run again
        if amae[0] <= 1.0e-1:
            continue
        else:
            count += 1

        mae.append(amae)
        mape.append(amape)
        rmse.append(armse)

        if count == success:
            logger.info('Successfully converged: {} times, after: {} times'.format(success,runid))
            break

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    logger.info('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean')
    for i in range(12):
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}'
        logger.info(log.format(i, amae[i], armse[i], amape[i]))
    
    logger.info('test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))
    logger.info('Done')