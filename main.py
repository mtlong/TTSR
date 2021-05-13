from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
import pickle
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.test_FixedRef):
        t.load(model_path=args.model_path)
        lr_folder = args.lr_path
        psnr_avg = 0.0
        ssim_avg = 0.0
        count = 0
        for img_idx in range(126):
            img_name_prefix = str(img_idx).zfill(3)
            lr_img_path = os.path.join(lr_folder, img_name_prefix + "_0.png")
            ref_img_path = os.path.join(lr_folder, img_name_prefix + "_1.png")
            save_path = os.path.join(args.save_dir, img_name_prefix + "_0_sr.png")
            psnr, ssim = t.test_FixedRef(lr_img_path, ref_img_path, save_path)
            psnr_avg += psnr
            ssim_avg += ssim
            count += 1
        psnr_avg = psnr_avg / count
        ssim_avg = ssim_avg / count
        psnr_ssim = {}
        psnr_ssim['PSNR'] = psnr_avg
        psnr_ssim['SSIM'] = ssim_avg
        print(psnr_ssim)
        pickle.dump(psnr_ssim, open(os.path.join(args.save_dir, "PSNR_SSIM.pickle"), 'wb'))
    elif (args.test_AllRef):
        t.load(model_path=args.model_path)
        lr_folder = args.lr_path
        dict_folder = args.ref_path
        dict_file_list = os.listdir(dict_folder)
        input_file_list = []
        psnr_avg = 0.0
        ssim_avg = 0.0
        count = 0
        for img_idx in range(126):
            print("Processing Image #" + str(img_idx))
            img_name_prefix = str(img_idx).zfill(3)
            lr_img_path = os.path.join(lr_folder, img_name_prefix + "_0.png")
            input_file_list.append(img_name_prefix + "_0.png")
            psnr_max = -1
            ssim_max = -1
            psnr_list = []
            ssim_list = []
            for dict_filename in dict_file_list:
                ref_img_path = os.path.join(dict_folder, dict_filename)
                psnr, ssim = t.test_woSave(lr_img_path, ref_img_path)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                if psnr_max < psnr:
                    psnr_max = psnr
                if ssim_max < ssim:
                    ssim_max = ssim
            psnr_ssim_img = {}
            psnr_ssim_img["PSNR"] = psnr_list
            psnr_ssim_img["SSIM"] = ssim_list
            pickle.dump(psnr_ssim_img, open(os.path.join(args.save_dir, "PSNR_SSIM_List_" + img_name_prefix + "_0.pickle"), "wb"))
            psnr_avg += psnr_max
            ssim_avg += ssim_max
            count += 1
        psnr_ssim = {}
        psnr_ssim['PSNR'] = psnr_avg / count
        psnr_ssim['SSIM'] = ssim_avg / count
        print(psnr_ssim)
        pickle.dump(psnr_ssim, open(os.path.join(args.save_dir, "PSNR_SSIM_Oracle.pickle"), 'wb'))
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
