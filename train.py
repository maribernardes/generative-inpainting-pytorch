import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger
import traceback

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info("Configuration: {}".format(config))


    try:
        # Load the dataset
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])

        # # Load validation dataset (if provided)
        # val_dataset = Dataset(data_path=config['val_data_path'],
        #                           with_subfolder=config['data_with_subfolder'],
        #                           image_shape=config['image_shape'],
        #                           random_crop=False)  # No random crop for validation

        # Define trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))

        if cuda:
            trainer = nn.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer
        """ 
        # Ensure netD is in training mode and gradients are enabled
        trainer_module.localD.train()
        trainer_module.globalD.train()
        for param in trainer_module.localD.parameters():
            param.requires_grad = True
        for param in trainer_module.globalD.parameters():
            param.requires_grad = True

        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

        # Resume training if applicable
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1


        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

        """        
        # Resume training if applicable
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)
        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                ground_truth = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth = next(iterable_train_loader)

            # Prepare the inputs
            device = ground_truth.device
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)
            x, mask, ground_truth = x.to(device), mask.to(device), ground_truth.to(device)

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)

            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update Discriminator
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            trainer_module.optimizer_d.step()

            # Update Generator
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                # Explicitly ensure losses require gradients
                l1_loss = losses.get('l1', torch.tensor(0.0, device=x.device, requires_grad=True))
                ae_loss = losses.get('ae', torch.tensor(0.0, device=x.device, requires_grad=True))
                wgan_g = losses.get('wgan_g', torch.tensor(0.0, device=x.device, requires_grad=True))

                # Combine losses (Ensure the computation is within the graph)
                loss_g = (
                    l1_loss * config['l1_loss_alpha'] +
                    ae_loss * config['ae_loss_alpha'] +
                    wgan_g * config['gan_loss_alpha']
                )

                # Ensure loss_g is tracking gradients
                if not loss_g.requires_grad:
                    raise RuntimeError("Loss_g does not require gradients! Check if losses are detached.")

                # Perform backward
                loss_g.backward()
                trainer_module.optimizer_g.step()


            # Logging and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = f'speed: {speed:.2f} batches/s '
                time_count = time.time()

                message = f'Iter: [{iteration}/{config["niter"]}] {speed_msg}'
                for k in log_losses:
                    if k in losses:  # Ensure key exists to prevent KeyError
                        v = losses[k]
                        writer.add_scalar(k, v.item(), iteration)
                        message += f'{k}: {v:.6f} '
                logger.info(message)

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    device = x.device  # Get the device of x
                    viz_images = torch.stack([
                        x.to(device),
                        inpainted_result.to(device),
                        offset_flow.to(device)
                    ], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)
            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)

    except Exception as e:
        logger.error(f"Training Error: {e}")
        raise e


if __name__ == '__main__':
    main()