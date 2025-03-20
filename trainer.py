import os
import torch
import torch.nn as nn
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis

from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        # Initialize Networks
        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        # Optimizers
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()

        # # Ensure discriminators are in training mode
        # self.localD.train()
        # self.globalD.train()
        # for param in self.localD.parameters():
        #     param.requires_grad = True
        # for param in self.globalD.parameters():
        #     param.requires_grad = True

        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2, offset_flow = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

        """
        # Debugging: Ensure real_data requires gradients
        local_patch_gt.requires_grad_(True)
        ground_truth.requires_grad_(True) 
        """

        # Discriminator forward pass
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach()
        )
        global_real_pred, global_fake_pred = self.dis_forward(
            self.globalD, ground_truth, x2_inpaint.detach()
        )

        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
            torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']

        # Gradient penalty loss
        local_penalty = self.calc_gradient_penalty(self.localD, local_patch_gt, local_patch_x2_inpaint)
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        return losses, x2_inpaint, offset_flow


    def dis_forward(self, netD, real_data, fake_data):
        assert real_data.size() == fake_data.size()
        batch_size = real_data.size(0)
        batch_data = torch.cat([real_data, fake_data], dim=0)
        batch_output = netD(batch_data)  # Removed .detach()
        
        if batch_output.shape[0] != batch_data.shape[0]:
            raise ValueError(f"Discriminator output shape mismatch! Expected {batch_data.shape[0]}, got {batch_output.shape[0]}")

        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)

        # Ensure real_data and fake_data require gradients
        real_data.requires_grad_(True)
        fake_data.requires_grad_(True)

        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolates = (alpha * real_data + (1 - alpha) * fake_data)
        interpolates.requires_grad_(True)  # Ensure this is in the computation graph

        # Ensure netD is in training mode
        netD.train()

        disc_interpolates = netD(interpolates)

        # If disc_interpolates is detached, warn and retry
        if not disc_interpolates.requires_grad:
            print("WARNING: disc_interpolates.requires_grad is False, forcing computation...")
            disc_interpolates = disc_interpolates.clone().detach().requires_grad_(True)
            disc_interpolates = netD(interpolates)

        grad_outputs = torch.ones_like(disc_interpolates, device=real_data.device)

        # Compute gradients
        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=grad_outputs, create_graph=True,
            retain_graph=True, only_inputs=True, allow_unused=False
        )[0]

        if gradients is None:
            print("ERROR: autograd.grad returned None! Ensure interpolates is in the computation graph.")
            raise RuntimeError("Gradient computation failed. Check whether interpolates is detached.")

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print(f"Resume from {checkpoint_dir} at iteration {iteration}")
        logger.info(f"Resume from {checkpoint_dir} at iteration {iteration}")

        return iteration