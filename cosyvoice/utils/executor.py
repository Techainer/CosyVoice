import logging
from contextlib import nullcontext
import os

import torch
from tqdm import tqdm

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False):
        self.gan = gan
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device(f'cuda:{self.rank}')

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join):
        total_batches = len(train_data_loader)
        total_epochs = info_dict.get("max_epoch", 1)
        logging.info(f'Using accumulate grad, new batch size is {info_dict["accum_grad"]} times larger than before')

        model.train()
        pbar = tqdm(enumerate(train_data_loader), total=total_batches, dynamic_ncols=True)

        for batch_idx, batch_dict in pbar:
            info_dict.update({
                "tag": "TRAIN", "step": self.step,
                "epoch": self.epoch, "batch_idx": batch_idx
            })
            try:
                info_dict = batch_forward(model, batch_dict, scaler, info_dict)
            except Exception as e:
                # logging.error(e)
                continue
            info_dict = batch_backward(model, scaler, info_dict)
            info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
            log_per_step(writer, info_dict)

            pbar.set_description(f"[TRAIN] epoch={self.epoch}/{total_epochs}| step={self.step}: loss={info_dict['loss_dict'].get('loss', 0):.4f}| acc={info_dict['loss_dict'].get('acc', 0):.2f}| lr={info_dict.get('lr', 0):.8f}| grad_norm={info_dict.get('grad_norm', 0):.4f}")

            if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
               (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                model.train()

            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1

        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        total_batches = len(train_data_loader)
        total_epochs = info_dict.get("total_epoch", 1)
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {self.epoch} TRAIN info lr {lr} rank {self.rank}')
        logging.info(f'Using accumulate grad, new batch size is {info_dict["accum_grad"]} times larger than before')

        model.train()
        pbar = tqdm(enumerate(train_data_loader), total=total_batches, dynamic_ncols=True)

        for batch_idx, batch_dict in pbar:
            info_dict.update({
                "tag": "TRAIN", "step": self.step,
                "epoch": self.epoch, "batch_idx": batch_idx
            })

            batch_dict['turn'] = 'discriminator'
            info_dict = batch_forward(model, batch_dict, scaler, info_dict)
            info_dict = batch_backward(model, scaler, info_dict)
            info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
            optimizer.zero_grad()

            batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, scaler, info_dict)
            info_dict = batch_backward(model, scaler, info_dict)
            info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
            optimizer_d.zero_grad()

            log_per_step(writer, info_dict)
            loss_dict = info_dict['loss_dict']
            pbar.set_description(f"[TRAIN] epoch={self.epoch}/{total_epochs}| step={self.step}| batch={batch_idx+1}/{total_batches}| loss={loss_dict.get('loss'):.4f}| loss_gen={loss_dict.get('loss_gen'):.4f}| loss_fm={loss_dict.get('loss_fm'):.4f}| loss_mel={loss_dict.get('loss_mel'):.4f}| loss_tpr={loss_dict.get('loss_tpr'):.4f}| loss_f0={loss_dict.get('loss_f0'):.4f}|  lr={loss_dict.get('lr')}| grad_norm={loss_dict.get('grad_norm')}")

            if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                model.train()

            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1


        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        total_batches = len(cv_data_loader)
        total_epochs = info_dict.get("total_epoch", 1)
        logging.info(f'Epoch {self.epoch} Step {self.step + 1} on_batch_end {on_batch_end} CV rank {self.rank}')
        model.eval()

        total_num_utts, total_loss_dict = 0, {}
        pbar = tqdm(enumerate(cv_data_loader), total=total_batches, dynamic_ncols=True)

        for batch_idx, batch_dict in pbar:
            info_dict.update({
                "tag": "CV", "step": self.step,
                "epoch": self.epoch, "batch_idx": batch_idx
            })

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan:
                batch_dict['turn'] = 'generator'
            try:
                info_dict = batch_forward(model, batch_dict, None, info_dict)
            except Exception as e:
                continue

            for k, v in info_dict['loss_dict'].items():
                total_loss_dict.setdefault(k, []).append(v.item() * num_utts)

            pbar.set_description(f"[VALID] epoch={self.epoch}/{total_epochs}| step={self.step}: loss={info_dict['loss_dict'].get('loss', 0):.4f}| acc={info_dict['loss_dict'].get('acc', 0):.2f}")
            log_per_step(None, info_dict)

        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict

        log_per_save(writer, info_dict)
        model_name = f'epoch_{self.epoch}_whole' if on_batch_end else f'epoch_{self.epoch}_step_{self.step + 1}'
        save_model(model, model_name, info_dict)
