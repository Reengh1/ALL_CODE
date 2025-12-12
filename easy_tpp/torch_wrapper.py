""" Initialize a Pytorch model wrapper that feed into Model Runner   """

import torch
from torch.utils.tensorboard import SummaryWriter
from easy_tpp.utils import RunnerPhase, set_optimizer, set_device
from easy_tpp.model.torch_model.utlis import time_loss, type_loss
from easy_tpp.utils.hcl_utils import event2seq_embedding, event_contrastive_loss, seq_contrastive_loss,sampling_positive_seqs,sampling_negative_seqs_random, build_attn_mask_from_nonpad, visualize_event_significance

from easy_tpp.utils.dhcl_utils import make_negatives_by_swaps, make_positive_by_swaps
import torch.nn as nn
class TorchModelWrapper:
    def __init__(self, model, base_config, model_config, trainer_config):
        """A wrapper class for Torch backends.
        Args:
            model (BaseModel): a TPP model.
            base_config (EasyTPP.Config): basic configs.
            model_config (EasyTPP.ModelConfig): model spec configs.
            trainer_config (EasyTPP.TrainerConfig): trainer spec configs.
        """
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.model_id = self.base_config.model_id
        self.device = set_device(self.trainer_config.gpu)

        self.model.to(self.device)

        if self.model_config.is_training:
            # set up optimizer
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.opt = set_optimizer(optimizer, self.model.parameters(), self.learning_rate)

        # set up tensorboard
        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.trainer_config.use_tfb:
            self.train_summary_writer = SummaryWriter(log_dir=self.base_config.specs['tfb_train_dir'])
            self.valid_summary_writer = SummaryWriter(log_dir=self.base_config.specs['tfb_valid_dir'])

    def restore(self, ckpt_dir):
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """

        self.model.load_state_dict(torch.load(ckpt_dir), strict=False)

    def save(self, ckpt_dir):
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        torch.save(self.model.state_dict(), ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """Write the kv_paris into the tensorboard

        Args:
            epoch (int): epoch index in the training.
            kv_pairs (dict): metrics dict.
            phase (RunnerPhase): a const that defines the stage of model runner.
        """
        if self.trainer_config.use_tfb:
            summary_writer = None
            if phase == RunnerPhase.TRAIN:
                summary_writer = self.train_summary_writer
            elif phase == RunnerPhase.VALIDATE:
                summary_writer = self.valid_summary_writer
            elif phase == RunnerPhase.PREDICT:
                pass

            if summary_writer is not None:
                for k, v in kv_pairs.items():
                    if k != 'num_events':
                        summary_writer.add_scalar(k, v, epoch)

                summary_writer.flush()
        return

    def close_summary(self):
        """Close the tensorboard summary writer.
        """
        if self.train_summary_writer is not None:
            self.train_summary_writer.close()

        if self.valid_summary_writer is not None:
            self.valid_summary_writer.close()
        return

    def run_batch_mle(self, batch, phase):
        """Run one batch.

        Args:
            batch (EasyTPP.BatchEncoding): preprocessed batch data that go into the model.
            phase (RunnerPhase): a const that defines the stage of model runner.

        Returns:
            tuple: for training and validation we return loss, prediction and labels;
            for prediction we return prediction.
        """

        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model
            with torch.set_grad_enabled(grad_flag):
                _, _, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)

            # Assume we dont do prediction on train set
            #pred_dtime, pred_type, label_dtime, label_type, mask = None, None, None, None, None
            #pred_dtime, pred_type = self.model.predict_one_step_at_every_event(batch=batch)
            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            #print(batch_non_pad_mask)
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 不忽略任何标签
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                (loss+pred_loss+se/100).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event
        
    def run_batch_hcl_thp(self, batch, phase):
        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model

            with torch.set_grad_enabled(grad_flag):
                lamda, enc_out, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)
            #[B, L, C]->[B, L]
            _, enc_att_g = self.model.forward(batch[0], batch[2], batch[4], True)
            event_significance = torch.sum(enc_att_g, dim=1)

            #Original Sequence
            label_time, label_type, batch_non_pad_mask = batch[0], batch[2], batch[3]
            seq_emb = event2seq_embedding(enc_out, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            pos_event_type, pos_event_time, pos_pad_mask = sampling_positive_seqs(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                ratio_remove=0.2)

            neg_event_type, neg_event_time, neg_pad_mask = sampling_negative_seqs_random(label_type,
                                                                label_time,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                num_neg=20,
                                                                ratio_remove=0.2)
            pos_attn_mask = build_attn_mask_from_nonpad(pos_pad_mask)
            neg_attn_mask = build_attn_mask_from_nonpad(neg_pad_mask)
            #Above all, [B, L]
            enc_out_p, _ = self.model.forward(pos_event_time[:, :-1], pos_event_type[:, :-1], pos_attn_mask[:, :-1, :-1], True)
            enc_out_n, _ = self.model.forward(neg_event_time[:, :-1], neg_event_type[:, :-1], neg_attn_mask[:, :-1, :-1], True)
            seq_emb_p = event2seq_embedding(enc_out_p, pos_pad_mask[:,:-1].unsqueeze(-1))
            seq_emb_n = event2seq_embedding(enc_out_n, neg_pad_mask[:,:-1].unsqueeze(-1))
            #print(lamda.shape, label_type.shape, batch_non_pad_mask.shape)
            nce1 = event_contrastive_loss(lamda, label_type[:, 1:], batch_non_pad_mask[:, 1:].unsqueeze(-1))
            nce2 = seq_contrastive_loss(seq_emb, seq_emb_p.detach(), seq_emb_n.detach())

            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 不忽略任何标签
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                (loss+pred_loss+se/100+10*nce1+10*nce2).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event
    def run_batch_dhcl_thp(self, batch, phase):
        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model
            with torch.set_grad_enabled(grad_flag):
                lamda, enc_out, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)
            #[B, L, C]->[B, L]
            _, enc_att_g = self.model.forward(batch[0], batch[2], batch[4], True)
            event_significance = torch.sum(enc_att_g, dim=1)
            #Original Sequence
            label_time, label_type, batch_non_pad_mask = batch[0], batch[2], batch[3]
            seq_emb = event2seq_embedding(enc_out, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            pos_event_type, pos_event_time= make_positive_by_swaps(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                mask =batch_non_pad_mask,
                                                                frac_low=0.3,
                                                                swap_factor=0.05,
                                                                max_pairs=20000)

            neg_event_type, neg_event_time, neg_non_pad_mask = make_negatives_by_swaps(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                mask =batch_non_pad_mask,
                                                                num_neg=20,
                                                                frac_top=0.3,
                                                                max_pairs=20000)
            #print(label_time.shape, label_type.shape)
            #print(pos_event_time.shape, pos_event_type.shape)
            #print(neg_event_time.shape, neg_event_type.shape)
            attn_mask = build_attn_mask_from_nonpad(batch_non_pad_mask)
            neg_attn_mask = build_attn_mask_from_nonpad(neg_non_pad_mask)
            enc_out_p, _ = self.model.forward(pos_event_time[:, :-1], pos_event_type[:, :-1], attn_mask[:, :-1, :-1], True)
            enc_out_n, _ = self.model.forward(neg_event_time[:, :-1], neg_event_type[:, :-1], neg_attn_mask[:, :-1, :-1], True)
            seq_emb_p = event2seq_embedding(enc_out_p, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            seq_emb_n = event2seq_embedding(enc_out_n, neg_non_pad_mask[:,:-1].unsqueeze(-1))
            #print(lamda.shape, label_type.shape, batch_non_pad_mask.shape)
            nce1 = event_contrastive_loss(lamda, label_type[:, 1:], batch_non_pad_mask[:, 1:].unsqueeze(-1))
            nce2 = seq_contrastive_loss(seq_emb, seq_emb_p.detach(), seq_emb_n.detach())


            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 不忽略任何标签
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                (loss+pred_loss+se/100+10*nce1+10*nce2).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event

    def run_batch_hcl_ANHP(self, batch, phase):
        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model

            with torch.set_grad_enabled(grad_flag):
                lamda, enc_out, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)
            
            #[B, L, C]->[B, L]
            _, enc_att_g = self.model.forward(batch[0], batch[2], batch[4], None, True)
            event_significance = torch.sum(enc_att_g, dim=1)
            #visualize_event_significance(event_significance, 0, None, "/home/guangchen_li/dev/DCL_TPP/pics/0.png")
            #print(event_significance)
            #print(event_significance)
            #Original Sequence
            label_time, label_type, batch_non_pad_mask = batch[0], batch[2], batch[3]
            seq_emb = event2seq_embedding(enc_out, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            #print(label_type.shape, label_time.shape,event_significance.shape,  batch_non_pad_mask.shape)
            pos_event_type, pos_event_time, pos_pad_mask = sampling_positive_seqs(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                ratio_remove=0.4)

            neg_event_type, neg_event_time, neg_pad_mask = sampling_negative_seqs_random(label_type,
                                                                label_time,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                num_neg=20,
                                                                ratio_remove=0.4)
            pos_attn_mask = build_attn_mask_from_nonpad(pos_pad_mask)
            neg_attn_mask = build_attn_mask_from_nonpad(neg_pad_mask)
            #Above all, [B, L]
            enc_out_p, _ = self.model.forward(pos_event_time[:, :-1], pos_event_type[:, :-1], pos_attn_mask[:, :-1, :-1], None, True)
            enc_out_n, _ = self.model.forward(neg_event_time[:, :-1], neg_event_type[:, :-1], neg_attn_mask[:, :-1, :-1], None, True)
            
            
            seq_emb_p = event2seq_embedding(enc_out_p, pos_pad_mask[:,:-1].unsqueeze(-1))
            seq_emb_n = event2seq_embedding(enc_out_n, neg_pad_mask[:,:-1].unsqueeze(-1))
            #print(lamda.shape, label_type.shape, batch_non_pad_mask.shape)
            nce1 = event_contrastive_loss(lamda, label_type[:, 1:], batch_non_pad_mask[:, 1:].unsqueeze(-1))
            nce2 = seq_contrastive_loss(seq_emb, seq_emb_p.detach(), seq_emb_n.detach())

            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 不忽略任何标签
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                print(loss, pred_loss, nce1, nce2)
                (loss+pred_loss+se/100+20*nce1+10*nce2).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event

    def run_batch_hcl_sahp(self, batch, phase):
        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model

            with torch.set_grad_enabled(grad_flag):
                lamda, enc_out, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)
            #[B, L, C]->[B, L]
            
            _, enc_att_g = self.model.forward(batch[0], batch[1],batch[2], batch[4], True)

            event_significance = torch.sum(enc_att_g, dim=1)

            #Original Sequence
            label_time, label_type, batch_non_pad_mask = batch[0], batch[2], batch[3]
            seq_emb = event2seq_embedding(enc_out, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            pos_event_type, pos_event_time, pos_pad_mask, pos_delta_t = sampling_positive_seqs(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                ratio_remove=0.2,
                                                                require_delta=True)

            neg_event_type, neg_event_time, neg_pad_mask, neg_delta_t = sampling_negative_seqs_random(label_type,
                                                                label_time,
                                                                pad_mask=batch_non_pad_mask,
                                                                pad_id=lamda.shape[2],
                                                                num_neg=20,
                                                                ratio_remove=0.2,
                                                                require_delta=True)
            pos_attn_mask = build_attn_mask_from_nonpad(pos_pad_mask)
            neg_attn_mask = build_attn_mask_from_nonpad(neg_pad_mask)
            #Above all, [B, L]

            enc_out_p, _ = self.model.forward(pos_event_time[:, :-1], pos_delta_t[:, :-1], pos_event_type[:, :-1], pos_attn_mask[:, :-1, :-1], True)
            enc_out_n, _ = self.model.forward(neg_event_time[:, :-1], neg_delta_t[:, :-1], neg_event_type[:, :-1], neg_attn_mask[:, :-1, :-1], True)
            
            seq_emb_p = event2seq_embedding(enc_out_p, pos_pad_mask[:,:-1].unsqueeze(-1))
            seq_emb_n = event2seq_embedding(enc_out_n, neg_pad_mask[:,:-1].unsqueeze(-1))
            #print(lamda.shape, label_type.shape, batch_non_pad_mask.shape)
            nce1 = event_contrastive_loss(lamda, label_type[:, 1:], batch_non_pad_mask[:, 1:].unsqueeze(-1))
            nce2 = seq_contrastive_loss(seq_emb, seq_emb_p.detach(), seq_emb_n.detach())

            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                #print(loss, pred_loss, nce1, nce2)
                (loss+pred_loss+se/100+10*nce1+10*nce2).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event
        
    def run_batch_dhcl_sahp(self, batch, phase):
        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model

            with torch.set_grad_enabled(grad_flag):
                lamda, enc_out, _, loss, num_event, time_pred, type_pred = self.model.loglike_loss(batch)
            #[B, L, C]->[B, L]
            
            _, enc_att_g = self.model.forward(batch[0], batch[1],batch[2], batch[4], True)

            event_significance = torch.sum(enc_att_g, dim=1)

            #Original Sequence
            label_time, label_type, batch_non_pad_mask = batch[0], batch[2], batch[3]
            seq_emb = event2seq_embedding(enc_out, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            pos_event_type, pos_event_time, pos_delta_t= make_positive_by_swaps(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                mask =batch_non_pad_mask,
                                                                frac_low=0.3,
                                                                swap_factor=0.05,
                                                                max_pairs=4,
                                                                require_delta_t = True)

            neg_event_type, neg_event_time, neg_non_pad_mask, neg_delta_t = make_negatives_by_swaps(label_type,
                                                                label_time,
                                                                significance=event_significance,
                                                                mask =batch_non_pad_mask,
                                                                num_neg=20,
                                                                frac_top=0.3,
                                                                max_pairs=200,
                                                                require_delta_t = True)
            attn_mask = build_attn_mask_from_nonpad(batch_non_pad_mask)
            neg_attn_mask = build_attn_mask_from_nonpad(neg_non_pad_mask)

            #Above all, [B, L]

            enc_out_p, _ = self.model.forward(pos_event_time[:, :-1], pos_delta_t[:, :-1], pos_event_type[:, :-1], attn_mask[:, :-1, :-1], True)
            enc_out_n, _ = self.model.forward(neg_event_time[:, :-1], neg_delta_t[:, :-1], neg_event_type[:, :-1], neg_attn_mask[:, :-1, :-1], True)
            
            seq_emb_p = event2seq_embedding(enc_out_p, batch_non_pad_mask[:,:-1].unsqueeze(-1))
            seq_emb_n = event2seq_embedding(enc_out_n, neg_non_pad_mask[:,:-1].unsqueeze(-1))
            #print(lamda.shape, label_type.shape, batch_non_pad_mask.shape)
            nce1 = event_contrastive_loss(lamda, label_type[:, 1:], batch_non_pad_mask[:, 1:].unsqueeze(-1))
            nce2 = seq_contrastive_loss(seq_emb, seq_emb_p.detach(), seq_emb_n.detach())

            label_time, label_type, batch_non_pad_mask = batch[0][:, 1:], batch[2][:, 1:], batch[3][:, 1:]
            pad_candidates = label_type[~batch_non_pad_mask]
            if pad_candidates.numel() > 0:
                pad_id = pad_candidates.mode().values.item()  # 多数值作为 pad_id
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
                valid_mask = (label_time != float(pad_id))
                label_time = label_time.masked_fill(~valid_mask, 0.0)
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  
            #print(type_pred.shape, label_type.shape)
            pred_loss, pred_num_event = type_loss(type_pred, label_type, loss_fn)
            
            se = time_loss(time_pred, label_time)
            batch_num_pred = batch_non_pad_mask.sum().item()
            # update grad
            if is_training:
                self.opt.zero_grad()
                #print(loss, pred_loss, nce1, nce2)
                (loss+pred_loss+se/100+10*nce1+10*nce2).backward()
                self.opt.step()
            return loss, pred_loss, se, pred_num_event, batch_num_pred, num_event