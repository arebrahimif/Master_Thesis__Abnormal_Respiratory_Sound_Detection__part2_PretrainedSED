import numpy as np
# Added dependencies for the ICBHI dataset loader.
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import wandb
import transformers
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# Added: use Lightning checkpoint selection for the best validation HS score.
from pytorch_lightning.callbacks import ModelCheckpoint
import sed_scores_eval

from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.m2d.M2D_wrapper import M2DWrapper
from models.asit.ASIT_wrapper import ASiTWrapper
from models.prediction_wrapper import PredictionsWrapper

# Adapted imports: use weak-label augmentation functions and mixup cross-entropy.
from helpers.augment import (frame_shift_weak, time_mask_weak, mixup_weak, mixup_ce_loss, filter_augmentation, mixstyle, RandomResizeCrop)

from helpers.utils import worker_init_fn
from data_util.audioset_strong import get_training_dataset, get_eval_dataset
from data_util.audioset_strong import get_temporal_count_balanced_sample_weights, get_uniform_sample_weights, \
    get_weighted_sampler
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes
from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
from models.frame_mn.utils import NAME_TO_WIDTH

# Adaptation: enable higher-precision float32 matrix multiplication.
torch.set_float32_matmul_precision("high")

class PLModule(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
                # Adaptation: configure validation bookkeeping for a two-class task.
        num_classes = 2  # or 4
                # Added: collect predictions and labels for epoch-level binary metrics.
        self.val_preds = []
        self.val_targets = []
                # Added: keep confusion-matrix reporting disabled during routine validation.
        self.log_confusion_matrix = False

        if config.pretrained == "scratch":
            checkpoint = None
        elif config.pretrained == "ssl":
            checkpoint = "ssl"
        elif config.pretrained == "weak":
            checkpoint = "weak"
        elif config.pretrained == "strong":
            checkpoint = "strong_1"
        else:
            raise ValueError(f"Unknown pretrained checkpoint: {config.pretrained}")

        # load transformer model
        if config.model_name == "BEATs":
            beats = BEATsWrapper()
            model = PredictionsWrapper(beats, checkpoint=f"BEATs_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "ATST-F":
            atst = ATSTWrapper()
            model = PredictionsWrapper(
            # Adaptation: set strong and weak model heads to two output classes.
                                        atst,
                                        checkpoint=f"ATST-F_{checkpoint}" if checkpoint else None,
                                        seq_model_type=config.seq_model_type,
                                        n_classes_strong=2,
                                        n_classes_weak=2,
                                        )
        elif config.model_name == "fpasst":
            fpasst = FPaSSTWrapper()
            model = PredictionsWrapper(fpasst, checkpoint=f"fpasst_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "M2D":
            m2d = M2DWrapper()
            model = PredictionsWrapper(m2d, checkpoint=f"M2D_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       embed_dim=m2d.m2d.cfg.feature_d)
        elif config.model_name == "ASIT":
            asit = ASiTWrapper()
            model = PredictionsWrapper(asit, checkpoint=f"ASIT_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name.startswith("frame_mn"):
            width = NAME_TO_WIDTH(config.model_name)
            frame_mn = FrameMNWrapper(width)
            embed_dim = frame_mn.state_dict()['frame_mn.features.16.1.bias'].shape[0]
            model = PredictionsWrapper(frame_mn, checkpoint=f"{config.model_name}_strong_1", embed_dim=embed_dim)
        else:
            raise NotImplementedError(f"Model {config.model_name} not (yet) implemented")

        self.model = model


        # prepare ingredients for knowledge distillation
        assert 0 <= config.distillation_loss_weight <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
                # Adaptation: use per-example cross-entropy for integer class labels.
        self.classification_loss = nn.CrossEntropyLoss(reduction="none")

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_durations_df = pd.read_csv(f"resources/eval_durations.csv",
                                            sep=",", header=None, names=["filename", "duration"])
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []

    def forward(self, batch):
        x = batch["audio"]
        mel = self.model.mel_forward(x)
                # Adaptation: use clip-level class scores instead of strong framewise scores.
        _, y_weak = self.model(mel)
        return y_weak

    def get_optimizer(
            self, lr, adamw=False, weight_decay=0.01, betas=(0.9, 0.999)
    ):
        # we split the parameters into two groups, one for the pretrained model and one for the downstream model
        # we also split each of them into <=1 dimensional and >=2 dimensional parameters, so we can only
        # apply weight decay to the >=2 dimensional parameters, thus excluding biases and batch norms, an idea from NanoGPT
        params_leq1D = []
        params_geq2D = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndimension() >= 2:
                    params_geq2D.append(param)
                else:
                    params_leq1D.append(param)

        param_groups = [
            {'params': params_leq1D, 'lr': lr},
            {'params': params_geq2D, 'lr': lr, 'weight_decay': weight_decay},
        ]

        if weight_decay > 0:
            assert adamw
        assert len(param_groups) > 0
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return torch.optim.Adam(param_groups, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=20000,
            lr_end=2e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.config.max_lr, adamw=self.config.adamw,
                                       weight_decay=self.config.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer,
                                          num_training_steps,
                                          schedule_mode=self.config.schedule_mode,
                                                                                    # Adaptation: pass the configured warmup period to the scheduler.
                                          num_warmup_steps=self.config.warmup_steps,
                                          lr_end=self.config.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]


        # Adaptation: train from ICBHI integer labels with weak-label augmentation and mixup.
    def training_step(self, train_batch, batch_idx):
        x = train_batch["audio"]
        labels = train_batch["label"].long()
    
        mel = self.model.mel_forward(x)
    
                # Adaptation: use weak-label augmentations that preserve the clip class.
        # Augmentations that do not change labels
        if self.config.frame_shift_range > 0:
            mel = frame_shift_weak(mel, shift_range=self.config.frame_shift_range)
    
        if self.config.mixstyle_p > random.random():
            mel = mixstyle(mel)
    
        if self.config.max_time_mask_size > 0:
            mel = time_mask_weak(
                mel,
                min_mask_ratio=0.05,
                max_mask_ratio=self.config.max_time_mask_size
            )
    
        if self.config.filter_augment_p > random.random():
            mel, _ = filter_augmentation(mel)
    
        if self.config.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)
    
                # Adaptation: compute the ordinary batch classification loss.
        # Original batch loss
        _, y_hat_original = self.model(mel)
        loss_original = self.classification_loss(y_hat_original, labels).mean()
    
                # Adaptation: add mixup loss weighted by the configured mixup probability.
        # Extra mixup loss
        if self.config.mixup_p > 0:
            mixed_mel, labels_a, labels_b, lam = mixup_weak(mel, labels)
            _, y_hat_mixed = self.model(mixed_mel)
    
            loss_mixup = mixup_ce_loss(
                self.classification_loss,
                y_hat_mixed,
                labels_a,
                labels_b,
                lam
            )
    
            loss = loss_original + self.config.mixup_p * loss_mixup
        else:
            loss = loss_original
    
        self.log("epoch", self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"trainer/lr_optimizer_{i}", param_group["lr"])
    
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/loss_original", loss_original.detach().cpu())
    
        if self.config.mixup_p > 0:
            self.log("train/loss_mixup", loss_mixup.detach().cpu())
    
        return loss


        # Retained upstream strong-label validation path, renamed and unused.
    def validation_step_1(self, val_batch, batch_idx):
        # bring ground truth into shape needed for evaluation
        for f, gt_string in zip(val_batch["filename"], val_batch["gt_string"]):
            f = f[:-len(".mp3")]
            events = [e.split(";;") for e in gt_string.split("++")]
            self.val_ground_truth[f] = [(float(e[0]), float(e[1]), e[2]) for e in events]
            self.val_duration[f] = self.val_durations_df[self.val_durations_df["filename"] == f]["duration"].values[0]

        y_hat_strong = self(val_batch)
        y_strong = val_batch["strong"]

        loss = self.classification_loss(y_hat_strong, y_strong)
        self.val_loss.append(loss.cpu())

        scores_raw, scores_postprocessed, prediction_dfs = batched_decode_preds(
            y_hat_strong.float(),
            val_batch['filename'],
            self.encoder,
            median_filter=self.config.median_window
        )

        self.val_predictions_strong.update(
            scores_postprocessed
        )

        # Adaptation: validate clip labels and retain predictions for binary metrics.
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        y = batch["label"].long()
    
        mel = self.model.mel_forward(x)
        _, y_hat = self.model(mel)
    
        loss = self.classification_loss(y_hat, y).mean()
        preds = torch.argmax(y_hat, dim=1)
    
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())
        self.val_loss.append(loss.detach().cpu())

        # Adaptation: calculate accuracy, sensitivity, specificity, and HS score.
    # validation for 2-class HS
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
    
        acc = (preds == targets).float().mean()
    
        cm = torch.zeros((2, 2), dtype=torch.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
    
        TP = cm[1, 1].float()
        TN = cm[0, 0].float()
        FP = cm[0, 1].float()
        FN = cm[1, 0].float()
    
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        
        hs_score = (2 * sensitivity * specificity) / (sensitivity + specificity + 1e-8)
    
        mean_val_loss = torch.stack([
                                    v if isinstance(v, torch.Tensor) else torch.tensor(v)
                                    for v in self.val_loss
                                    ]).mean()
    
        self.log("val/loss", mean_val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/sensitivity", sensitivity, on_step=False, on_epoch=True)
        self.log("val/specificity", specificity, on_step=False, on_epoch=True)
        self.log("val/HS_score", hs_score, prog_bar=True, on_step=False, on_epoch=True)
        
        # --------------------------------------------------
                # Added: log the confusion matrix and summary metrics for the selected checkpoint.
        # Log confusion matrix only for the best checkpoint
        # --------------------------------------------------
        if self.log_confusion_matrix and self.logger is not None:
        
            confusion_matrix = wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets.numpy(),
                preds=preds.numpy(),
                class_names=["normal", "abnormal"]
            )
    
            self.logger.experiment.log({
                "best_model/confusion_matrix": confusion_matrix,
                "best_model/accuracy": acc.item(),
                "best_model/sensitivity": sensitivity.item(),
                "best_model/specificity": specificity.item(),
                "best_model/HS_score": hs_score.item(),
            })
    
            print("\n========== BEST MODEL RESULTS ==========")
            print(f"Accuracy:     {acc.item():.4f}")
            print(f"Sensitivity:  {sensitivity.item():.4f}")
            print(f"Specificity:  {specificity.item():.4f}")
            print(f"HS-score:     {hs_score.item():.4f}")
            print("\nConfusion Matrix:")
            print(cm.numpy())
            print("========================================\n")
    
                # Added: collect predictions and labels for epoch-level binary metrics.
        self.val_preds = []
        self.val_targets = []
        self.val_loss = []

        # Retained upstream event-based validation path, renamed and unused.
    def on_validation_epoch_end_1(self):
        gt_unique_events = set([e[2] for f, events in self.val_ground_truth.items() for e in events])
        train_unique_events = set(self.encoder.labels)
        # evaluate on all classes that are in both train and test sets (407 classes)
        class_intersection = gt_unique_events.intersection(train_unique_events)

        assert len(class_intersection) == len(set(as_strong_train_classes).intersection(as_strong_eval_classes)) == 407, \
            f"Intersection unique events. Expected: {len(set(as_strong_train_classes).intersection(as_strong_eval_classes))}," \
            f" Actual: {len(class_intersection)}"

        # filter ground truth according to class_intersection
        val_ground_truth = {fid: [event for event in self.val_ground_truth[fid] if event[2] in class_intersection]
                            for fid in self.val_ground_truth}
        # drop audios without events - aligned with DESED evaluation procedure
        val_ground_truth = {fid: events for fid, events in val_ground_truth.items() if len(events) > 0}
        # keep only corresponding audio durations
        audio_durations = {
            fid: self.val_duration[fid] for fid in val_ground_truth.keys()
        }

        # filter files in predictions
        as_strong_preds = {
            fid: self.val_predictions_strong[fid] for fid in val_ground_truth.keys()
        }
        # filter classes in predictions
        unused_classes = list(set(self.encoder.labels).difference(class_intersection))
        for f, df in as_strong_preds.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        psds1 = sed_scores_eval.intersection_based.psds(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )

        # "val/psds1_macro_averaged" is psds1 without penalization for performance
        #  variations across classes
        logs = {"val/loss": torch.as_tensor(self.val_loss).mean().cuda(),
                "val/psds1": psds1[0],
                "val/psds1_macro_averaged": np.array([v for k, v in psds1[1].items()]).mean(),
                "val/pauroc": segment_based_pauroc[0]['mean'],
                }

        self.log_dict(logs, sync_dist=False)
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []


# Added dependencies for the ICBHI dataset loader.
import pandas as pd
import os
import librosa

# Added ICBHI dataset: reads metadata, caches 16 kHz audio, and normalizes clip length to 10 seconds.
class ICBHIDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_dir, split="train", use_4class=True):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.use_4class = use_4class
        self.df = self.df[self.df["fold"] == split].reset_index(drop=True)
        self.audio_cache = {}
        max_len = 16000 * 10
        
        for i in range(len(self.df)):
            file = self.df.iloc[i]["filename"]
            path = os.path.join(self.audio_dir, file)
        
            audio, _ = librosa.load(path, sr=16000)
        
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            else:
                audio = audio[:max_len]
        
            self.audio_cache[file] = torch.tensor(audio, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file = row["filename"]
        audio = self.audio_cache[file]
    
                # Choose the four-class label or binary target column from the metadata.
        if self.use_4class:
            label = int(row["label"])
        else:
            label = int(row["target"])

    
            # Disabled strong frame-target generation; this dataset returns clip-level labels.
        '''T = audio.shape[0] // 320
        strong = torch.zeros((T, num_classes), dtype=torch.float32)
        strong[:, label] = 1.0 '''
    
        return {
            "audio": audio,
            #"strong": strong,
            "label": label,
            "filename": file,
            #"gt_string": "0;;5;;class"
        }


def train(config):
    # Train Models on temporally-strong portion of AudioSet.

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="PTSED",
        notes="Pre-Training Transformers for Sound Event Detection on AudioSet Strong.",
        tags=["AudioSet Strong", "Sound Event Detection", "Pseudo Labels", "Knowledge Disitillation"],
        config=config,
        name=config.experiment_name
    )

    # encoder manages encoding and decoding of model predictions
    # Adaptation: replace the AudioSet class vocabulary with normal/abnormal labels.
    # encoder = ManyHotEncoder(as_strong_train_classes)
    encoder = ManyHotEncoder(["normal", "abnormal"])

        # Adaptation: load the ICBHI metadata and audio splits in place of AudioSet.
    csv_path = "ICBHI_dataset//meta.csv"
    audio_dir = "ICBHI_dataset//audios_cycles"
    
    train_set = ICBHIDataset(csv_path, audio_dir, split="train", use_4class=False)
    eval_set = ICBHIDataset(csv_path, audio_dir, split="test", use_4class=False)
    
        # Adaptation: do not use the upstream AudioSet balanced sampler.
    train_sampler = None

    # train dataloader
    train_dl = DataLoader(
        dataset=train_set,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
                # Adaptation: shuffle training examples directly without a sampler.
        shuffle=True
    )

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, encoder)

    #select the model checkpoint with the highest validation HS-score.
        # Added: save the checkpoint with the highest validation HS score.
    checkpoint_callback = ModelCheckpoint(monitor="val/HS_score",
                                          mode="max",
                                          save_top_k=1,
                                          filename="best-{epoch:02d}-{val_HS_score:.4f}")
    
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch
                         )

    # start training and validation for the specified number of epochs
        # Adaptation: after training, select and re-evaluate the best validation checkpoint.
    trainer.fit(pl_module, train_dl, eval_dl)
    
    # --------------------------------------------------
    # Load the best model according to validation HS-score
    # --------------------------------------------------
    
        # Read the best checkpoint path and score selected by ModelCheckpoint.
    best_checkpoint = checkpoint_callback.best_model_path
    best_hs_score = checkpoint_callback.best_model_score
    
    print("\n========================================")
    print("Best checkpoint:")
    print(best_checkpoint)
    print(f"Best validation HS-score: {best_hs_score:.4f}")
    print("========================================\n")
    
    # Create a new model
        # Recreate the module before loading the selected checkpoint weights.
    best_model = PLModule(config, encoder)
    
    # Load the saved checkpoint
    checkpoint = torch.load(
        best_checkpoint,
        map_location="cpu"
    )
    
    best_model.load_state_dict(checkpoint["state_dict"])
    
    # Tell validation to create the confusion matrix
        # Enable detailed confusion-matrix logging for the final best-checkpoint evaluation.
    best_model.log_confusion_matrix = True
    
    # --------------------------------------------------
    # Evaluate the best checkpoint
    # --------------------------------------------------
    
        # Evaluate the selected checkpoint on the validation/test loader.
    trainer.validate(
        best_model,
        eval_dl
    )
    
    wandb.finish()


def evaluate(config):
    # only evaluation of pre-trained models
    # encoder manages encoding and decoding of model predictions
    # Adaptation: replace the AudioSet class vocabulary with normal/abnormal labels.
    # encoder = ManyHotEncoder(as_strong_train_classes)
    encoder = ManyHotEncoder(["normal", "abnormal"])
    eval_set = get_eval_dataset(encoder)

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, encoder)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch)

    # start evaluation
    trainer.validate(pl_module, eval_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet_Strong")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--evaluate', action='store_true', default=False)
        # Adaptation: run validation every epoch so checkpoint selection can track HS score.
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)

    # model
    parser.add_argument('--model_name', type=str,
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"] + \
                                [f"frame_mn{width}" for width in ["06", "10"]],
                        default="ATST-F")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument('--pretrained', type=str, choices=["scratch", "ssl", "weak", "strong"],
                        default="weak")
    parser.add_argument('--seq_model_type', type=str, choices=["rnn"],
                        default=None)

    # training
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--epoch_len', type=int, default=100000)
    parser.add_argument('--median_window', type=int, default=9)

    # augmentation
    parser.add_argument('--wavmix_p', type=float, default=0.8)
    parser.add_argument('--freq_warp_p', type=float, default=0.8)
    parser.add_argument('--filter_augment_p', type=float, default=0.8)
    parser.add_argument('--frame_shift_range', type=float, default=0.125)  # in seconds
    parser.add_argument('--mixup_p', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.3)
    parser.add_argument('--max_time_mask_size', type=float, default=0.0)

    # optimizer
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")
    parser.add_argument('--max_lr', type=float, default=7e-5)
    parser.add_argument('--lr_end', type=float, default=2e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    # knowledge distillation
    parser.add_argument('--pseudo_labels_file', type=str,
                        default=None)

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
