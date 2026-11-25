import lightning.pytorch as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from models.modeling_finetune import NeuralTransformer
from utils import load_state_dict, BinaryBalancedAccuracy
from typing import Literal
import torchmetrics
import logging
import time
logger = logging.getLogger(__name__)
import logging
logger = logging.getLogger(__name__)

class NeuralTransformerLightning(pl.LightningModule):
    """
    Lightning wrapper for NeuralTransformer from modeling_finetune.py.
    Supports:
      - automatic reshaping of raw EEG signals into patches
      - head swapping with dimension checks
      - custom loss functions
      - pretrained checkpoint loading
    """

    def __init__(
            self,
            model_config: dict,
            num_classes: int = 1000,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-2,
            loss_fn: callable = F.cross_entropy,
            custom_head: nn.Module = None,
            pretrained_path: str = None,
            freeze_backbone: bool = False,
            output_modality: Literal["global", "patchwise", "seq2seq"] = "seq2seq",
            task_type: Literal["multiclass", "binary"] = "multiclass",
            track_gradients: bool = False,
            warmup_epochs: int = 0,
            warmup_start_lr: float = 0.0,
            min_lr: float = 0,
    ):
        """
        Initializes Lighting-wrapped LaBraM Neural Transformer.
        Args:
          model_config: Configuration dictionary for the model in the unwrapped implementation.
          num_classes: Number of output classes in the task.
          learning_rate: Learning rate for the optimizer.
          weight_decay: Weight decay for the optimizer.
          loss_fn: Loss function to use.
          custom_head: Optional custom head for the model.
          pretrained_path: Optional path to a pretrained model by the LaBraM makers. If present, will be loaded.
          freeze_backbone: Whether to freeze the backbone model.
          output_modality: Output modality for the model, as in ["global", "patchwise", "seq2seq"].
          task_type: Task type (multiclass or binary).
          track_gradients: Whether to track the gradients of the model during training. Takes storage.
        """
        super().__init__()
        # Save hyperparameters (excluding non-serializable)
        self.save_hyperparameters(ignore=["loss_fn", "custom_head", "pretrained_path"])
        # Extract reshape params
        self.num_classes = num_classes
        self.signal_length = model_config.get('EEG_size')
        self.patch_size = model_config.get('patch_size')
        self.freeze_backbone = freeze_backbone
        self.track_grad_norm = track_gradients



        if self.signal_length is None or self.patch_size is None:
            raise ValueError("model_config must include 'EEG_size' and 'patch_size' for reshaping input")
        if self.signal_length % self.patch_size != 0:
            raise ValueError(f"EEG_size ({self.signal_length}) must be divisible by patch_size ({self.patch_size})")
        self.num_patches = self.signal_length // self.patch_size
        self.channel_amount = model_config.get('in_channels')

        # Initialize core model
        self.model = NeuralTransformer(
            num_classes=num_classes if task_type=="multiclass" else 1,
            **model_config
        )
        self.loss_fn = loss_fn
        self.output_modality = output_modality
        
        if output_modality == "seq2seq":
            self.model.head = Seq2SeqHead(
                in_features = self.channel_amount,
                out_features = num_classes if task_type == "multiclass" else 1,
            )  
        elif output_modality == "patchwise":
            self.model.head = PatchwiseHead(
                in_channels = self.channel_amount,
                in_patch_size = self.model.embed_dim,
                out_features = num_classes if task_type == "multiclass" else 1,
                hidden_dim = model_config.get('head_hidden_dim', None),
                n_hidden_layers = model_config.get('head_n_hidden_layers', 0),
                dropout = model_config.get('drop_rate')
            )
            
        # Custom head swap
        self.model.head = custom_head if custom_head else self.model.head

        # Load pretrained weights
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)
            load_state_dict(self.model, state_dict)
            print(f"Loaded pretrained weights from {pretrained_path}")
        
        if self.freeze_backbone:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze head
            for param in self.model.head.parameters():
                param.requires_grad = True
        # Save optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr


        # Define metrics based on task type
        if self.hparams.task_type == 'binary':
            metrics = torchmetrics.MetricCollection({
                'accuracy': torchmetrics.Accuracy(task='binary'),
                'balanced_accuracy': BinaryBalancedAccuracy(),
                'precision': torchmetrics.Precision(task='binary'),
                'recall': torchmetrics.Recall(task='binary'),
                'f1': torchmetrics.F1Score(task='binary'),
                'auroc': torchmetrics.AUROC(task='binary'),
            })
            confmatrix = torchmetrics.ConfusionMatrix(task='binary')
        else:  # multiclass
            metrics = torchmetrics.MetricCollection({
                'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes),
                'balanced_accuracy': torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average='macro', task='multiclass'),
                'auroc': torchmetrics.AUROC(task='multiclass', num_classes=self.hparams.num_classes),
            })
            confmatrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes)
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.train_confmatrix = confmatrix.clone()
        self.val_confmatrix = confmatrix.clone()
        self.test_confmatrix = confmatrix.clone()
        logger.info(f"Initialized NeuralTransformerLightning with output modality '{self.output_modality}' and task type '{self.hparams.task_type}'")

    def forward(self, x: torch.Tensor, input_chans) -> torch.Tensor:
        """
        Forward pass with automatic reshaping.

        Args:
            x: Tensor of shape:
               - [batch, in_chans, signal_length], or
               - [batch, in_chans, num_patches, patch_size]
        Returns:
            logits from NeuralTransformer
        """
        # Reshape raw EEG to patches if needed
        if x.dim() == 3:
            B, C, L = x.shape
            assert L == self.signal_length, \
                f"Expected signal length {self.signal_length}, got {L}"
            x = x.view(B, C, self.num_patches, self.patch_size)
        elif x.dim() == 4:
            B, C, A, T = x.shape
            assert A == self.num_patches and T == self.patch_size, \
                f"Expected input shape [B, C, {self.num_patches}, {self.patch_size}], got [B, C, {A}, {T}]"
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {tuple(x.shape)}")
        # As of now, model only works with identical input channels in one batch.
        assert input_chans.eq(input_chans[0]).all(), "Input channels must be identical in one batch."
        
        if self.output_modality == "patchwise":
            out = self.model(x, input_chans[0], return_patch_tokens=True)
        elif self.output_modality == "seq2seq": #ToDo: Output of model is flattened in eeg dimension...
            out = self.model(x, input_chans[0], return_patch_tokens=True)
        elif self.output_modality == "global":
            out = self.model(x, input_chans[0], return_patch_tokens=False)
        else:
            raise ValueError(f"Invalid output modality: {self.output_modality}")
        
        logger.debug(f"Forward pass output shape: {out.shape} for output modality '{self.output_modality}'")
        return out

    def reset_head(self, new_head: nn.Module):
        """
        Replace classification head, ensuring dimension compatibility.
        """
        assert hasattr(new_head, 'in_features'), \
            f"New head must have 'in_features', got {type(new_head)}"
        expected_dim = self.model.embed_dim
        assert new_head.in_features == expected_dim, \
            f"New head in_features ({new_head.in_features}) != model embed_dim ({expected_dim})"
        self.model.head = new_head

    def shared_step(self, batch):
        start_time_step = time.time()
        x, y, chans = batch['signal'], batch['label'], batch['eeg_channel_indices']
        logits = self(x, chans)
        # Returned logits shape: [batch, n_channels * n_patches, n_classes] in case of default head (patchwise, channelwise)
        # If our head is seq2seq, we need to reshape it to [batch, n_patches * patch_size], n_classes]
        # If our head is patchwise, we need to reshape it to [batch, n_patches, n_classes].

        if self.hparams.task_type == "binary":
            logits = logits[:, 0] # For binary mode, extract first channel logit
        
        # We should do this through custom heads and appropriate label formatting.
        loss = self.loss_fn(logits, y)
        end_time_step = time.time()
        logger.debug(f"Time taken for shared_step: {end_time_step - start_time_step:.4f} seconds")
        logger.debug(f'completed loss calculation in shared_step with task type {self.hparams.task_type}')
        logger.debug(f"Calculated loss: {loss.item()} with logits shape {logits.shape} and labels shape {y.shape}")
        return loss, logits

    def log_confusion_matrix(self, phase, confusion_matrix):
        """
        Plot metrics for visualization.
        """
        # Confusion matrix
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
             fig, ax_ = confusion_matrix.plot(cmap="Blues",)
             self.logger.experiment.add_figure(
                 f"{phase}_confusion_matrix",
                 fig , 
                 global_step=self.global_step, 
                 close=True,
                 )
             logger.debug(f"Logged confusion matrix for {phase} phase at global step {self.global_step}")


    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        start_time_metrics = time.time()
        self.train_metrics.update(preds, batch['label'])
        self.train_confmatrix.update(preds, batch['label'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)
        end_time_metrics = time.time()
        logger.debug(f"Time taken for training metrics: {end_time_metrics - start_time_metrics:.4f} seconds")
        logger.debug(f"Completed training step for batch {batch_idx} with loss {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        start_time_metrics = time.time()
        self.val_metrics.update(preds, batch['label'])
        self.val_confmatrix.update(preds, batch['label'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False)
        end_time_metrics = time.time()
        logger.debug(f"Time taken for validation metrics: {end_time_metrics - start_time_metrics:.4f} seconds")
        logger.debug(f"Completed validation step for batch {batch_idx} with loss {loss.item()}")
    

    def test_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        self.test_metrics.update(preds, batch['label'])
        self.test_confmatrix.update(preds, batch['label'])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        logger.debug(f"Completed test step for batch {batch_idx} with loss {loss.item()}")
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        self.log_confusion_matrix("train", self.train_confmatrix)
        self.train_confmatrix.reset()
    
    def on_test_epoch_end(self):
        self.log_confusion_matrix("test", self.test_confmatrix)
        self.test_confmatrix.reset()
    
    def on_validation_epoch_end(self):
        self.log_confusion_matrix("val", self.val_confmatrix)
        self.val_confmatrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.head.parameters() if self.freeze_backbone else self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs = self.warmup_epochs,
            warmup_start_lr = self.warmup_start_lr,
            eta_min = self.min_lr,
            max_epochs=self.trainer.max_epochs if hasattr(self, 'trainer') else 10,
        )
        logger.debug(f"initialized optimizers with learning rate {self.hparams.learning_rate} and weight decay {self.hparams.weight_decay}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_before_optimizer_step(self, optimizer):
        if self.track_grad_norm:
            # Log gradient norm
            norms = pl.utilities.grad_norm(self, norm_type=2)
            self.log_dict(norms)
            logger.debug(f"Logged gradient norms")

class Seq2SeqHead(nn.Module):
    """
    Custom head for seq2seq output.
    """
    def __init__(self, in_features, out_features: int, kernel_size: int = 11):
        super().__init__()
        assert kernel_size % 2, f"Kernel size must be odd, got {kernel_size}"
        self.in_features = in_features
        self.out_features = out_features
        # Using convolutional layer to assemble all channels into out_features.
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_channels*num_patches, patch_size] in "return_patch_tokens" mode
        x = rearrange(x, "B (NC NP) T -> B NC (NP T)", NC=self.in_features ) # Rearrange into multivariate time series
        out = self.conv(x)
        return out
    
class PatchwiseHead(nn.Module):
    """
    Custom head for patchwise output per timestep.
    Summarizes channels.
    """
    def __init__(self, in_channels: int, in_patch_size,out_features: int, dropout: float = 0.0, hidden_dim: int = None, n_hidden_layers: int = 0):
        super().__init__()
        self.in_features = in_channels* in_patch_size
        self.in_channels = in_channels
        self.in_patch_size = in_patch_size
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden = None
        if n_hidden_layers and n_hidden_layers > 0:
            first_layer = nn.Linear(self.in_features, self.hidden_dim)
            hidden_layers = []
            for _ in range(n_hidden_layers - 1):
                hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                hidden_layers.append(nn.ReLU())
                hidden_layers.append(self.dropout)
            last_layer = nn.Linear(self.hidden_dim, self.out_features)
            self.layers = nn.Sequential(
                first_layer,
                self.dropout,
                *hidden_layers,
                nn.ReLU(),
                last_layer,
            )
        else:
            self.layers = nn.Linear(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_channels*num_patches, patch_size] in "return_patch_tokens" mode
        x = rearrange(x, "B (NC NP) T -> B NP (NC T)",NC=self.in_channels) 
        x = self.dropout(x)
        # x shape: [batch, num_patches, num_channels * patch_size]
        out = self.layers(x)  # Apply linear layer to summarize channels
        # out shape: [batch, num_patches, out_features]
        # for cross entropy loss, we need to reshape to (batch, num classes, num_patches)
        out = rearrange(out, "B NP C -> B C NP")
        return out