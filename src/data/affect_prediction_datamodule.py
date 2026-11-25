import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import get_input_chans, filter_eeg_channels
from datasets import load_dataset
import logging
from typing import Literal
from datasets.fingerprint import Hasher

class AffectPredictionDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the BrainHeartEmotion dataset,
    classifying the tertile split of arousal.

    Configurable parameters:
      - dataset_path: local or HuggingFace path to dataset script
      - dataset_name: subset name within the script
      - data_dir: cache directory for data files

    Splits:
      - train: fine-tuning
      - validation/test: evaluation

    Each batch returns:
      - signal: Tensor of shape [batch, channels, signal_length]
      - label: Integer class 0=low,1=medium,2=high
    """
    def __init__(
        self,
        dataset_path: str = ".",
        dataset_name: str = "brain_heart_emotion",
        data_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        emotional_dimension: Literal["arousal", "valence"] = 'arousal',
        num_classes: int = 3,
        sliding_window_step: int = None,
        exclude_electrodes: str = None,
        include_electrodes: str = None,
        ablation_strategy: Literal["drop", "zero"] = "drop",
        load_dataset_kwargs: dict = {}
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = num_classes
        self.emotional_dimension = emotional_dimension
        self.exclude_electrodes = exclude_electrodes
        self.include_electrodes = include_electrodes
        self.ablation_strategy = ablation_strategy
        self.logger = logging.getLogger(__name__)
        self.sliding_window_step = sliding_window_step
        self.load_dataset_kwargs = load_dataset_kwargs

    def prepare_data(self):
        pass

        

    def setup(self, stage=None):
        # Load dataset splits from cache
        self.logger.debug(f"Loading dataset {self.dataset_name} from {self.dataset_path} with {self.num_workers} workers...")
        ds = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            data_dir=self.data_dir,
            trust_remote_code=True,
            num_proc=self.num_workers,
            sliding_window=self.sliding_window_step,
            **self.load_dataset_kwargs
        )
        self.ds = ds

        self.train_ds = ds.get("train")
        self.val_ds = ds.get("validation")
        self.test_ds = ds.get("test")
    
        # Preprocess examples: raw EEG tensor and aggregated label
        self.train_ds = self.train_ds.rename_column("eeg_data", "signal")
        self.val_ds = self.val_ds.rename_column("eeg_data", "signal")
        self.test_ds = self.test_ds.rename_column("eeg_data", "signal")
        if self.num_classes == 3:
            self.train_ds = self.train_ds.rename_column(f'{self.emotional_dimension}_tertile_split_label', 'label')
            self.val_ds = self.val_ds.rename_column(f'{self.emotional_dimension}_tertile_split_label', 'label')
            self.test_ds = self.test_ds.rename_column(f'{self.emotional_dimension}_tertile_split_label', 'label')
        elif self.num_classes == 2:
            self.train_ds = self.train_ds.rename_column(f'{self.emotional_dimension}_binary_split_label', 'label')
            self.val_ds = self.val_ds.rename_column(f'{self.emotional_dimension}_binary_split_label', 'label')
            self.test_ds = self.test_ds.rename_column(f'{self.emotional_dimension}_binary_split_label', 'label')
        else:
            raise ValueError(f"Unsupported number of classes: {self.num_classes}. Must be 2 or 3.")
        
        raw_ch_names = self.train_ds[0]['eeg_channels']
        keep_indices, excluded_indices, filtered_ch_names = filter_eeg_channels(
            raw_ch_names, self.exclude_electrodes, self.include_electrodes
        )
        self.keep_indices = keep_indices
        self.excluded_indices = excluded_indices
        self.filtered_ch_names = filtered_ch_names

        full_eeg_channel_indices = get_input_chans(raw_ch_names)
        if self.exclude_electrodes or self.include_electrodes:
            if self.ablation_strategy == "drop":
                eeg_channel_indices = [full_eeg_channel_indices[0]] + [full_eeg_channel_indices[i + 1] for i in keep_indices]
            else:
                eeg_channel_indices = full_eeg_channel_indices
        else:
            eeg_channel_indices = full_eeg_channel_indices

        exclude_electrodes = self.exclude_electrodes
        include_electrodes = self.include_electrodes
        ablation_strategy = self.ablation_strategy

        def process_example(example):
            """
            Add EEG channel indices and apply electrode ablation if specified.
            """
            example['eeg_channel_indices'] = eeg_channel_indices
            if exclude_electrodes or include_electrodes:
                signal_arr = np.array(example['signal'])
                if ablation_strategy == "drop":
                    example['signal'] = signal_arr[keep_indices, :]
                elif ablation_strategy == "zero":
                    signal_arr[excluded_indices, :] = 0.0
                    example['signal'] = signal_arr
            return example
        
        remove_cols = list(set(self.train_ds.column_names) - {'signal', 'label', 'eeg_channel_indices'})
        remove_cols.sort() # Sort for consistent hashing
        
        logging.debug(f"Removing columns: {remove_cols} with hash {Hasher.hash(remove_cols)}")
        logging.debug(f"EEG channel indices: {eeg_channel_indices} with hash {Hasher.hash(eeg_channel_indices)}")
        logging.debug(f"function process_example hash: {Hasher.hash(process_example)}")

        from datasets import Features, Array2D, Sequence, Value

        signal_feat = self.train_ds.features['signal']
        orig_n_chans, signal_len = signal_feat.shape
        signal_dtype = signal_feat.dtype

        n_chans = len(keep_indices) if (exclude_electrodes or include_electrodes) and ablation_strategy == "drop" else orig_n_chans

        new_features = Features({
            'signal': Array2D(shape=(n_chans, signal_len), dtype=signal_dtype),
            'label': self.train_ds.features['label'],
            'eeg_channel_indices': Sequence(Value('int64'))
        })

        # Remove all other columns and process example
        self.train_ds = self.train_ds.map(
            process_example,
            remove_columns=remove_cols,
            features=new_features,
            batched=False,
            num_proc=self.num_workers
        )
        self.val_ds = self.val_ds.map(
            process_example,
            remove_columns=remove_cols,
            features=new_features,
            batched=False,
            num_proc=self.num_workers
        )
        self.test_ds = self.test_ds.map(
            process_example,
            remove_columns=remove_cols,
            features=new_features,
            batched=False,
            num_proc=self.num_workers
        )

        # Set PyTorch tensor format
        self.train_ds.set_format(type='torch', columns=['signal', 'label', 'eeg_channel_indices'])
        self.val_ds.set_format(type='torch', columns=['signal', 'label', 'eeg_channel_indices'])
        self.test_ds.set_format(type='torch', columns=['signal', 'label', 'eeg_channel_indices'])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        # Alias of validation
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) 

if __name__ == '__main__':
    dm = AffectPredictionDataModule(
        dataset_path='./src/data/brain_heart_emotion',
        dataset_name='brain_heart_emotion',
        data_dir='data/emotional_state_physio_neural/phase3/derivatives/preproc',
        batch_size=4
    )
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    signals, labels = batch['signal'], batch['label']
    print(f"Signals batch shape: {signals.shape}")
    print(f"Labels batch shape: {labels.shape}")