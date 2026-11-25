import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import get_input_chans
from datasets import load_dataset
import logging

class IsRpeakPredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str = ".",
        dataset_name: str = "brain_heart_rpeak",
        data_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        load_dataset_kwargs= dict()
    ):
        super().__init__()
        self.dataset_kwargs = load_dataset_kwargs
        self.logger = logging.getLogger(__name__)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

        
    def setup(self, stage=None):
        self.logger.info(f"Loading dataset {self.dataset_name} from {self.dataset_path} with {self.num_workers} workers...")
        ds = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            data_dir=self.data_dir,
            trust_remote_code=True,
            num_proc=self.num_workers,
            **self.dataset_kwargs
        )

        self.ds = ds
        self.train_ds = ds.get("train")
        self.val_ds = ds.get("validation")
        self.test_ds = ds.get("test")

        eeg_indices = get_input_chans(self.train_ds[0]['eeg_channels'])
        # Preprocess examples: raw EEG tensor and aggregated label
        def add_chan_indices(example):
            # EEG data as numpy array (channels, signal_length)
            return {'eeg_channel_indices': eeg_indices}

        remove_cols= list(self.train_ds.column_names)
        self.train_ds = self.train_ds.rename_columns({"eeg_data": "signal", "is_r_peak": "label"})
        self.val_ds = self.val_ds.rename_columns({"eeg_data": "signal", "is_r_peak": "label"})
        self.test_ds = self.test_ds.rename_columns({"eeg_data": "signal", "is_r_peak": "label"})
        remove_cols.remove("eeg_data")
        remove_cols.remove("eeg_channels")
        remove_cols.remove("is_r_peak")
        # Remove all other columns
        self.train_ds = self.train_ds.map(
            add_chan_indices,
            remove_columns=remove_cols,
            batched=False,
            num_proc=self.num_workers
        )
        self.val_ds = self.val_ds.map(
            add_chan_indices,
            remove_columns=remove_cols,
            batched=False,
            num_proc=self.num_workers
        )
        self.test_ds = self.test_ds.map(
            add_chan_indices,
            remove_columns=remove_cols,
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
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

if __name__ == '__main__':
    dm = IsRpeakPredictionDataModule(
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