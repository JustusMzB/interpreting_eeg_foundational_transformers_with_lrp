import os
import sys
import pytest
import numpy as np
import torch

# Add src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import (
    standard_1020,
    normalize_electrode_name,
    parse_electrode_list,
    filter_eeg_channels,
    get_input_chans
)

def test_normalize_electrode_name():
    assert normalize_electrode_name("p6") == "P6"
    assert normalize_electrode_name("FCz.") == "FCZ"
    assert normalize_electrode_name("Fp1-") == "FP1"
    assert normalize_electrode_name("FFC6") == "FC6"

def test_parse_electrode_list():
    res1 = parse_electrode_list("P6, FCz, CPz")
    assert res1 == {"P6", "FCZ", "CPZ"}

    res2 = parse_electrode_list(["AF7", "FP1", "FPZ", "FP2"])
    assert res2 == {"AF7", "FP1", "FPZ", "FP2"}

def test_filter_eeg_channels_exclude():
    ch_names = ["FP1", "FP2", "FCz", "Cz", "P6", "O1"]
    keep_indices, excluded_indices, filtered_ch = filter_eeg_channels(
        ch_names, exclude_electrodes="P6, FCz"
    )
    assert keep_indices == [0, 1, 3, 5]
    assert excluded_indices == [2, 4]
    assert filtered_ch == ["FP1", "FP2", "Cz", "O1"]

def test_filter_eeg_channels_include():
    ch_names = ["FP1", "FP2", "FCz", "Cz", "P6", "O1"]
    keep_indices, excluded_indices, filtered_ch = filter_eeg_channels(
        ch_names, include_electrodes="FCz, Cz, P6"
    )
    assert keep_indices == [2, 3, 4]
    assert excluded_indices == [0, 1, 5]
    assert filtered_ch == ["FCz", "Cz", "P6"]

def test_arousal_ablation_conditions():
    arousal_channels = [
        'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
        'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
        'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
    ]
    assert len(arousal_channels) == 60

    # 1. Exclude P6
    k1, e1, f1 = filter_eeg_channels(arousal_channels, exclude_electrodes="P6")
    assert len(k1) == 59
    assert "P6" not in f1

    # 2. Exclude Arousal Cluster (FCz, FC2, C2, CP2, CPz, Cz)
    cluster = "FCz, FC2, C2, CP2, CPz, Cz"
    k2, e2, f2 = filter_eeg_channels(arousal_channels, exclude_electrodes=cluster)
    assert len(k2) == 54

    # 3. Exclude Both
    k3, e3, f3 = filter_eeg_channels(arousal_channels, exclude_electrodes=f"{cluster}, P6")
    assert len(k3) == 53

    # 4. Include Only Cluster + P6
    k4, e4, f4 = filter_eeg_channels(arousal_channels, include_electrodes=f"{cluster}, P6")
    assert len(k4) == 7
    assert set(f4) == {"FCZ", "FC2", "C2", "CP2", "CPZ", "CZ", "P6"}

def test_physiomi_ablation_conditions():
    physio_channels = [
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
        'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'F1', 'FZ', 'F2', 'P1', 'PZ', 'P2', 'AF7', 'AF3', 'AFZ',
        'AF4', 'AF8', 'F7', 'F5', 'F3', 'F4', 'F6', 'F8', 'FT7', 'FT8',
        'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P4',
        'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2',
        'IZ', 'FP1', 'FPZ', 'FP2'
    ]
    # 1. Exclude Ocular
    ocular = "AF7, FP1, FPZ, FP2, AF8, AF3, AFz, AF4"
    k1, e1, f1 = filter_eeg_channels(physio_channels, exclude_electrodes=ocular)
    assert len(k1) == 56

    # 2. Exclude Facial Motorcortex related
    facial = "T9, T10, T7, T8, FT7, FC5, C5, FFC6, C6"
    k2, e2, f2 = filter_eeg_channels(physio_channels, exclude_electrodes=facial)
    assert len(k2) == 55

    # 3. Include Only Facial Motorcortex related
    k3, e3, f3 = filter_eeg_channels(physio_channels, include_electrodes=facial)
    assert len(k3) == 9

def test_dataset_ablation_drop_formatting():
    from datasets import Dataset, Features, Array2D, Value, Sequence, ClassLabel
    from torch.utils.data import DataLoader

    # Construct synthetic dataset mimicking raw EEG dataset (64 channels, 800 time steps)
    raw_data = {
        "signal": [np.random.randn(64, 800).astype(np.float32) for _ in range(4)],
        "label": [0, 1, 0, 1],
        "eeg_channels": [["Ch" + str(i) for i in range(64)] for _ in range(4)],
    }
    raw_features = Features({
        "signal": Array2D(shape=(64, 800), dtype="float32"),
        "label": ClassLabel(names=["class0", "class1"]),
        "eeg_channels": Sequence(Value("string")),
    })
    ds = Dataset.from_dict(raw_data, features=raw_features)

    # Simulate dropping 8 electrodes (keeping 56 electrodes)
    keep_indices = list(range(56))
    n_chans = len(keep_indices)

    def process_fn(example):
        sig = np.array(example["signal"])[keep_indices, :]
        return {"signal": sig, "label": example["label"], "eeg_channel_indices": list(range(57))}

    new_features = Features({
        "signal": Array2D(shape=(n_chans, 800), dtype="float32"),
        "label": ClassLabel(names=["class0", "class1"]),
        "eeg_channel_indices": Sequence(Value("int64")),
    })

    mapped_ds = ds.map(process_fn, remove_columns=["eeg_channels"], features=new_features)
    mapped_ds.set_format(type="torch", columns=["signal", "label", "eeg_channel_indices"])

    loader = DataLoader(mapped_ds, batch_size=2)
    batch = next(iter(loader))

    assert batch["signal"].shape == torch.Size([2, 56, 800])
    assert batch["label"].shape == torch.Size([2])

