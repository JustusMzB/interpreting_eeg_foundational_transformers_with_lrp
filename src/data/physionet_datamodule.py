import glob
import os
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, List, Optional
from zipfile import BadZipFile

import datasets.utils.file_utils
import lightning.pytorch as pl
import mne
import numpy as np
import requests
from datasets import Array2D, DatasetDict, Dataset, Features, Value, ClassLabel, Sequence
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tqdm import tqdm

from utils import get_input_chans, adapt_channel_names_to_mne, filter_eeg_channels
import logging

logger = logging.getLogger(__name__)

class PhysionetDatamodule(pl.LightningDataModule):
    # Ground-truth subject splits for all 109 EEGMMIDB subjects under seed 42.
    # Must stay in sync with _EXPECTED_* in test/data/test_physionet_datamodule.py.
    _TRAIN_SUBJECTS = [
        'S001', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011',
        'S012', 'S013', 'S014', 'S016', 'S017', 'S018', 'S019', 'S020', 'S023',
        'S025', 'S026', 'S027', 'S028', 'S029', 'S031', 'S032', 'S033', 'S034',
        'S035', 'S036', 'S037', 'S039', 'S040', 'S041', 'S042', 'S043', 'S044',
        'S045', 'S046', 'S047', 'S048', 'S049', 'S050', 'S051', 'S054', 'S055',
        'S056', 'S057', 'S058', 'S059', 'S060', 'S062', 'S063', 'S065', 'S066',
        'S067', 'S068', 'S069', 'S070', 'S071', 'S073', 'S074', 'S076', 'S077',
        'S078', 'S079', 'S080', 'S081', 'S082', 'S084', 'S085', 'S086', 'S089',
        'S090', 'S091', 'S092', 'S094', 'S095', 'S096', 'S097', 'S099', 'S101',
        'S102', 'S104', 'S105', 'S106', 'S108', 'S109',
    ]
    _VAL_SUBJECTS = [
        'S002', 'S003', 'S022', 'S024', 'S030', 'S038', 'S053', 'S064', 'S088',
        'S098', 'S107',
    ]
    _TEST_SUBJECTS = [
        'S015', 'S021', 'S052', 'S061', 'S072', 'S075', 'S083', 'S087', 'S093',
        'S100', 'S103',
    ]
    # All EEG imagery runs: LR (3,4,7,8) and hands/feet (5,6,9,10)
    _IMAGERY_RUNS = [3, 4, 5, 6, 7, 8, 9, 10]
    _PHYSIONET_FILE_BASE = "https://www.physionet.org/files/eegmmidb/1.0.0"

    def __init__(
            self,
            dataset_path: str, # TODO: Find way around this bw compatibility
            data_dir: str,
            classes: List[str] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            load_dataset_kwargs: dict = None,
            mini_n_train: Optional[int] = None,
            mini_n_val: Optional[int] = None,
            mini_n_test: Optional[int] = None,
            exclude_electrodes: Optional[str] = None,
            include_electrodes: Optional[str] = None,
            ablation_strategy: str = "drop",
            *args: Any,
            **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.dataset_url = "https://physionet.org/content/eegmmidb/get-zip/1.0.0/"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.zip_path = os.path.join(self.data_dir, "eegmmidb.zip")
        self.extracted_path = os.path.join(self.data_dir, "files")
        self.exclude_electrodes = exclude_electrodes
        self.include_electrodes = include_electrodes
        self.ablation_strategy = ablation_strategy
        self.hf_dataset = None


        # Some files have incorrect sampling frequencies in their files.
        # We identified them by observing the annotation lengths
        # They are supposed to be 4s.
        self.faulty_subjects = [
            'S092', # All non-rest events are 5.25 s instead of 4,
            'S088',
            'S0100',
        ]
        # Validate and map target classes
        valid_classes = ["left_hand", "right_hand", "both_hands", "both_feet"]
        self.classes = classes if classes is not None else valid_classes

        for cls in self.classes:
            if cls not in valid_classes:
                raise ValueError(f"Class '{cls}' invalid. Must be one of {valid_classes}")

        # Map string classes to target indices (0 to len(classes)-1) for CrossEntropyLoss
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.mini_n_train = mini_n_train
        self.mini_n_val = mini_n_val
        self.mini_n_test = mini_n_test
        self._is_mini = any(x is not None for x in (mini_n_train, mini_n_val, mini_n_test))

    def _download_physionet(self, num_threads=None):
        if num_threads is None:
            num_threads = self.num_workers
        logger.info(f"Checking server for parallel download support from {self.dataset_url}...")

        # 1. Ask the server for file metadata using a HEAD request
        head_response = requests.head(self.dataset_url, allow_redirects=True)
        head_response.raise_for_status()

        total_size = int(head_response.headers.get('content-length', 0))

        # 2. Check if the server supports partial downloads (Range requests)
        accepts_ranges = head_response.headers.get('accept-ranges', '').lower() == 'bytes'

        if not accepts_ranges or total_size == 0:
            logger.info("Server doesn't support range requests. Falling back to sequential download.")
            self._download_sequential()  # (You should keep your original code in this fallback method)
            return

        logger.info(f"Downloading ~{total_size / (1024 ** 3):.2f} GB across {num_threads} threads...")

        # 3. Pre-allocate the file on disk to the exact size needed
        with open(self.zip_path, 'wb') as f:
            f.truncate(total_size)

        # 4. Calculate byte ranges for each thread
        chunk_size = total_size // num_threads
        ranges = []
        for i in range(num_threads):
            start_byte = i * chunk_size
            # The last thread takes all remaining bytes
            end_byte = total_size - 1 if i == num_threads - 1 else start_byte + chunk_size - 1
            ranges.append((start_byte, end_byte))

        # 5. Set up thread-safe progress tracking
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        tqdm_lock = Lock()  # Needed to prevent threads from mangling the progress bar

        # 6. Define the worker function for each thread
        def download_chunk(start, end):
            headers = {'Range': f'bytes={start}-{end}'}
            # Make the GET request for just this specific byte range
            response = requests.get(self.dataset_url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()

            # Open a new file handle in read/write binary mode
            with open(self.zip_path, 'r+b') as f:
                f.seek(start)  # Jump to the correct position in the file
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        with tqdm_lock:
                            progress_bar.update(len(chunk))

        # 7. Execute the threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(download_chunk, start, end) for start, end in ranges]

            # Wait for all to finish and catch any exceptions
            for future in as_completed(futures):
                future.result()

        progress_bar.close()
        logger.info("Download complete.")

    def _download_sequential(self):
        logger.info("Starting sequential download...")
        response = requests.get(self.dataset_url, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

        with open(self.zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        logger.info("Sequential download complete.")

    def _download_mini_physionet(self) -> None:
        """Downloads EDF files for a subject subset while preserving the full split structure.

        All 109 subject directories are created so that setup()'s seed-42 shuffle
        assigns subjects to the same splits as the full dataset.  Only the selected
        subjects are populated with EDF files; empty directories are silently skipped
        during data loading.

        Files are downloaded in parallel (up to min(num_workers, 8) threads) and
        progress is reported per file so the bar advances continuously.
        """
        faulty = set(self.faulty_subjects)
        available_train = [s for s in self._TRAIN_SUBJECTS if s not in faulty]
        available_val   = [s for s in self._VAL_SUBJECTS   if s not in faulty]
        available_test  = [s for s in self._TEST_SUBJECTS  if s not in faulty]

        selected = (
            available_train[: self.mini_n_train or 0]
            + available_val  [: self.mini_n_val   or 0]
            + available_test [: self.mini_n_test  or 0]
        )
        selected_set = set(selected)

        all_subjects = sorted(self._TRAIN_SUBJECTS + self._VAL_SUBJECTS + self._TEST_SUBJECTS)
        logger.info(f"Creating {len(all_subjects)} subject directories to preserve split structure …")
        for subj in all_subjects:
            os.makedirs(os.path.join(self.extracted_path, subj), exist_ok=True)

        # Build the full list of (subject, run) pairs to download, skipping existing files.
        tasks = [
            (subj, run)
            for subj in sorted(selected_set)
            for run in self._IMAGERY_RUNS
            if not os.path.exists(
                os.path.join(self.extracted_path, subj, f"{subj}R{run:02d}.edf")
            )
        ]

        n_subjects = len(selected_set)
        n_files = len(selected_set) * len(self._IMAGERY_RUNS)
        already_done = n_files - len(tasks)
        logger.info(
            f"Downloading EDF files for {n_subjects} subjects "
            f"({self.mini_n_train or 0} train / {self.mini_n_val or 0} val / "
            f"{self.mini_n_test or 0} test) — "
            f"{len(tasks)} files to fetch, {already_done} already present …"
        )

        if not tasks:
            return

        n_threads = min(self.num_workers or 4, 8)
        tqdm_lock = Lock()

        def download_file(subj: str, run: int, progress: tqdm) -> None:
            filename = f"{subj}R{run:02d}.edf"
            dest = os.path.join(self.extracted_path, subj, filename)
            url = f"{self._PHYSIONET_FILE_BASE}/{subj}/{filename}?download"
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            with tqdm_lock:
                progress.update(1)
                progress.set_postfix_str(filename, refresh=True)

        with tqdm(total=len(tasks), unit="file", desc="Mini-PhysioMI EDF files") as progress:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = {
                    executor.submit(download_file, subj, run, progress): (subj, run)
                    for subj, run in tasks
                }
                for future in as_completed(futures):
                    subj, run = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.warning(
                            f"Failed to download {subj}R{run:02d}.edf: {exc}"
                        )

    def prepare_data(self) -> None:
        """Downloads and extracts the dataset files."""
        os.makedirs(self.data_dir, exist_ok=True)

        if self._is_mini:
            if not os.path.exists(self.extracted_path):
                self._download_mini_physionet()
            else:
                logger.info("Mini dataset already exists. Skipping download.")
            return

        if os.path.exists(self.zip_path) and not os.path.exists(self.extracted_path):
            try:
                logger.info("Extracting zip file...")
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                logger.info("Extraction complete.")
            except BadZipFile as e:
                logger.warning(f"Error extracting zip file: {e}. Removing and trying to re-download")
                os.remove(self.zip_path)

        if not os.path.exists(self.extracted_path):
            if not os.path.exists(self.zip_path):
                self._download_physionet()
                logger.info("Extracting zip file...")
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                logger.info("Extraction complete.")

        else:
            logger.info("Data already downloaded and extracted. Skipping.")

    def setup(self, stage: str=None) -> None:
        """
        Setup the dataset splits (train, validation, test).
        We do all at once as they are cached on disk as hf datasets
        """
        if self.hf_dataset is not None:
            return

        if self._is_mini:
            _mini_banner = (
                "\n" + "!" * 70 + "\n"
                "!  WARNING: MINI DATASET MODE ACTIVE                                !\n"
                f"!  train={self.mini_n_train or 0} / val={self.mini_n_val or 0} / test={self.mini_n_test or 0} subjects populated with data.   !\n"
                "!  Results are NOT representative of the full experiment.           !\n"
                "!" * 70
            )
            warnings.warn(_mini_banner, UserWarning, stacklevel=2)
            logger.warning(_mini_banner)

        subject_dirs = sorted(glob.glob(os.path.join(self.extracted_path, "S*")))
        subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]

        np.random.seed(42)
        np.random.shuffle(subject_ids)
        n_subs = len(subject_ids)

        splits = {
            "train": subject_ids[:int(0.8 * n_subs)],
            "validation": subject_ids[int(0.8 * n_subs):int(0.9 * n_subs)],
            "test": subject_ids[int(0.9 * n_subs):]
        }

        extracted_path = self.extracted_path
        classes = list(self.classes)
        class_to_idx = dict(self.class_to_idx)
        faulty_subjects = set(self.faulty_subjects)

        def eeg_generator(subjects):
            mne.set_log_level('WARNING')

            for subj in subjects:
                edf_files = sorted(glob.glob(os.path.join(extracted_path, subj, "*.edf")))
                for file_path in edf_files:
                    try:
                        basename = os.path.basename(file_path)
                        run_num = int(basename.split('.')[0].split('R')[-1])
                        # https://www.physionet.org/content/eegmmidb/1.0.0/ for details on how trials/runs relate to stimuli
                        allowed_annotations = {}
                        if run_num in [3, 4, 7, 8]:
                            if "left_hand" in classes:
                                allowed_annotations["T1"] = class_to_idx["left_hand"]
                            if "right_hand" in classes:
                                allowed_annotations["T2"] = class_to_idx["right_hand"]
                        elif run_num in [5, 6, 9, 10]:
                            if "both_hands" in classes:
                                allowed_annotations["T1"] = class_to_idx["both_hands"]
                            if "both_feet" in classes:
                                allowed_annotations["T2"] = class_to_idx["both_feet"]

                        if not allowed_annotations:
                            continue

                        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                        # logger.info(f"Sampling Frequency (sfreq): {raw.info['sfreq']} Hz")
                        # logger.info(f"average annotation_length: {raw.annotations.duration.mean()}")
                        if raw.info['sfreq'] != 160.0:

                            logger.info(
                                f'found unusual file with wrong sampling rate.\n'
                                f'belongs to subject {subj} and run {run_num}\n'
                                f"Raw data sampling frequency is {raw.info['sfreq']} in file {file_path} but expected 160.0 Hz.\n"
                                f'expect an error, which leads to automatic exclusion of this file'
                            )
                        # SIGNAL PROCESSING
                        raw = raw.set_eeg_reference('average', verbose=False)
                        raw = raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
                        raw = raw.notch_filter(freqs=50.0, verbose=False)

                        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

                        if len(events) == 0:
                            continue

                        mne_to_target = {}
                        for ann, target_idx in allowed_annotations.items():
                            if ann in event_id_map:
                                mne_to_target[event_id_map[ann]] = target_idx

                        if not mne_to_target:
                            continue

                        mask = np.isin(events[:, 2], list(mne_to_target.keys()))
                        filtered_events = events[mask]

                        if len(filtered_events) == 0:
                            continue

                        epochs = mne.Epochs(raw, filtered_events, tmin=0, tmax=4.0,
                                            baseline=None, preload=True, verbose=False)
                        # THIS DROPS EPOCHS IF THEY ARE TOO SHORT
                        epochs = epochs.resample(sfreq=200, verbose=False) # Resample after epoch, avoid event mismatch
                        data = epochs.get_data()
                        # MANUAL SCALING
                        data = data[:, :, :800]
                        data = data * 10000.0

                        for i in range(len(epochs)):
                            mne_label = int(filtered_events[i, 2])
                            target_label = mne_to_target[mne_label]

                            yield {
                                "subject_id": subj,
                                "run_file": basename,
                                "event_label": target_label,
                                "event_name": classes[target_label],
                                "eeg_channels": epochs.ch_names,  # <-- Added Channel Names here
                                "eeg_signal": data[i].tolist()
                            }
                    except Exception as e:
                        if subj in faulty_subjects:
                            logger.info(f"As expected: Excluding file {file_path} from subject {subj} due to error: {e}")
                        else:
                            logger.warning(f"Skipping {file_path} due to error: {e}")
                        continue

        eeg_features = Features({
            "subject_id": Value("string"),
            "run_file": Value("string"),
            "event_label": ClassLabel(names=self.classes),
            "event_name": Value("string"),
            "eeg_channels": Sequence(Value("string")),
            "eeg_signal": Array2D(shape = (64, 800), dtype="float32")
        })

        print(f"Building Hugging Face Datasets for classes: {self.classes}...")

        self.hf_dataset = DatasetDict({
            split_name: Dataset.from_generator(
                eeg_generator,
                features=eeg_features,
                gen_kwargs={"subjects": subs},
                num_proc=self.num_workers,
                split=split_name,
            ) for split_name, subs in splits.items()
        })

        if len(self.hf_dataset['train']) == 0:
            raise ValueError("Train dataset is empty! Check generator logging.")

        raw_ch_names = adapt_channel_names_to_mne(self.hf_dataset['train'][0]['eeg_channels'])
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

        def combined_map(example):
            example['eeg_channel_indices'] = eeg_channel_indices
            sig = np.array(example['eeg_signal'])
            if exclude_electrodes or include_electrodes:
                if ablation_strategy == "drop":
                    sig = sig[keep_indices, :]
                elif ablation_strategy == "zero":
                    sig[excluded_indices, :] = 0.0
            std_val = np.std(sig)
            if std_val > 0:
                sig = (sig - np.mean(sig)) / std_val
            example['eeg_signal'] = sig
            return example


        remove_cols = sorted(list(
            set(self.hf_dataset['train'].column_names) - {'eeg_signal', 'event_label', 'eeg_channel_indices'}
        )) # Sorted list for determinism in hash, so that cache is used.

        n_chans = len(keep_indices) if (exclude_electrodes or include_electrodes) and ablation_strategy == "drop" else 64
        mapped_features = Features({
            "subject_id": Value("string"),
            "run_file": Value("string"),
            "event_label": ClassLabel(names=self.classes),
            "event_name": Value("string"),
            "eeg_channels": Sequence(Value("string")),
            "eeg_signal": Array2D(shape=(n_chans, 800), dtype="float32"),
            "eeg_channel_indices": Sequence(Value("int64"))
        })

        self.hf_dataset = self.hf_dataset.map(
            combined_map,
            features=mapped_features,
            num_proc=self.num_workers if self.num_workers > 0 else 1,
            # remove_columns=remove_cols

        ).rename_columns({
            "eeg_signal": "signal",
            "event_label": "label",
        })
        self.hf_dataset.set_format(
            type="torch",
            columns=["signal", "label", 'eeg_channel_indices'],
            output_all_columns=True
        )
        print("Dataset preparation complete!")

        # Some parts of the pipeline require the full ds
        self.train_ds = self.hf_dataset['train']
        self.val_ds = self.hf_dataset['validation']
        self.test_ds = self.hf_dataset['test']

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.hf_dataset["train"],
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.hf_dataset["validation"],
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.hf_dataset["test"],
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)