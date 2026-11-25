
import mne
import os
import datasets
import pandas as pd
import random
import bids
import numpy as np
import json
import logging

LOGGER = logging.getLogger(__name__)
random.seed(42)
_CITATION = """\
@article{fourcade2024linking,
  title={Linking brain--heart interactions to emotional arousal in immersive virtual reality},
  author={Fourcade, Antonin and Klotzsche, Felix and Hofmann, SM and Mariola, A and Nikulin, Vadim V and Villringer, Arno and Gaebler, Michael},
  journal={Psychophysiology},
  volume={61},
  number={12},
  pages={e14696},
  year={2024},
  publisher={Wiley Online Library}
}
"""

_DESCRIPTION = """\
This dataset is derived / comparable to data raised in the study "Linking brain--heart interactions to emotional arousal in immersive virtual reality" by Fourcade et al. (2024).
It is assembled from the partially preprocessed BIDS dataset, thus requiring a custom loading script.
"""

_HOMEPAGE = ""
_LICENSE = "NOT AVAILABLE FOR USE"""

class BrainHeartEmotionConfig(datasets.BuilderConfig):
    """BuilderConfig for BrainHeartEmotion."""

    def __init__(self,
                 name,
                 version,
                 features,
                 standardize_input,
                 preproc_dir,
                 example_length_in_seconds,
                 expand_low_frequency_signals=False,
                 majority_vote_labels=True,
                 tertiary_split_labels=True,
                 binary_split_labels=False,
                 sliding_window=None,
                 kfold_k=None,
                 kfold_seed=None,
                 kfold_index=None,
                 **kwargs
                 ):
        super(BrainHeartEmotionConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            description=_DESCRIPTION,
            **kwargs,
        )
        assert not (majority_vote_labels and expand_low_frequency_signals), \
            "majority_vote_labels and expand_low_frequency_signals cannot be used together"

        self.majority_vote_labels = majority_vote_labels
        self.data_dir = preproc_dir
        self.standardize_input = standardize_input
        self.expand_low_frequency_signals = expand_low_frequency_signals
        self.features = features
        self.data_dir = preproc_dir
        self.standardize_input = standardize_input
        self.expand_low_frequency_signals = expand_low_frequency_signals
        self.features = features
        self.example_length_in_seconds = example_length_in_seconds
        self.tertiary_split_labels = tertiary_split_labels
        self.binary_split_labels = binary_split_labels
        self.sliding_window = sliding_window
        self.kfold_k = kfold_k
        self.kfold_seed = kfold_seed
        self.kfold_index = kfold_index


class BrainHeartEmotion(datasets.GeneratorBasedBuilder):
    """Brain-Heart-Emotion dataset."""
    VERSION = "0.0.2"
    BUILDER_CONFIGS = [
        BrainHeartEmotionConfig(
            name="brain_heart_emotion",
            preproc_dir="path/to/your/preproc/directory",
            version=VERSION,
            features=datasets.Features({
                "subject": datasets.Value("string"),
                "eeg_channels": datasets.Sequence(datasets.Value("string")),
                "eeg_data": datasets.Array2D(shape=(60, 800), dtype="float64"),
                "valence": datasets.Sequence(datasets.Value("float64")),
                "valence_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"], num_classes=3)),
                "valence_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"], num_classes=2)),
                "arousal": datasets.Sequence(datasets.Value("float64")),
                "arousal_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"], num_classes=3),),
                "arousal_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"], num_classes=2)),
                "flubber_frequency": datasets.Sequence(datasets.Value("float64")),
                "flubber_amplitude": datasets.Sequence(datasets.Value("float64")),
            }),
            standardize_input=True,
            expand_low_frequency_signals=False,
            majority_vote_labels=False,
            binary_split_labels=True,
            example_length_in_seconds=4,
            sliding_window=1
        ),
        BrainHeartEmotionConfig(
            name="brain_heart_rpeak",
            preproc_dir="path/to/your/preproc/directory",
            version=VERSION,
            features=datasets.Features({
                "subject": datasets.Value("string"),
                "eeg_channels": datasets.Sequence(datasets.Value("string")),
                "eeg_data": datasets.Array2D(shape=(60, 800), dtype="float64"),
                "valence": datasets.Sequence(datasets.Value("float64")),
                "valence_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"])),
                "valence_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"])),
                "arousal": datasets.Sequence(datasets.Value("float64")),
                "arousal_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"])),
                "arousal_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"])),
                "flubber_frequency": datasets.Sequence(datasets.Value("float64")),
                "flubber_amplitude": datasets.Sequence(datasets.Value("float64")),
                "ecg_data": datasets.Sequence(datasets.Value("float64")),
                "is_r_peak": datasets.Sequence(datasets.ClassLabel(names=["0", "1"])),
            }),
            standardize_input=True,
            expand_low_frequency_signals=False,
            majority_vote_labels=False,
            tertiary_split_labels=False,
            binary_split_labels=True,
            example_length_in_seconds=4,
            sliding_window=1
        ),
        BrainHeartEmotionConfig(
            name="brain_heart_emotion_long",
            preproc_dir="path/to/your/preproc/directory",
            version=VERSION,
            features=datasets.Features({
                "subject": datasets.Value("string"),
                "eeg_channels": datasets.Sequence(datasets.Value("string")),
                "eeg_data": datasets.Array2D(shape=(60, 2400), dtype="float64"),
                "valence": datasets.Sequence(datasets.Value("float64")),
                "valence_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"], num_classes=3)),
                "valence_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"], num_classes=2)),
                "arousal": datasets.Sequence(datasets.Value("float64")),
                "arousal_tertiary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "medium", "high"], num_classes=3),),
                "arousal_binary_split_label": datasets.Sequence(datasets.ClassLabel(names=["low", "high"], num_classes=2)),
                "flubber_frequency": datasets.Sequence(datasets.Value("float64")),
                "flubber_amplitude": datasets.Sequence(datasets.Value("float64")),
            }),
            standardize_input=True,
            expand_low_frequency_signals=False,
            majority_vote_labels=False,
            binary_split_labels=True,
            example_length_in_seconds=12,
            sliding_window=1,
        ),
    ]

    DEFAULT_CONFIG_NAME = "brain_heart_emotion"
    # Set MNE log level to WARNING to avoid cluttering the output
    mne.set_log_level("ERROR")

    def _info(self):
        """Returns the dataset info."""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
    @staticmethod
    def _calculate_tertiary_split(data, column_name):
        """Calculate tertiles and create a tertiary split label for a given column."""
        tertile1 = data[column_name].quantile(1/3)
        tertile2 = data[column_name].quantile(2/3)
        if tertile1 == tertile2:
            print(
                f"Warning: Tertile values for {column_name} are equal: {tertile1}")
        data[f"{column_name}_tertiary_split_label"] = pd.cut(
            data[column_name],
            bins=[-np.inf, tertile1, tertile2, np.inf],
            right=True,
            labels=["low", "medium", "high"]
        )
    @staticmethod
    def _binary_split(data, column_name):
        """Calculate binary split for a given column."""
        median = data[column_name].median()
        data[f"{column_name}_binary_split_label_"] = (data[column_name] > median).astype(int)
        data[f"{column_name}_binary_split_label"] = ('low')
        data.loc[data[f"{column_name}_binary_split_label_"] == 1, f"{column_name}_binary_split_label"] = 'high'
        return data
    
    def _expand_low_frequency_signal(signal, sampling_rate=200):
        """Expand 1Hz signal to new higher frequency through repetition. Defaults to 200Hz."""
        upsampled= np.repeat(signal, sampling_rate)
        return upsampled

    @staticmethod
    def _generate_examples_functional(
        subj_paths,
        example_length_in_seconds,
        features,
        tertiary_split_labels,
        binary_split_labels,
        sliding_window,
        standardize_input,
        expand_low_frequency_signals,
        majority_vote_labels,
    ):
        """Yields examples. Functional variant that takes config as arguments."""
        # Extract config parameters
        length_in_seconds = example_length_in_seconds
        read_ecg = "ecg_data" in features
        create_rpeak_labels = "is_r_peak" in features

        assert not (create_rpeak_labels and not read_ecg), \
            "create_rpeak_labels is enabled, but 'ecg_data' is not in the features. Please add it to the features."
        # assert misconfiguration of tertiary_split_labels and features
        assert not (tertiary_split_labels and "arousal_tertiary_split_label" not in features), \
            "Tertiary split labels are enabled, but 'arousal_tertiary_split_label' is not in the features. Please add it to the features."
        assert not (tertiary_split_labels and "valence_tertiary_split_label" not in features), \
            "Tertiary split labels are enabled, but 'valence_tertiary_split_label' is not in the features. Please add it to the features."

        for subj_path in subj_paths:
            # Assume that a Subject directory path is passed

            # Extract the subject ID from the path
            subject = os.path.basename(subj_path)
            # Load filtered data before ICA
            # Example filepath in a subject folder: sub-001/eeg/sub-001_task-AVR_eeg_preprocessed_filtered_0.1-45_before_ica.fif
            filtered_data_path = os.path.join(
                subj_path, "eeg", f"{subject}_task-AVR_eeg_preprocessed_filtered_0.1-45_before_ica.fif")
            # Load the filtered data
            filtered_data = mne.io.read_raw_fif(
                filtered_data_path, preload=True)
            # Load the emotional arousal recording file
            # Example filepath in a subject folder: sub-001/beh/sub-001_task-AVR_beh_preprocessed.tsv
            emotional_arousal_path = os.path.join(
                subj_path, "beh", f"{subject}_task-AVR_beh_preprocessed.tsv")

            ecg_data_path = os.path.join(
                subj_path, "eeg", f"{subject}_task-AVR_physio_ecg_preprocessed_scaled_manually-cleaned.tsv")

            # Processing of the EEG data
            # Re-reference the EEG recording to its mean
            filtered_data.set_eeg_reference(
                ref_channels='average', projection=False)
            # Select the EEG channels
            filtered_data.pick("eeg")
            # Cutoff at the last completed second. Data is sampled at 500Hz.
            # Get the number of complete seconds in the recording
            n_complete_seconds = int(
                filtered_data.n_times / filtered_data.info['sfreq'])
            # Crop the data to the last complete second
            filtered_data.crop(tmax=n_complete_seconds - 1)
            channels = filtered_data.ch_names
            filtered_data.resample(200, npad="auto")

            # Processing of the emotional arousal data
            emotional_arousal_data = pd.read_csv(
                emotional_arousal_path, sep="\t")

            if tertiary_split_labels or binary_split_labels:
                # interpolate with nearest neighbor (we know of 2 nan entries in the data)
                emotional_arousal_data = emotional_arousal_data.bfill()
                # Calculate tertiles and tertiary split labels for valence and arousal

                for column in ["valence", "arousal"]:
                    BrainHeartEmotion._calculate_tertiary_split(emotional_arousal_data, column)
                    BrainHeartEmotion._binary_split(emotional_arousal_data, column)
                # Crop the emotional arousal data to match the number of complete seconds
                emotional_arousal_data = emotional_arousal_data.iloc[:n_complete_seconds]
            eeg_data = filtered_data.get_data()
            LOGGER.info(
                f"Loaded EEG data for subject {subject} with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} samples.")
            LOGGER.info(
                f"Loaded emotional arousal data for subject {subject} with {emotional_arousal_data.shape[0]} rows.")
            # Ensure that the EEG data and emotional arousal data have the same number of samples

            # ECG-Related operations
            if "ecg_data" in features:
                # Load the ECG data
                ecg_data = pd.read_csv(ecg_data_path, sep="\t")
                r_peaks = ecg_data["R-peaks"].dropna()
                if len(r_peaks) == 0:
                    LOGGER.error(
                        f"No R-peaks found in ECG data for subject {subject}.")
                    raise ValueError(
                        f"No R-peaks found in ECG data for subject {subject}. Please check the ECG data file: {ecg_data_path}.")
                
                # Downsample 500Hz ecg signal to 200Hz
                ecg_data.set_index("timestamp", inplace=True)
                ecg_data.index = pd.to_timedelta(ecg_data.index, unit='s')
                ecg_data = ecg_data.resample(
                    "5ms").mean().interpolate(method="linear")
                
                LOGGER.info(
                    f"Downsampled ECG data for subject {subject} has {ecg_data.shape[0]} rows.")

                if "is_r_peak" in features:
                    downsampled_r_peaks = (r_peaks * (200 / 500)).astype(int)
                    # create new column with "is_r_peak" boolean
                    ecg_data["is_r_peak"] = 0
                    # Weird list operation is because of future deprecation of chained indexing
                    # Assume col -1 is rpeaks
                    ecg_data.iloc[downsampled_r_peaks, -1] = 1
                    # expand r peaks around the peak for 20ms
                    ecg_data["is_r_peak"] = ecg_data["is_r_peak"].rolling(
                        window="20ms", center=True).sum().astype(bool)
                    
            if sliding_window is not None:
                step_size_seconds = sliding_window
            else:
                step_size_seconds = example_length_in_seconds
            final_index = eeg_data.shape[1] // 200 - (length_in_seconds -1)

            for index in range(0, final_index, step_size_seconds):
                segment = eeg_data[:, index*200:(index+length_in_seconds)*200]
                valence = emotional_arousal_data["valence"].values[index:index+length_in_seconds]
                arousal = emotional_arousal_data["arousal"].values[index:index+length_in_seconds]
                flubber_frequency = emotional_arousal_data[
                    "flubber_frequency"].values[index:index+length_in_seconds]
                flubber_amplitude = emotional_arousal_data[
                    "flubber_amplitude"].values[index:index+length_in_seconds]

                if standardize_input:
                    # Standardize the EEG data per sample as in LaBraM paper
                    segment = (segment - np.mean(segment, axis=1, keepdims=True)
                               ) / np.std(segment, axis=1, keepdims=True)

                if expand_low_frequency_signals:
                    # Expand low frequency signals such as emotional arousal
                    valence = BrainHeartEmotion._expand_low_frequency_signal(valence)
                    arousal = BrainHeartEmotion._expand_low_frequency_signal(arousal)
                    flubber_frequency = BrainHeartEmotion._expand_low_frequency_signal(flubber_frequency)
                    flubber_amplitude = BrainHeartEmotion._expand_low_frequency_signal(flubber_amplitude)

                if tertiary_split_labels:
                    arousal_tertiary_split_label = emotional_arousal_data[
                        "arousal_tertiary_split_label"].values[index:index+length_in_seconds]
                    valence_tertiary_split_label = emotional_arousal_data[
                        "valence_tertiary_split_label"].values[index:index+length_in_seconds]
                    if majority_vote_labels:
                        # Majority vote valence and arousal labels
                        valence = int(np.bincount(valence).argmax())
                        arousal = int(np.bincount(arousal).argmax())
                if binary_split_labels:
                    arousal_binary_split_label = emotional_arousal_data[
                        "arousal_binary_split_label"].values[index:index+length_in_seconds]
                    valence_binary_split_label = emotional_arousal_data[
                        "valence_binary_split_label"].values[index:index+length_in_seconds]
                    if majority_vote_labels:
                        # Majority vote valence and arousal labels
                        valence_binary_split_label = int(
                            np.bincount(valence_binary_split_label).argmax())
                        arousal_binary_split_label = int(
                            np.bincount(arousal_binary_split_label).argmax())
                
                    
                example = {
                    "subject": subject,
                    "eeg_channels": channels,
                    "eeg_data": segment,
                    "valence": valence,
                    "arousal": arousal,
                    "flubber_frequency": flubber_frequency,
                    "flubber_amplitude": flubber_amplitude
                }

                if tertiary_split_labels:
                    example["valence_tertiary_split_label"] = valence_tertiary_split_label
                    example["arousal_tertiary_split_label"] = arousal_tertiary_split_label
                if binary_split_labels:
                    example["arousal_binary_split_label"] = arousal_binary_split_label
                    example["valence_binary_split_label"] = valence_binary_split_label
                
                if read_ecg:
                    ecg_segment = ecg_data.iloc[
                        index*200 :(index + length_in_seconds) * 200
                        ]
                    example["ecg_data"] = ecg_segment["ECG"].values
                    if create_rpeak_labels:
                        rpeak_data = ecg_segment["is_r_peak"].values
                        if rpeak_data.shape[0] != length_in_seconds * 200:
                            LOGGER.warning(f'R-peak segment for subject {subject} at index {index} has only {rpeak_data.shape[0]} rows. This might be due to the sliding window size or the length of the ECG data.')
                        example["is_r_peak"] = rpeak_data

                yield subject+str(index), example
    @staticmethod
    def reproducable_kfold_separation(to_split, n_folds, fold_index, seed, val_size=4):
        '''
        Reproducible K-Fold separation for cross-validation.
        Given i, n_folds and seed, return train, validation and test splits where test is the
        i-th fold, and validation is a random subset of the training data.
        Args:
            to_split: The data to split.
            n_folds: The number of folds.
            fold_index: The index of the fold to use as the test set.
            seed: The random seed for reproducibility.
            val_size: The size of the validation set.
        Returns:
            A tuple of (train, validation, test) splits.
        '''
        if not isinstance(to_split, (list, np.ndarray)):
            LOGGER.error("to_split must be a list or numpy array.")
            raise TypeError("to_split must be a list or numpy array.")
        if not isinstance(n_folds, int) or n_folds < 2:
            LOGGER.error("n_folds must be an integer >= 2.")
            raise ValueError("n_folds must be an integer >= 2.")
        if not isinstance(fold_index, int) or not (0 <= fold_index < n_folds):
            LOGGER.error(f"fold_index {fold_index} is out of range for n_folds={n_folds}.")
            raise ValueError("fold_index must be in range [0, n_folds-1].")
        if not isinstance(seed, (int, np.integer)):
            LOGGER.warning("Seed should be an integer for reproducibility.")
        if val_size >= len(to_split) - len(to_split) // n_folds:
            val_size = max(1, len(to_split) - len(to_split) // n_folds - 1)
            LOGGER.warning(f"val_size is too large, reducing to {val_size} to fit available training data.")
        
        rng = np.random.default_rng(seed)
        rng.shuffle(to_split)
        folds = np.array_split(to_split, n_folds)
        LOGGER.debug(f"Folds created: {len(folds)}. Fold shapes: {[fold.shape for fold in folds]}")
        test = folds.pop(fold_index)
        train = np.concatenate(folds)
        val = rng.choice(train, size=val_size, replace=False)
        train = np.setdiff1d(train, val, assume_unique=True) # Make sure this doesn't force recaching
        return train, val, test
    
    def _generate_examples(self, subj_path):
        """Yields examples."""
        yield from self._generate_examples_functional(
            subj_paths=subj_path,
            example_length_in_seconds=self.config.example_length_in_seconds,
            features=self.config.features,
            tertiary_split_labels=self.config.tertiary_split_labels,
            binary_split_labels=self.config.binary_split_labels,
            sliding_window=self.config.sliding_window,
            standardize_input=self.config.standardize_input,
            expand_low_frequency_signals=self.config.expand_low_frequency_signals,
            majority_vote_labels=self.config.majority_vote_labels,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        # Download the dataset
        # data_dir = dl_manager.download_and_extract(self.config.data_dir)
        # Get the list of subject directories
        data_dir = self.config.data_dir

        with open(os.path.join(self.base_path, "split_definitions.json"), "r") as f:
            split_definition = json.load(f)
        if 'exclude' in split_definition:
            exclude_subjects = split_definition["exclude"]
            LOGGER.info(f"Excluding subjects: {exclude_subjects}")
            # Assert that the excluded subjects are not in the train, test or validation sets
            for split in ["train", "test", "validation"]:
                for subj in split_definition[split]:
                    assert subj not in exclude_subjects, f"Subject {subj} is in the excluded list but also in the {split} set."
        train_subjects = split_definition["train"]
        test_subjects = split_definition["test"]
        val_subjects = split_definition["validation"]

        if self.config.kfold_k is not None:
            LOGGER.info(f"Using k-fold crossval dataset with {self.config.kfold_k} folds, index {self.config.kfold_index}, seed {self.config.kfold_seed}.")
            train_subjects, val_subjects, test_subjects = BrainHeartEmotion.reproducable_kfold_separation(
                to_split=train_subjects + val_subjects + test_subjects,
                n_folds=self.config.kfold_k,
                fold_index=self.config.kfold_index,
                seed=self.config.kfold_seed,
                val_size=4
            )

        def pathify(x): return os.path.join(data_dir, f'sub-{x}')
        train_subjects = [pathify(subj) for subj in train_subjects]
        test_subjects = [pathify(subj) for subj in test_subjects]
        val_subjects = [pathify(subj) for subj in val_subjects]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"subj_path": train_subjects},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"subj_path": val_subjects},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"subj_path": test_subjects},
            ),
        ]

if __name__ == "__main__":
    # Example usage

    dataset = BrainHeartEmotion(config_name="brain_heart_rpeak", data_dir="data/brain_heart_emotion")
    gen = dataset._generate_examples(["data/brain_heart_emotion"])
    for i in gen:
        print("loaded one")