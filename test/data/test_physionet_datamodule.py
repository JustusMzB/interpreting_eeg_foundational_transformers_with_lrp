import os
import random
import time

import pytest
from unittest.mock import patch, MagicMock, call
from torch.utils.data import DataLoader
import torch
from src.data.physionet_datamodule import PhysionetDatamodule

# Ground-truth splits for all 109 EEGMMIDB subjects under seed 42.
# If this test fails, the datamodule's split logic has changed — do NOT
# silently update these lists without re-running all published experiments.
_EXPECTED_TRAIN = sorted([
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
])
_EXPECTED_VAL = sorted([
    'S002', 'S003', 'S022', 'S024', 'S030', 'S038', 'S053', 'S064', 'S088',
    'S098', 'S107',
])
_EXPECTED_TEST = sorted([
    'S015', 'S021', 'S052', 'S061', 'S072', 'S075', 'S083', 'S087', 'S093',
    'S100', 'S103',
])


class TestPhysionetDatamodule:

    # ==========================================
    # 1. Initialization Tests
    # ==========================================
    def test_init_default_classes(self, tmp_path):
        """Test if the datamodule initializes with default classes correctly."""
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path))
        assert dm.data_dir == str(tmp_path)
        assert dm.batch_size == 32
        assert len(dm.classes) == 4
        assert dm.class_to_idx["left_hand"] == 0
        assert dm.hf_dataset is None

    def test_init_custom_classes(self, tmp_path):
        """Test if it correctly maps a subset of classes."""
        custom_classes = ["left_hand", "both_feet"]
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path), classes=custom_classes)
        assert len(dm.classes) == 2
        assert dm.class_to_idx["left_hand"] == 0
        assert dm.class_to_idx["both_feet"] == 1

    def test_init_invalid_class_raises_error(self, tmp_path):
        """Test if passing an invalid class throws a ValueError."""
        with pytest.raises(ValueError, match="invalid"):
            PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path), classes=["left_hand", "invalid_class"])

    # ==========================================
    # 2. prepare_data Tests (Mocked)
    # ==========================================
    @patch("src.data.physionet_datamodule.requests.get")
    @patch("src.data.physionet_datamodule.zipfile.ZipFile")
    def test_prepare_data_downloads_when_missing(self, mock_zip, mock_get, tmp_path):
        """Test if prepare_data triggers download and extraction when data is missing."""
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path))

        # Setup mock for requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake_chunk_1", b"fake_chunk_2"]
        mock_get.return_value = mock_response

        dm.prepare_data()

        # Verify download was requested
        mock_get.assert_called()
        # Verify zip extraction was called
        mock_zip.assert_called_once_with(dm.zip_path, 'r')

    @patch("src.data.physionet_datamodule.requests.get")
    def test_prepare_data_skips_if_already_extracted(self, mock_get, tmp_path):
        """Test if prepare_data respects idempotency (skips if folder exists)."""
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path))

        # Manually create the target directory to simulate already extracted data
        os.makedirs(dm.extracted_path)

        dm.prepare_data()

        # Requests should NOT be called
        mock_get.assert_not_called()

    # ==========================================
    # 3. Dataloader Tests
    # ==========================================
    def test_dataloaders_return_correct_types(self, tmp_path):
        """
        Test if the dataloader methods return PyTorch DataLoaders with the right settings.
        We mock the hf_dataset to avoid running the heavy MNE setup generator.
        """
        dm = PhysionetDatamodule(dataset_path= None, data_dir=str(tmp_path), batch_size=16, num_workers=2)

        # Inject a mock Hugging Face DatasetDict
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100  # Mock length

        dm.hf_dataset = {
            "train": mock_dataset,
            "validation": mock_dataset,
            "test": mock_dataset
        }

        # Test Train Dataloader
        train_loader = dm.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 16
        assert train_loader.num_workers == 2
        # Train should be shuffled
        assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler) if hasattr(train_loader,
                                                                                           'sampler') else True

        # Test Val Dataloader
        val_loader = dm.val_dataloader()
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 16

        # Test Test Dataloader
        test_loader = dm.test_dataloader()
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 16

    # 5. Network Tests
    # ==========================================
    @pytest.mark.network
    def test_dataset_url_is_valid_zip(self, tmp_path):
        """
        Test if the hardcoded URL actually leads to a zip file by making a real network request.
        Reads only the first few bytes to check the ZIP file signature without downloading the whole file.
        """
        import requests
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path))

        response = requests.get(dm.dataset_url, stream=True)
        assert response.status_code == 200, f"Failed to reach URL. Status code: {response.status_code}"

        magic_bytes = next(response.iter_content(chunk_size=4))
        assert magic_bytes.startswith(b"PK"), f"The URL does not point to a valid ZIP file. Got bytes: {magic_bytes}"
        response.close()

    # ==========================================
    # 6. Full End-to-End Test (Skipped by default)
    # ==========================================
    @pytest.mark.long
    def test_full_e2e_download_and_process(self):
        """
        WARNING: This test actually downloads the 1.2GB zip file, extracts it,
        and processes the EDF files. It is skipped by default to prevent CI timeouts.
        It uses tempfile.TemporaryDirectory to guarantee immediate disk cleanup.
        """
        import tempfile
        import datasets

        with tempfile.TemporaryDirectory() as temp_dir:
            original_hf_cache = datasets.config.HF_DATASETS_CACHE
            datasets.config.HF_DATASETS_CACHE = os.path.join(temp_dir, "hf_cache")

            try:
                dm = PhysionetDatamodule(
                    dataset_path=None,
                    data_dir=temp_dir,
                    classes=["left_hand", "right_hand"],
                    batch_size=2,
                    num_workers=0
                )

                dm.prepare_data()

                assert os.path.exists(dm.extracted_path)
                assert os.path.exists(dm.zip_path)

                dm.setup(stage="fit")

                assert dm.hf_dataset is not None
                assert "train" in dm.hf_dataset

                train_loader = dm.train_dataloader()
                batch = next(iter(train_loader))

                # Check new expected column mappings
                assert "signal" in batch
                assert "label" in batch
                assert "eeg_channel_indices" in batch
                assert batch["signal"].shape[0] == 2
            finally:
                datasets.config.HF_DATASETS_CACHE = original_hf_cache

    # Define the skip condition
    # This will evaluate to True (and thus skip the test) if the flag is NOT '1'
    skip_slow_e2e = pytest.mark.skipif(
        os.environ.get("RUN_SLOW_E2E_TEST") != "1",
        reason="Slow E2E test skipped. Set RUN_SLOW_E2E_TEST=1 to run."
    )
    @staticmethod
    def physionet_found_locally():
        """Helper method to check if the Physionet dataset is found locally."""
        physionet_found = False
        for candidate in ['~/Projects/datasets/physiomi', '~/high_throughput/physiomi', '/data/physiomi']:
            if os.path.exists(os.path.expanduser(candidate)):
                print(f"Found local physiomi dataset at {os.path.expanduser(candidate)}")
                physionet_found = True
                break
        return physionet_found
    
    skip_if_no_physionet = pytest.mark.skipif(
        not physionet_found_locally(),
        reason="PhysioNet dataset not found locally. Download would take too long."
    )
    @staticmethod
    @pytest.fixture
    def datamodule():
        """Fixture to initialize and set up the DataModule."""
        # Initialize with your specific parameters
        # try to find a physiomi dataset locally
        data_path = None
        for candidate in ['~/Projects/datasets/physiomi', '~/high_throughput/physiomi', '/data/physiomi']:
            if os.path.exists(os.path.expanduser(candidate)):
                print(f"Using local physiomi dataset at {os.path.expanduser(candidate)}")
                data_path = os.path.expanduser(candidate)
        if data_path is None:
            pytest.fail("Physionet dataset not found. Consider setting skip flag fort this test under these circumstances.")

        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=data_path,
            classes=["left_hand", "right_hand"],
            num_workers=10
        )
        dm.prepare_data()
        dm.setup(stage='fit')
        return dm

    @pytest.mark.long
    @skip_if_no_physionet
    def test_dataloader_batch_yield_time(self, datamodule):
        """Tests how quickly the dataloader can yield randomized batches."""
        dataloader = datamodule.train_dataloader()
        dl_iter = iter(dataloader)

        # 1. Warm-up
        try:
            _ = next(dl_iter)
        except StopIteration:
            pytest.skip("Dataloader is empty.")

        # 2. Benchmark
        num_batches_to_test = 10
        batches_loaded = 0

        start_time = time.perf_counter()
        for _ in range(num_batches_to_test):
            try:
                _ = next(dl_iter)
                batches_loaded += 1
            except StopIteration:
                break

        end_time = time.perf_counter()

        assert batches_loaded > 0, "Not enough batches to test."

        avg_time_per_batch = (end_time - start_time) / batches_loaded

        # 3. Assert against a threshold
        max_allowed_time = 0.05

        assert avg_time_per_batch < max_allowed_time, (
            f"Dataloader is too slow! "
            f"Averaged {avg_time_per_batch:.4f}s per batch. "
            f"Threshold is {max_allowed_time}s."
        )

    @pytest.mark.long
    @skip_if_no_physionet
    def test_dataset_random_access_time(self, datamodule):
        """Tests pure random access speed on the underlying Dataset (__getitem__)."""
        dataset = datamodule.train_dataloader().dataset
        dataset_size = len(dataset)

        assert dataset_size > 0, "Dataset is empty."

        num_samples_to_test = min(100, dataset_size)
        random_indices = random.sample(range(dataset_size), num_samples_to_test)

        # 1. Warm-up
        _ = dataset[0]

        # 2. Benchmark
        start_time = time.perf_counter()
        for idx in random_indices:
            _ = dataset[idx]
        end_time = time.perf_counter()

        avg_time_per_item = (end_time - start_time) / num_samples_to_test

        # 3. Assert against a threshold
        max_allowed_time_per_item = 0.005

        assert avg_time_per_item < max_allowed_time_per_item, (
            f"Dataset __getitem__ random access is too slow! "
            f"Averaged {avg_time_per_item:.5f}s per item. "
            f"Threshold is {max_allowed_time_per_item}s."
        )

    requires_cuda = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available. Skipping GPU transfer test."
    )

    @pytest.mark.long
    @requires_cuda
    @skip_if_no_physionet
    def test_dataloader_batch_to_device_time(self, datamodule):
        """Tests the time to yield a batch and transfer it to the GPU using Lightning."""
        dataloader = datamodule.train_dataloader()
        dl_iter = iter(dataloader)
        
        # Automatically grab the default CUDA device assigned by your srun environment
        device = torch.device('cuda')

        # 1. Warm-up dataloader and CUDA
        try:
            batch = next(dl_iter)
        except StopIteration:
            pytest.skip("Dataloader is empty.")
        
        # Use Lightning's native transfer hook (dataloader_idx is typically 0 for single dataloaders)
        _ = datamodule.transfer_batch_to_device(batch, device, dataloader_idx=0)
        torch.cuda.synchronize()

        # 2. Benchmark
        num_batches_to_test = 10
        batches_loaded = 0
        
        start_time = time.perf_counter()
        for _ in range(num_batches_to_test):
            try:
                batch = next(dl_iter)
                _ = datamodule.transfer_batch_to_device(batch, device, dataloader_idx=0)
                batches_loaded += 1
            except StopIteration:
                break
                
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        assert batches_loaded > 0, "Not enough batches to test."
        
        avg_time_per_batch = (end_time - start_time) / batches_loaded
        
        # 3. Assert against a threshold
        max_allowed_time = 0.06 
        
        assert avg_time_per_batch < max_allowed_time, (
            f"Dataloader + GPU transfer is too slow! "
            f"Averaged {avg_time_per_batch:.4f}s per batch. "
            f"Threshold is {max_allowed_time}s."
        )


class TestPhysionetSubjectSplits:
    """
    Regression guard for the seed-42 subject split.

    If any of these assertions fail it means the splitting logic in
    PhysionetDatamodule.setup() has changed.  Do NOT silently update the
    expected lists — all published experiments must be re-evaluated first.
    """

    @pytest.fixture()
    def subject_dirs(self, tmp_path):
        """Creates all 109 EEGMMIDB subject directories (no EDF files needed)."""
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        for i in range(1, 110):
            (files_dir / f"S{i:03d}").mkdir()
        return tmp_path

    def _make_mock_dataset(self):
        mock_ds = MagicMock()
        mock_ds.column_names = [
            "subject_id", "run_file", "event_label", "event_name",
            "eeg_channels", "eeg_signal",
        ]
        mock_ds.__getitem__ = MagicMock(
            return_value={"eeg_channels": ["Fc5.", "Fc3."]}
        )
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.rename_columns = MagicMock(return_value=mock_ds)
        mock_ds.set_format = MagicMock()
        return mock_ds

    def test_subject_splits_match_expected(self, subject_dirs):
        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(subject_dirs),
            classes=["left_hand", "right_hand"],
            num_workers=0,
        )

        captured: dict[str, list[str]] = {}

        mock_ds = self._make_mock_dataset()

        def mock_from_generator(fn, features=None, gen_kwargs=None, **kwargs):
            captured[id(gen_kwargs["subjects"])] = list(gen_kwargs["subjects"])
            return mock_ds

        mock_hf = MagicMock()
        mock_hf.__getitem__.return_value = mock_ds
        mock_hf.map = MagicMock(return_value=mock_hf)
        mock_hf.rename_columns = MagicMock(return_value=mock_hf)
        mock_hf.set_format = MagicMock()

        with (
            patch("src.data.physionet_datamodule.Dataset.from_generator", side_effect=mock_from_generator),
            patch("src.data.physionet_datamodule.DatasetDict", return_value=mock_hf),
            patch("src.data.physionet_datamodule.get_input_chans", return_value=[0, 1]),
            patch("src.data.physionet_datamodule.adapt_channel_names_to_mne", return_value=["Fc5.", "Fc3."]),
        ):
            dm.setup()

        assert len(captured) == 3, "Expected exactly 3 dataset splits to be created"

        # splits dict is built in insertion order: train, validation, test
        train_subjects, val_subjects, test_subjects = [
            sorted(v) for v in captured.values()
        ]

        assert train_subjects == _EXPECTED_TRAIN, (
            "Train split changed! Re-evaluate all published experiments before updating."
        )
        assert val_subjects == _EXPECTED_VAL, (
            "Validation split changed! Re-evaluate all published experiments before updating."
        )
        assert test_subjects == _EXPECTED_TEST, (
            "Test split changed! Re-evaluate all published experiments before updating."
        )


class TestPhysionetMiniMode:
    """Tests for the mini dataset mode (mini_n_train / mini_n_val / mini_n_test)."""

    def test_is_mini_flag_off_by_default(self, tmp_path):
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path))
        assert dm._is_mini is False

    def test_is_mini_flag_set_when_any_param_provided(self, tmp_path):
        dm = PhysionetDatamodule(dataset_path=None, data_dir=str(tmp_path), mini_n_train=3)
        assert dm._is_mini is True

    def test_mini_prepare_data_creates_all_109_dirs(self, tmp_path):
        """Mini prepare_data must create all 109 subject dirs to preserve split assignments."""
        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            mini_n_train=2, mini_n_val=1, mini_n_test=1,
        )

        with patch.object(dm, "_download_mini_physionet", wraps=dm._download_mini_physionet) as spy, \
             patch("src.data.physionet_datamodule.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"fake_edf_data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dm.prepare_data()

        files_dir = os.path.join(str(tmp_path), "files")
        created_dirs = sorted(
            d for d in os.listdir(files_dir)
            if os.path.isdir(os.path.join(files_dir, d))
        )
        assert len(created_dirs) == 109, (
            f"Expected 109 subject dirs; got {len(created_dirs)}. "
            "All dirs must exist for the split logic to work correctly."
        )

    def test_mini_prepare_data_only_downloads_selected_subjects(self, tmp_path):
        """Only selected subjects should have EDF files; others stay empty."""
        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            mini_n_train=2, mini_n_val=1, mini_n_test=1,
        )

        with patch("src.data.physionet_datamodule.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"fake_edf_data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dm.prepare_data()

        files_dir = os.path.join(str(tmp_path), "files")
        dirs_with_files = [
            d for d in os.listdir(files_dir)
            if os.listdir(os.path.join(files_dir, d))
        ]
        # 2 train + 1 val + 1 test = 4 subjects with data
        assert len(dirs_with_files) == 4

    def test_mini_selected_subjects_are_from_correct_splits(self, tmp_path):
        """Selected subjects must belong to the expected split lists — not cross-contaminated."""
        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            mini_n_train=2, mini_n_val=1, mini_n_test=1,
        )

        with patch("src.data.physionet_datamodule.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"fake_edf_data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dm.prepare_data()

        files_dir = os.path.join(str(tmp_path), "files")
        subjects_with_data = {
            d for d in os.listdir(files_dir)
            if os.listdir(os.path.join(files_dir, d))
        }

        train_overlap = subjects_with_data & set(_EXPECTED_TRAIN)
        val_overlap   = subjects_with_data & set(_EXPECTED_VAL)
        test_overlap  = subjects_with_data & set(_EXPECTED_TEST)

        assert len(train_overlap) == 2, f"Expected 2 train subjects; got {train_overlap}"
        assert len(val_overlap)   == 1, f"Expected 1 val subject; got {val_overlap}"
        assert len(test_overlap)  == 1, f"Expected 1 test subject; got {test_overlap}"

    def test_mini_excludes_faulty_subjects(self, tmp_path):
        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            mini_n_train=10, mini_n_val=5, mini_n_test=5,
        )

        with patch("src.data.physionet_datamodule.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"fake_edf_data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dm.prepare_data()

        files_dir = os.path.join(str(tmp_path), "files")
        subjects_with_data = {
            d for d in os.listdir(files_dir)
            if os.listdir(os.path.join(files_dir, d))
        }
        faulty = set(dm.faulty_subjects)
        assert not subjects_with_data & faulty, (
            f"Faulty subjects must never be selected for download: {subjects_with_data & faulty}"
        )

    @pytest.mark.network
    @pytest.mark.long
    def test_mini_real_download_and_setup(self, tmp_path):
        """Downloads a minimal real subset (1 train / 1 val / 1 test) and runs setup().

        Verifies that:
        - All 109 subject dirs are present after prepare_data().
        - Exactly 3 subjects have EDF files (one per split).
        - Those subjects belong to the correct expected splits.
        - setup() produces non-empty train, validation, and test datasets.
        - setup() emits a UserWarning about mini mode.

        Skipped by default; run with RUN_LONG_TESTS=1.
        """
        import datasets as hf_datasets

        hf_cache = tmp_path / "hf_cache"
        hf_cache.mkdir()

        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            classes=["left_hand", "right_hand"],
            num_workers=0,
            mini_n_train=1, mini_n_val=1, mini_n_test=1,
        )

        dm.prepare_data()

        files_dir = os.path.join(str(tmp_path), "files")
        all_dirs = [d for d in os.listdir(files_dir) if os.path.isdir(os.path.join(files_dir, d))]
        assert len(all_dirs) == 109

        subjects_with_data = {
            d for d in os.listdir(files_dir)
            if os.listdir(os.path.join(files_dir, d))
        }
        assert len(subjects_with_data) == 3
        assert subjects_with_data & set(_EXPECTED_TRAIN)
        assert subjects_with_data & set(_EXPECTED_VAL)
        assert subjects_with_data & set(_EXPECTED_TEST)

        original_cache = hf_datasets.config.HF_DATASETS_CACHE
        hf_datasets.config.HF_DATASETS_CACHE = str(hf_cache)
        try:
            with pytest.warns(UserWarning, match="MINI DATASET MODE ACTIVE"):
                dm.setup()
        finally:
            hf_datasets.config.HF_DATASETS_CACHE = original_cache

        assert dm.hf_dataset is not None
        assert len(dm.train_ds) > 0, "Mini train split should have at least one sample"
        assert len(dm.val_ds) > 0,   "Mini val split should have at least one sample"
        assert len(dm.test_ds) > 0,  "Mini test split should have at least one sample"

    def test_mini_setup_emits_warning(self, tmp_path):
        """setup() must emit a UserWarning when mini mode is active."""
        files_dir = os.path.join(str(tmp_path), "files")
        os.makedirs(files_dir)
        for i in range(1, 110):
            os.makedirs(os.path.join(files_dir, f"S{i:03d}"))

        dm = PhysionetDatamodule(
            dataset_path=None,
            data_dir=str(tmp_path),
            mini_n_train=2, mini_n_val=1, mini_n_test=1,
            num_workers=0,
        )

        mock_ds = MagicMock(
            column_names=[], map=MagicMock(return_value=MagicMock(rename_columns=MagicMock(return_value=MagicMock()))),
            __getitem__=MagicMock(return_value={"eeg_channels": []}),
            __len__=MagicMock(return_value=1)
        )
        mock_hf = MagicMock()
        mock_hf.__getitem__.return_value = mock_ds
        mock_hf.map = MagicMock(return_value=mock_hf)
        mock_hf.rename_columns = MagicMock(return_value=mock_hf)
        mock_hf.set_format = MagicMock()

        with patch("src.data.physionet_datamodule.Dataset.from_generator", return_value=mock_ds), \
             patch("src.data.physionet_datamodule.DatasetDict", return_value=mock_hf), \
             patch("src.data.physionet_datamodule.get_input_chans", return_value=[]), \
             patch("src.data.physionet_datamodule.adapt_channel_names_to_mne", return_value=[]), \
             pytest.warns(UserWarning, match="MINI DATASET MODE ACTIVE"):
            dm.setup()