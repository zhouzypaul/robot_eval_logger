"""Tests for all storage optimizations introduced in StorageConfig.

Covers: JPEG image compression, array stacking, lz4 pickle compression,
pickle protocol 5, async saving, batch HF uploads, EvalLogger.flush(),
and backward compatibility.

How to run (from the repository root)::

    # All tests under tests/ (see pyproject.toml [tool.pytest.ini_options])
    pytest

    # This file only, verbose
    pytest tests/test_storage_optimizations.py -v

    # Without coverage flags if pytest-cov is not installed
    python -m pytest tests/test_storage_optimizations.py -v -o addopts=

Shared fixtures (``rng``, ``sample_traj_data``) live in ``tests/conftest.py``.
"""
import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_eval_logger.storage.base_saver import StorageConfig
from robot_eval_logger.storage.local import LocalStorage
from robot_eval_logger.typing.traj_data import _LZ4_MAGIC, TrajData

# ---------------------------------------------------------------------------
# StorageConfig defaults & customization
# ---------------------------------------------------------------------------


class TestStorageConfig:
    def test_defaults_all_enabled(self):
        cfg = StorageConfig()
        assert cfg.compress_images is True
        assert cfg.stack_arrays is True
        assert cfg.compress_pickle is True
        assert cfg.use_highest_pickle_protocol is True
        assert cfg.async_saving is True

    def test_individual_toggle(self):
        cfg = StorageConfig(compress_images=False, async_saving=False)
        assert cfg.compress_images is False
        assert cfg.async_saving is False
        assert cfg.stack_arrays is True

    def test_image_quality_custom(self):
        cfg = StorageConfig(image_quality=50)
        assert cfg.image_quality == 50


# ---------------------------------------------------------------------------
# JPEG image compression
# ---------------------------------------------------------------------------


class TestImageCompression:
    def test_images_encoded_to_bytes(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_images=True)
        loaded = TrajData.load(path)
        for cam, img_list in loaded.obs.items():
            for img in img_list:
                assert isinstance(
                    img, bytes
                ), f"Expected bytes for {cam}, got {type(img)}"

    def test_compressed_file_smaller(self, sample_traj_data, tmp_path):
        raw_path = str(tmp_path / "raw.pkl")
        compressed_path = str(tmp_path / "compressed.pkl")
        sample_traj_data.save(raw_path, compress_images=False)
        sample_traj_data.save(compressed_path, compress_images=True)
        raw_size = os.path.getsize(raw_path)
        compressed_size = os.path.getsize(compressed_path)
        assert (
            compressed_size < raw_size
        ), f"Compressed ({compressed_size}) should be smaller than raw ({raw_size})"

    def test_decode_images_roundtrip(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_images=True, image_quality=95)
        loaded = TrajData.load(path)
        decoded = TrajData.decode_images(loaded.obs)
        for cam in sample_traj_data.obs:
            assert cam in decoded
            assert len(decoded[cam]) == len(sample_traj_data.obs[cam])
            for orig, dec in zip(sample_traj_data.obs[cam], decoded[cam]):
                assert isinstance(dec, np.ndarray)
                assert dec.shape == orig.shape
                assert dec.dtype == orig.dtype

    def test_decode_images_passthrough_ndarray(self, sample_traj_data):
        """decode_images should pass through arrays that aren't bytes."""
        result = TrajData.decode_images(sample_traj_data.obs)
        for cam in sample_traj_data.obs:
            for orig, dec in zip(sample_traj_data.obs[cam], result[cam]):
                np.testing.assert_array_equal(orig, dec)

    def test_no_compression_keeps_ndarray(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_images=False)
        loaded = TrajData.load(path)
        for cam, img_list in loaded.obs.items():
            for img in img_list:
                assert isinstance(img, np.ndarray)

    def test_lower_quality_smaller_file(self, sample_traj_data, tmp_path):
        path_q95 = str(tmp_path / "q95.pkl")
        path_q30 = str(tmp_path / "q30.pkl")
        sample_traj_data.save(path_q95, compress_images=True, image_quality=95)
        sample_traj_data.save(path_q30, compress_images=True, image_quality=30)
        assert os.path.getsize(path_q30) < os.path.getsize(path_q95)


# ---------------------------------------------------------------------------
# Array stacking
# ---------------------------------------------------------------------------


class TestArrayStacking:
    def test_arrays_stacked_to_2d(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, stack_arrays=True)
        loaded = TrajData.load(path)
        assert isinstance(loaded.action, np.ndarray)
        assert loaded.action.ndim == 2
        assert loaded.action.shape == (5, 7)

    def test_all_numeric_fields_stacked(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, stack_arrays=True)
        loaded = TrajData.load(path)
        for name in (
            "action",
            "joint_position",
            "joint_velocity",
            "end_effector_pose",
            "gripper",
            "joint_effort",
        ):
            val = getattr(loaded, name)
            assert isinstance(val, np.ndarray), f"{name} should be stacked ndarray"
            assert val.ndim == 2, f"{name} should be 2D after stacking"

    def test_no_stacking_keeps_lists(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, stack_arrays=False)
        loaded = TrajData.load(path)
        assert isinstance(loaded.action, list)
        assert len(loaded.action) == 5

    def test_original_data_unchanged(self, sample_traj_data, tmp_path):
        """save() with stack_arrays should not mutate the original TrajData."""
        path = str(tmp_path / "traj.pkl")
        assert isinstance(sample_traj_data.action, list)
        sample_traj_data.save(path, stack_arrays=True)
        assert isinstance(
            sample_traj_data.action, list
        ), "Original action should still be a list"


# ---------------------------------------------------------------------------
# lz4 pickle compression
# ---------------------------------------------------------------------------


class TestLz4Compression:
    def test_lz4_magic_header(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_pickle=True)
        with open(path, "rb") as f:
            header = f.read(4)
        assert header == _LZ4_MAGIC

    def test_no_lz4_no_magic(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_pickle=False)
        with open(path, "rb") as f:
            header = f.read(4)
        assert header != _LZ4_MAGIC

    def test_lz4_smaller_file(self, sample_traj_data, tmp_path):
        raw_path = str(tmp_path / "raw.pkl")
        lz4_path = str(tmp_path / "lz4.pkl")
        sample_traj_data.save(raw_path, compress_pickle=False)
        sample_traj_data.save(lz4_path, compress_pickle=True)
        assert os.path.getsize(lz4_path) < os.path.getsize(raw_path)

    def test_load_autodetects_lz4(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_pickle=True)
        loaded = TrajData.load(path)
        assert loaded.language_command == sample_traj_data.language_command
        assert loaded.success == sample_traj_data.success

    def test_load_handles_uncompressed(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path, compress_pickle=False)
        loaded = TrajData.load(path)
        assert loaded.language_command == sample_traj_data.language_command


# ---------------------------------------------------------------------------
# Pickle protocol
# ---------------------------------------------------------------------------


class TestPickleProtocol:
    def test_highest_protocol(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(
            path, use_highest_pickle_protocol=True, compress_pickle=False
        )
        with open(path, "rb") as f:
            proto_byte = f.read(2)
        used_protocol = proto_byte[1]
        assert used_protocol == pickle.HIGHEST_PROTOCOL

    def test_default_protocol(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(
            path, use_highest_pickle_protocol=False, compress_pickle=False
        )
        with open(path, "rb") as f:
            proto_byte = f.read(2)
        used_protocol = proto_byte[1]
        assert used_protocol == pickle.DEFAULT_PROTOCOL


# ---------------------------------------------------------------------------
# Combined optimizations — overall size reduction
# ---------------------------------------------------------------------------


class TestCombinedOptimizations:
    def test_all_on_much_smaller_than_all_off(self, sample_traj_data, tmp_path):
        off_path = str(tmp_path / "off.pkl")
        on_path = str(tmp_path / "on.pkl")
        sample_traj_data.save(off_path)
        sample_traj_data.save(
            on_path,
            compress_images=True,
            stack_arrays=True,
            compress_pickle=True,
            use_highest_pickle_protocol=True,
        )
        off_size = os.path.getsize(off_path)
        on_size = os.path.getsize(on_path)
        assert (
            on_size < off_size * 0.5
        ), f"All-on ({on_size}) should be <50% of all-off ({off_size})"

    def test_roundtrip_preserves_episode_fields(self, sample_traj_data, tmp_path):
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(
            path,
            compress_images=True,
            stack_arrays=True,
            compress_pickle=True,
            use_highest_pickle_protocol=True,
        )
        loaded = TrajData.load(path)
        assert loaded.language_command == "pick up the red block"
        assert loaded.success is True
        assert loaded.episode_length == 5
        assert loaded.policy_id == "test_policy"
        assert loaded.duration_seconds == 2.5


# ---------------------------------------------------------------------------
# LocalStorage with async saving
# ---------------------------------------------------------------------------


class TestAsyncSaving:
    def _make_storage(self, tmp_path, async_on):
        cfg = StorageConfig(
            compress_images=False,
            stack_arrays=False,
            compress_pickle=False,
            use_highest_pickle_protocol=False,
            async_saving=async_on,
        )
        storage = LocalStorage(str(tmp_path), config=cfg)
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()
        return storage

    def test_async_file_exists_after_flush(self, sample_traj_data, tmp_path):
        storage = self._make_storage(tmp_path, async_on=True)
        path = storage.save_episode(0, sample_traj_data)
        storage.flush()
        assert os.path.exists(path)

    def test_sync_file_exists_immediately(self, sample_traj_data, tmp_path):
        storage = self._make_storage(tmp_path, async_on=False)
        path = storage.save_episode(0, sample_traj_data)
        assert os.path.exists(path)

    def test_async_data_correct_after_flush(self, sample_traj_data, tmp_path):
        storage = self._make_storage(tmp_path, async_on=True)
        path = storage.save_episode(0, sample_traj_data)
        storage.flush()
        loaded = TrajData.load(path)
        assert loaded.language_command == sample_traj_data.language_command

    def test_async_multiple_episodes(self, sample_traj_data, tmp_path):
        storage = self._make_storage(tmp_path, async_on=True)
        paths = []
        for i in range(5):
            paths.append(storage.save_episode(i, sample_traj_data))
        storage.flush()
        for p in paths:
            assert os.path.exists(p)

    def test_flush_propagates_errors(self, tmp_path):
        cfg = StorageConfig(async_saving=True)
        storage = LocalStorage(str(tmp_path), config=cfg)
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()

        bad_traj = MagicMock()
        bad_traj.save.side_effect = IOError("disk full")

        storage.save_episode(0, bad_traj)
        with pytest.raises(IOError, match="disk full"):
            storage.flush()

    def test_config_flags_passed_to_traj_save(self, sample_traj_data, tmp_path):
        cfg = StorageConfig(
            compress_images=True,
            image_quality=75,
            stack_arrays=True,
            compress_pickle=True,
            use_highest_pickle_protocol=True,
            async_saving=False,
        )
        storage = LocalStorage(str(tmp_path), config=cfg)
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()

        path = storage.save_episode(0, sample_traj_data)
        loaded = TrajData.load(path)
        # Images should be JPEG bytes
        for img in loaded.obs["image_primary"]:
            assert isinstance(img, bytes)
        # Arrays should be stacked
        assert isinstance(loaded.action, np.ndarray)
        assert loaded.action.ndim == 2


# ---------------------------------------------------------------------------
# HuggingFace batch uploads (mocked)
# ---------------------------------------------------------------------------


class TestBatchHfUploads:
    @pytest.fixture
    def hf_storage(self, tmp_path):
        from robot_eval_logger.storage.hugging_face import HuggingFaceStorage

        cfg = StorageConfig(
            compress_images=False,
            stack_arrays=False,
            compress_pickle=False,
            async_saving=False,
            batch_hf_uploads=True,
        )
        storage = HuggingFaceStorage(
            storage_dir=str(tmp_path),
            repo_id="fake/repo",
            config=cfg,
        )
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()
        return storage

    @pytest.fixture
    def hf_storage_no_batch(self, tmp_path):
        from robot_eval_logger.storage.hugging_face import HuggingFaceStorage

        cfg = StorageConfig(
            compress_images=False,
            stack_arrays=False,
            compress_pickle=False,
            async_saving=False,
            batch_hf_uploads=False,
        )
        storage = HuggingFaceStorage(
            storage_dir=str(tmp_path),
            repo_id="fake/repo",
            config=cfg,
        )
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()
        return storage

    def test_episodes_queued_not_uploaded(self, hf_storage, sample_traj_data):
        with patch.object(hf_storage.api, "upload_file") as mock_upload:
            with patch.object(hf_storage.api, "create_commit"):
                hf_storage.save_episode(0, sample_traj_data)
                hf_storage.save_episode(1, sample_traj_data)
                mock_upload.assert_not_called()
                assert len(hf_storage._pending_hf_uploads) == 2

    def test_flush_calls_create_commit_once(self, hf_storage, sample_traj_data):
        with patch.object(hf_storage.api, "create_commit") as mock_commit:
            hf_storage.save_episode(0, sample_traj_data)
            hf_storage.save_episode(1, sample_traj_data)
            hf_storage.save_episode(2, sample_traj_data)
            hf_storage.flush()
            mock_commit.assert_called_once()
            args, kwargs = mock_commit.call_args
            assert len(kwargs["operations"]) == 3
            assert "3 episodes" in kwargs["commit_message"]

    def test_flush_clears_queue(self, hf_storage, sample_traj_data):
        with patch.object(hf_storage.api, "create_commit"):
            hf_storage.save_episode(0, sample_traj_data)
            hf_storage.flush()
            assert len(hf_storage._pending_hf_uploads) == 0

    def test_no_batch_uploads_immediately(self, hf_storage_no_batch, sample_traj_data):
        with patch.object(hf_storage_no_batch.api, "upload_file") as mock_upload:
            hf_storage_no_batch.save_episode(0, sample_traj_data)
            mock_upload.assert_called_once()

    def test_batch_fallback_on_commit_failure(self, hf_storage, sample_traj_data):
        with patch.object(
            hf_storage.api,
            "create_commit",
            side_effect=Exception("server error"),
        ):
            with patch.object(hf_storage.api, "upload_file") as mock_individual:
                hf_storage.save_episode(0, sample_traj_data)
                hf_storage.save_episode(1, sample_traj_data)
                hf_storage.flush()
                assert mock_individual.call_count == 2


# ---------------------------------------------------------------------------
# EvalLogger.flush() integration
# ---------------------------------------------------------------------------


class TestEvalLoggerFlush:
    def test_flush_delegates_to_data_saver(self):
        from robot_eval_logger.eval_logger import EvalLogger

        mock_saver = MagicMock()
        logger = EvalLogger(data_saver=mock_saver)
        logger.flush()
        mock_saver.flush.assert_called_once()

    def test_flush_noop_without_saver(self):
        from robot_eval_logger.eval_logger import EvalLogger

        logger = EvalLogger(data_saver=None)
        logger.flush()  # should not raise


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_traj_save_no_args(self, sample_traj_data, tmp_path):
        """Default TrajData.save() still works like the original."""
        path = str(tmp_path / "traj.pkl")
        sample_traj_data.save(path)
        loaded = TrajData.load(path)
        assert loaded.language_command == sample_traj_data.language_command
        # Images remain as ndarrays (no JPEG encoding by default)
        for img in loaded.obs["image_primary"]:
            assert isinstance(img, np.ndarray)
        # Action remains as list (no stacking by default)
        assert isinstance(loaded.action, list)

    def test_local_storage_no_config(self, sample_traj_data, tmp_path):
        """LocalStorage(storage_dir) without config still works."""
        storage = LocalStorage(str(tmp_path))
        storage.make_eval_id_and_timestamp("widowx", "test_eval")
        storage.make_save_dir()
        path = storage.save_episode(0, sample_traj_data)
        storage.flush()
        assert os.path.exists(path)

    def test_load_old_format_pickle(self, sample_traj_data, tmp_path):
        """Files saved with plain pickle.dump can still be loaded."""
        path = str(tmp_path / "old.pkl")
        with open(path, "wb") as f:
            pickle.dump(sample_traj_data, f)
        loaded = TrajData.load(path)
        assert loaded.language_command == sample_traj_data.language_command
        assert loaded.success is True
