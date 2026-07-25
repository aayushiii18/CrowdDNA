"""Tests for the model artifact downloader."""

import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from crowdflow_dna.downloader import (
    _CANONICAL_MODEL_URL,
    resolve_model_path,
)


@pytest.fixture
def mock_cache_path(tmp_path):
    """Override the cache path to point to a temporary directory."""
    test_cache_path = tmp_path / ".cache" / "crowdflow_dna" / "deployment.pt"
    with mock.patch("crowdflow_dna.downloader._CACHE_PATH", test_cache_path), \
         mock.patch("crowdflow_dna.downloader._CACHE_DIR", test_cache_path.parent):
        yield test_cache_path


def test_resolve_local_override_exists(tmp_path):
    """Test that an existing local file provided via env overrides cache and download."""
    override_file = tmp_path / "custom_deployment.pt"
    override_file.write_text("dummy model data")

    result = resolve_model_path(str(override_file))
    assert result == str(override_file)


def test_resolve_local_override_missing_falls_back(mock_cache_path):
    """Test that a missing local override falls back to checking the cache."""
    # Create the cached file
    mock_cache_path.parent.mkdir(parents=True, exist_ok=True)
    mock_cache_path.write_text("cached data")

    # Pass a path that doesn't exist
    result = resolve_model_path("does_not_exist.pt")
    
    # Should fall back to the cache
    assert result == str(mock_cache_path)


def test_resolve_uses_cache_if_present(mock_cache_path):
    """Test that existing cached file is reused without downloading."""
    mock_cache_path.parent.mkdir(parents=True, exist_ok=True)
    mock_cache_path.write_text("cached data")

    with mock.patch("urllib.request.urlretrieve") as mock_retrieve:
        result = resolve_model_path(None)
        
        assert result == str(mock_cache_path)
        mock_retrieve.assert_not_called()


def test_resolve_downloads_if_cache_missing(mock_cache_path):
    """Test that a missing cache triggers a download."""
    
    def mock_urlretrieve(url, filename):
        assert url == _CANONICAL_MODEL_URL
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text("downloaded data")
    
    with mock.patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve) as mock_retrieve:
        result = resolve_model_path(None)
        
        assert result == str(mock_cache_path)
        assert mock_cache_path.exists()
        assert mock_cache_path.read_text() == "downloaded data"
        mock_retrieve.assert_called_once()


def test_resolve_download_failure_returns_none(mock_cache_path):
    """Test that download network failures safely return None (dummy mode)."""
    
    def mock_urlretrieve(url, filename):
        raise urllib.error.URLError("Network unreachable")
    
    with mock.patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
        result = resolve_model_path(None)
        
        assert result is None
        assert not mock_cache_path.exists()
