"""Unit tests for the Gradio UI in app.py.

These tests validate the data transformations, error handling, and Gradio
component interactions in process_video() and its helpers.

All pipeline calls and real video processing are mocked out to keep tests
fast and independent of YOLO/torch_geometric.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    _frames_to_video,
    _metadata_to_rows,
    _timeline_to_dataframe,
    process_video,
)
from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.pipeline import PipelineResult
from crowdflow_dna.rendering.timeline import TimelineEntry
from crowdflow_dna.schemas import RiskPrediction


# ---------------------------------------------------------------------------
# process_video()
# ---------------------------------------------------------------------------


def test_process_video_none_input_returns_error_status() -> None:
    """If video_file is None, returns error status without crashing."""
    out_video, tl_rows, md_rows, status = process_video(None)
    assert out_video is None
    assert tl_rows == []
    assert md_rows == []
    assert status == "Please upload a video file."


@patch("app.CrowdFlowPipeline")
@patch("app._frames_to_video")
def test_process_video_returns_four_tuple(mock_frames_to_vid, mock_pipeline_cls) -> None:
    """process_video must always return exactly 4 elements."""
    mock_pipeline_cls.return_value.run.return_value = PipelineResult(
        annotated_frames=[np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    mock_frames_to_vid.return_value = "fake_output.mp4"
    
    result = process_video("fake_input.mp4")
    assert isinstance(result, tuple)
    assert len(result) == 4


@patch("app.CrowdFlowPipeline")
@patch("app._frames_to_video")
def test_process_video_success_returns_video_path(mock_frames_to_vid, mock_pipeline_cls) -> None:
    """On success, the first element must be the output video path string."""
    mock_pipeline_cls.return_value.run.return_value = PipelineResult(
        annotated_frames=[np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    mock_frames_to_vid.return_value = "success_vid.mp4"
    
    out_video, _, _, _ = process_video("fake_input.mp4")
    assert out_video == "success_vid.mp4"


@patch("app.resolve_model_path", return_value=None)
@patch("app.CrowdFlowPipeline")
@patch("app._frames_to_video")
def test_process_video_success_status_contains_ok(mock_frames_to_vid, mock_pipeline_cls, mock_resolve) -> None:
    """On success with no model configured, status must indicate dummy mode."""
    mock_pipeline_cls.return_value.run.return_value = PipelineResult(
        annotated_frames=[np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    mock_frames_to_vid.return_value = "fake.mp4"

    _, _, _, status = process_video("fake_input.mp4")
    assert "dummy mode" in status.lower()


@patch("app.CrowdFlowPipeline")
def test_process_video_crowdflow_error_returns_none_path(mock_pipeline_cls) -> None:
    """If pipeline raises CrowdFlowError, return None for video path."""
    mock_pipeline_cls.return_value.run.side_effect = CrowdFlowError("Mock failure")
    
    out_video, tl_rows, md_rows, _ = process_video("fake_input.mp4")
    assert out_video is None
    assert tl_rows == []
    assert md_rows == []


@patch("app.CrowdFlowPipeline")
def test_process_video_error_status_contains_message(mock_pipeline_cls) -> None:
    """The exception message must be forwarded to the status text."""
    mock_pipeline_cls.return_value.run.side_effect = CrowdFlowError("Bad codec")
    
    _, _, _, status = process_video("fake_input.mp4")
    assert "Bad codec" in status


@patch("app.CrowdFlowPipeline")
def test_process_video_empty_frames_returns_empty_timeline(mock_pipeline_cls) -> None:
    """If pipeline returns no frames, return early without crashing."""
    mock_pipeline_cls.return_value.run.return_value = PipelineResult(annotated_frames=[])
    
    out_video, tl_rows, md_rows, status = process_video("fake_input.mp4")
    assert out_video is None
    assert tl_rows == []
    assert "no frames" in status.lower()


@patch("app.CrowdFlowPipeline")
@patch("app._frames_to_video")
def test_process_video_calls_pipeline_run(mock_frames_to_vid, mock_pipeline_cls) -> None:
    """The pipeline's run() method must be called with the uploaded path."""
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run.return_value = PipelineResult(
        annotated_frames=[np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    mock_frames_to_vid.return_value = "fake.mp4"
    
    process_video("uploaded_test.mp4")
    mock_pipeline.run.assert_called_once_with("uploaded_test.mp4")


@patch("app.CrowdFlowPipeline")
@patch("app._frames_to_video")
def test_process_video_timeline_row_count_matches_entries(mock_frames_to_vid, mock_pipeline_cls) -> None:
    """If dummy mode, row count must equal timeline entry count."""
    mock_pipeline_cls.return_value.run.return_value = PipelineResult(
        annotated_frames=[np.zeros((10, 10, 3), dtype=np.uint8)],
        timeline=[TimelineEntry(0), TimelineEntry(1)]
    )
    mock_frames_to_vid.return_value = "fake.mp4"
    
    _, tl_rows, _, _ = process_video("fake.mp4")
    assert len(tl_rows) == 2


# ---------------------------------------------------------------------------
# _timeline_to_dataframe()
# ---------------------------------------------------------------------------


def test_timeline_to_dataframe_empty_predictions() -> None:
    """Dummy mode timeline entries must produce 'No model' rows."""
    tl = [TimelineEntry(frame_index=0), TimelineEntry(frame_index=1)]
    rows = _timeline_to_dataframe(tl)
    assert len(rows) == 2
    assert rows[0] == [0, "—", "No model", "—"]
    assert rows[1] == [1, "—", "No model", "—"]


def test_timeline_to_dataframe_with_predictions() -> None:
    """Real predictions must be correctly mapped to columns."""
    tl = [
        TimelineEntry(
            frame_index=5,
            predictions=[
                RiskPrediction(region_id=2, label="Safe", confidence=0.981),
                RiskPrediction(region_id=3, label="Critical", confidence=0.876)
            ]
        )
    ]
    rows = _timeline_to_dataframe(tl)
    assert len(rows) == 2
    assert rows[0] == [5, 2, "Safe", "0.98"]
    assert rows[1] == [5, 3, "Critical", "0.88"]


def test_timeline_to_dataframe_empty_timeline() -> None:
    """Empty timeline must return an empty list."""
    assert _timeline_to_dataframe([]) == []


def test_timeline_column_count() -> None:
    """Every row must have exactly 4 columns."""
    tl = [
        TimelineEntry(0),
        TimelineEntry(1, [RiskPrediction(1, "Safe", 0.9)])
    ]
    rows = _timeline_to_dataframe(tl)
    for row in rows:
        assert len(row) == 4


# ---------------------------------------------------------------------------
# _metadata_to_rows()
# ---------------------------------------------------------------------------


def test_metadata_to_rows_contains_fps() -> None:
    """FPS must appear in the metadata rows."""
    md = {"fps": 30.0, "width": 100, "height": 100}
    rows = _metadata_to_rows(md)
    # Check if any row has "FPS" as the first element
    assert any(row[0] == "FPS" and row[1] == "30.00" for row in rows)


def test_metadata_to_rows_contains_resolution() -> None:
    """WidthxHeight must appear in the metadata rows."""
    md = {"fps": 30.0, "width": 640, "height": 480}
    rows = _metadata_to_rows(md)
    assert any(row[0] == "Resolution" and row[1] == "640x480" for row in rows)


def test_metadata_to_rows_property_value_columns() -> None:
    """Each metadata row must have exactly 2 columns."""
    md = {"fps": 30.0, "width": 10, "height": 10, "frame_count": 100}
    rows = _metadata_to_rows(md)
    for row in rows:
        assert len(row) == 2


# ---------------------------------------------------------------------------
# _frames_to_video()
# ---------------------------------------------------------------------------


def test_frames_to_video_with_zero_frames_raises() -> None:
    """Empty frame list must raise CrowdFlowError."""
    with pytest.raises(CrowdFlowError, match="no frames"):
        _frames_to_video([], fps=30.0)


@patch("app.cv2.VideoWriter")
def test_frames_to_video_creates_file(mock_vw) -> None:
    """Helper must instantiate cv2.VideoWriter and call write()."""
    mock_out = MagicMock()
    mock_out.isOpened.return_value = True
    mock_vw.return_value = mock_out

    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    
    path = _frames_to_video(frames, fps=30.0)
    
    assert str(path).endswith(".mp4")
    mock_out.write.assert_called_once()
    mock_out.release.assert_called_once()


# ---------------------------------------------------------------------------
# App / Gradio configuration
# ---------------------------------------------------------------------------


def test_gradio_app_object_exists() -> None:
    """app.demo must be a gr.Blocks instance."""
    from app import demo
    import gradio as gr
    assert isinstance(demo, gr.Blocks)


@patch("app.demo.launch")
def test_gradio_app_is_not_launched_on_import(mock_launch) -> None:
    """Importing app.py must not automatically launch the server."""
    # It is already imported at the top of the test file.
    mock_launch.assert_not_called()
