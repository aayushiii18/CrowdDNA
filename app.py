"""Gradio UI entry point for CrowdFlow DNA Phase 8.

Provides an interface to upload a video, run the CrowdFlowPipeline,
and display the annotated video, risk timeline, and metadata.

Set the ``CROWDDNA_MODEL_PATH`` environment variable to the path of an
exported ``.pt`` or ``.onnx`` deployment model to enable inference mode.
When the variable is absent, the pipeline runs in dummy mode.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.inference import ModelNotFoundError, UnsupportedModelFormatError
from crowdflow_dna.pipeline import CrowdFlowPipeline
from crowdflow_dna.rendering.timeline import TimelineEntry
from crowdflow_dna.downloader import resolve_model_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def _frames_to_video(frames: List[np.ndarray], fps: float) -> str:
    """Convert a list of BGR frames to an MP4 video file.

    Args:
        frames: List of BGR numpy arrays. Must not be empty.
        fps: The effective frames per second for the output video.

    Returns:
        String path to the temporary MP4 file.

    Raises:
        CrowdFlowError: If the frame list is empty or VideoWriter fails.
    """
    if not frames:
        raise CrowdFlowError("Cannot create video: no frames provided.")

    height, width = frames[0].shape[:2]
    
    # Create a temporary file that Gradio can read later.
    # Windows compatibility: close the file handle so cv2 can open it.
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise CrowdFlowError(f"Failed to open VideoWriter for {output_path}")

    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()

    return output_path


def _timeline_to_dataframe(timeline: List[TimelineEntry]) -> List[List[Any]]:
    """Convert timeline entries to a list of lists for gr.DataFrame.

    If a frame has no predictions (dummy mode), outputs a row with
    sentinel values indicating no model was used.
    """
    rows = []
    for entry in timeline:
        if not entry.predictions:
            rows.append([entry.frame_index, "—", "No model", "—"])
        else:
            for pred in entry.predictions:
                rows.append(
                    [
                        entry.frame_index,
                        pred.region_id,
                        pred.label,
                        f"{pred.confidence:.2f}",
                    ]
                )
    return rows


def _metadata_to_rows(metadata: Dict[str, Any]) -> List[List[Any]]:
    """Convert metadata dict to key-value rows for gr.DataFrame."""
    if not metadata:
        return []
    
    return [
        ["FPS", f"{metadata.get('fps', 0):.2f}"],
        ["Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"],
        ["Total Frames", metadata.get("frame_count", 0)],
        ["Duration (s)", f"{metadata.get('duration_seconds', 0):.2f}"],
        ["Sample Rate", metadata.get("sample_rate", 1)],
    ]


def process_video(
    video_file: Optional[str],
) -> Tuple[Optional[str], List[List[Any]], List[List[Any]], str]:
    """Gradio callback to run the pipeline on an uploaded video.

    Args:
        video_file: Path to the uploaded video file provided by Gradio.

    Returns:
        Tuple of (output_video_path, timeline_rows, metadata_rows, status_msg).
    """
    if not video_file:
        return None, [], [], "Please upload a video file."

    logger.info("Processing uploaded video: %s", video_file)

    # Attempt to build an inference-mode pipeline when a model path is configured.
    # Fall back to dummy mode gracefully on any loading error.
    inference_active = False
    raw_env_path = os.environ.get("CROWDDNA_MODEL_PATH")
    model_path = resolve_model_path(raw_env_path)
    try:
        pipeline = CrowdFlowPipeline(model_path=model_path)
        inference_active = model_path is not None
    except (ModelNotFoundError, UnsupportedModelFormatError) as exc:
        logger.warning(
            "Could not load deployment model (%s). Falling back to dummy mode.", exc
        )
        pipeline = CrowdFlowPipeline(model_path=None)

    try:
        result = pipeline.run(video_file)
    except CrowdFlowError as exc:
        logger.error("Pipeline failed: %s", exc)
        return None, [], [], f"Error: {exc}"
    except Exception as exc:
        logger.exception("Unexpected error during pipeline run.")
        return None, [], [], f"Error: Unexpected failure: {exc}"

    if not result.annotated_frames:
        return None, [], [], "Error: Pipeline returned no frames."

    # Use the effective FPS for the sampled frames to maintain normal speed playback
    fps = result.metadata.get("fps", 30.0)
    sample_rate = result.metadata.get("sample_rate", 1)
    effective_fps = max(1.0, float(fps) / float(sample_rate))

    try:
        output_path = _frames_to_video(result.annotated_frames, effective_fps)
    except CrowdFlowError as exc:
        return None, [], [], f"Error: {exc}"

    timeline_data = _timeline_to_dataframe(result.timeline)
    metadata_data = _metadata_to_rows(result.metadata)

    if inference_active:
        backend = result.metadata.get("backend", "TorchScript")
        fmt = result.metadata.get("model_format", "unknown")
        version = result.metadata.get("model_version") or "unversioned"
        status_msg = (
            f"✅ Analysis complete — inference mode · {backend} ({fmt}) · version: {version}."
        )
    else:
        status_msg = "✅ Analysis complete (dummy mode — no risk model loaded)."
    return output_path, timeline_data, metadata_data, status_msg


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CrowdFlow DNA — Crowd Risk Analyser") as demo:
    gr.Markdown(
        """
        # CrowdFlow DNA — Crowd Risk Analyser
        Upload a video to run the end-to-end vision pipeline.
        Operating mode (inference or dummy) is determined dynamically by the `CROWDDNA_MODEL_PATH` environment variable.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input Video", format="mp4")
            run_btn = gr.Button("Run Analysis", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=1):
            output_video = gr.Video(label="Annotated Output", interactive=False, format="mp4")

    with gr.Row():
        timeline_table = gr.DataFrame(
            label="Risk Timeline",
            headers=["Frame", "Region ID", "Risk Label", "Confidence"],
            interactive=False,
        )

    with gr.Row():
        metadata_table = gr.DataFrame(
            label="Video Metadata",
            headers=["Property", "Value"],
            interactive=False,
        )

    run_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, timeline_table, metadata_table, status_box],
    )

if __name__ == "__main__":
    demo.launch()
