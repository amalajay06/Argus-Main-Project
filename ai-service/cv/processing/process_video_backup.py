from pathlib import Path
import json
import sys

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from cv.reid.extractor import ReIDExtractor
from cv.reid.vehicle_extractor import VehicleReIDExtractor


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm"
}

MODEL_EXTENSIONS = {
    ".pt"
}


# ============================================================
# RE-ID SETTINGS
# ============================================================

REID_MIN_CONFIDENCE = 0.60

REID_MIN_WIDTH = 80

REID_MIN_HEIGHT = 60

REID_FRAME_GAP = 30


# ============================================================
# RE-ID CLASSES
# ============================================================

PERSON_CLASSES = {
    "person"
}

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

REID_CLASSES = (
    PERSON_CLASSES
    | VEHICLE_CLASSES
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_videos():
    """Return supported video files from the data directory."""

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    return sorted(
        [
            file
            for file in DATA_DIR.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]
    )


def list_models():
    """Return YOLO model files from the project root."""

    return sorted(
        [
            file
            for file in MODEL_DIR.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in MODEL_EXTENSIONS
                and file.name.startswith("yolov8")
            )
        ]
    )


def choose_from_list(items, item_type):
    """Display a numbered list and let the user select an item."""

    if not items:

        print(
            f"\nNo {item_type} found."
        )

        return None

    print(
        f"\nAvailable {item_type}:"
    )

    for index, item in enumerate(
        items,
        start=1
    ):

        print(
            f"  {index}. {item.name}"
        )

    while True:

        choice = input(
            f"\nSelect {item_type} "
            f"(1-{len(items)}): "
        ).strip()

        try:

            choice = int(choice)

            if 1 <= choice <= len(items):

                return items[
                    choice - 1
                ]

        except ValueError:

            pass

        print(
            "Invalid selection. "
            "Please enter a valid number."
        )


def get_video_metadata(video_path):
    """Read basic metadata from the input video."""

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    duration_sec = (
        frame_count / fps
        if fps > 0
        else 0
    )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height
    }


def get_model_name(model_path):
    """Convert model filename into a simple model name."""

    return model_path.stem


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_video(
    video_path,
    model_path
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # VIDEO METADATA
    # --------------------------------------------------------

    metadata = get_video_metadata(
        video_path
    )

    fps = metadata["fps"]
    frame_count = metadata["frame_count"]
    duration_sec = metadata["duration_sec"]
    width = metadata["width"]
    height = metadata["height"]

    model_name = get_model_name(
        model_path
    )

    video_id = video_path.stem

    print(
        "\n" + "=" * 60
    )

    print(
        "VIDEO INFORMATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Video       : {video_path.name}"
    )

    print(
        f"Model       : {model_name}"
    )

    print(
        f"Resolution  : {width} x {height}"
    )

    print(
        f"FPS         : {fps:.2f}"
    )

    print(
        f"Frames      : {frame_count}"
    )

    print(
        f"Duration    : {duration_sec:.2f} seconds"
    )

    print(
        "Tracker     : BoT-SORT"
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    json_output = (
        OUTPUT_DIR
        / f"{video_id}_{model_name}.json"
    )

    annotated_output = (
        OUTPUT_DIR
        / f"{video_id}_{model_name}_annotated.mp4"
    )

    # --------------------------------------------------------
    # RE-ID EMBEDDING DIRECTORY
    # --------------------------------------------------------

    reid_embeddings_dir = (
        OUTPUT_DIR
        / "reid_embeddings"
    )

    reid_embeddings_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print(
        "\nLoading YOLO model..."
    )

    model = YOLO(
        str(model_path)
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD PERSON RE-ID
    # --------------------------------------------------------

    print(
        "\nLoading Person Re-ID..."
    )

    person_reid = ReIDExtractor()

    print(
        "Person Re-ID loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD VEHICLE RE-ID
    # --------------------------------------------------------

    print(
        "\nLoading Vehicle Re-ID..."
    )

    vehicle_reid = VehicleReIDExtractor()

    print(
        "Vehicle Re-ID loaded successfully."
    )

    # --------------------------------------------------------
    # TRACK-LEVEL RE-ID SAMPLING
    # --------------------------------------------------------

    last_reid_frame = {}

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(annotated_output),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create annotated video: "
            f"{annotated_output}"
        )

    cap.release()

    # --------------------------------------------------------
    # JSON STRUCTURE
    # --------------------------------------------------------

    output_data = {

        "video_id": video_id,

        "filename": video_path.name,

        "fps": fps,

        "frame_count": frame_count,

        "duration_sec": duration_sec,

        "width": width,

        "height": height,

        "model": model_name,

        "tracker": "BoT-SORT",

        "frames": []
    }

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    print(
        "\nStarting detection + tracking + Re-ID..."
    )

    print(
        "Person Re-ID : FastReID Market1501"
    )

    print(
        "Vehicle Re-ID: FastReID VeRi"
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    results = model.track(

        source=str(video_path),

        tracker="botsort.yaml",

        stream=True,

        persist=True,

        device=0,

        verbose=False
    )

    # --------------------------------------------------------
    # PROCESS EACH FRAME
    # --------------------------------------------------------

    for frame_idx, result in enumerate(
        results
    ):

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp_sec = (
            frame_idx / fps
            if fps > 0
            else 0
        )

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        detections = []

        boxes = result.boxes

        if (
            boxes is not None
            and len(boxes) > 0
        ):

            xyxy = (
                boxes.xyxy
                .cpu()
                .tolist()
            )

            confidences = (
                boxes.conf
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .cpu()
                .tolist()
            )

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = [
                    None
                ] * len(xyxy)

            for (
                bbox,
                confidence,
                class_id,
                track_id
            ) in zip(
                xyxy,
                confidences,
                class_ids,
                track_ids
            ):

                class_id = int(
                    class_id
                )

                class_name = (
                    model.names[class_id]
                )

                # ------------------------------------------------
                # BASIC DETECTION
                # ------------------------------------------------

                detection = {

                    "track_id": (
                        int(track_id)
                        if track_id is not None
                        else None
                    ),

                    "global_id": None,

                    "class": class_name,

                    "confidence": round(
                        float(confidence),
                        4
                    ),

                    "bbox": [
                        round(
                            float(value),
                            2
                        )
                        for value in bbox
                    ],

                    "attributes": {},

                    "reid": {

                        "embedding_id": None,

                        "similarity": None
                    }
                }

                # ------------------------------------------------
                # RE-ID
                # ------------------------------------------------

                if (

                    track_id is not None

                    and class_name in REID_CLASSES

                    and float(confidence)
                    >= REID_MIN_CONFIDENCE

                ):

                    should_extract = (

                        track_id
                        not in last_reid_frame

                        or

                        frame_idx
                        - last_reid_frame[
                            track_id
                        ]
                        >= REID_FRAME_GAP
                    )

                    if should_extract:

                        x1, y1, x2, y2 = map(
                            int,
                            bbox
                        )

                        x1 = max(
                            0,
                            x1
                        )

                        y1 = max(
                            0,
                            y1
                        )

                        x2 = min(
                            width,
                            x2
                        )

                        y2 = min(
                            height,
                            y2
                        )

                        crop_width = (
                            x2 - x1
                        )

                        crop_height = (
                            y2 - y1
                        )

                        if (

                            crop_width
                            >= REID_MIN_WIDTH

                            and

                            crop_height
                            >= REID_MIN_HEIGHT

                        ):

                            crop = (
                                result.orig_img[
                                    y1:y2,
                                    x1:x2
                                ]
                            )

                            if crop.size > 0:

                                # --------------------------------
                                # SELECT RE-ID MODEL
                                # --------------------------------

                                if class_name in PERSON_CLASSES:

                                    embedding = (
                                        person_reid
                                        .extract_from_array(
                                            crop
                                        )
                                    )

                                    embedding_prefix = (
                                        "person"
                                    )

                                else:

                                    embedding = (
                                        vehicle_reid
                                        .extract_from_array(
                                            crop
                                        )
                                    )

                                    embedding_prefix = (
                                        "vehicle"
                                    )

                                # --------------------------------
                                # EMBEDDING ID
                                # --------------------------------

                                embedding_id = (

                                    f"{embedding_prefix}"
                                    f"_track_{track_id}"
                                    f"_frame_{frame_idx}"
                                )

                                embedding_path = (

                                    reid_embeddings_dir
                                    / f"{embedding_id}.npy"
                                )

                                np.save(
                                    embedding_path,
                                    embedding.numpy()
                                )

                                detection[
                                    "reid"
                                ][
                                    "embedding_id"
                                ] = embedding_id

                                last_reid_frame[
                                    track_id
                                ] = frame_idx

                detections.append(
                    detection
                )

        # ----------------------------------------------------
        # RELATIONSHIPS
        # ----------------------------------------------------

        relations = []

        # ----------------------------------------------------
        # FRAME DATA
        # ----------------------------------------------------

        frame_data = {

            "frame_idx": frame_idx,

            "timestamp_sec": round(
                timestamp_sec,
                3
            ),

            "detections": detections,

            "relations": relations
        }

        output_data[
            "frames"
        ].append(
            frame_data
        )

        # ----------------------------------------------------
        # ANNOTATED FRAME
        # ----------------------------------------------------

        annotated_frame = result.plot(

            conf=True,

            labels=True,

            boxes=True
        )

        writer.write(
            annotated_frame
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if frame_idx % 100 == 0:

            progress = (

                (frame_idx + 1)
                / frame_count
                * 100

                if frame_count > 0

                else 0
            )

            print(
                f"Processing frame "
                f"{frame_idx + 1}/"
                f"{frame_count} "
                f"({progress:.1f}%)"
            )

    # --------------------------------------------------------
    # RELEASE VIDEO WRITER
    # --------------------------------------------------------

    writer.release()

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    print(
        "\nSaving JSON..."
    )

    with open(
        json_output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"JSON output     : "
        f"{json_output}"
    )

    print(
        f"Annotated video : "
        f"{annotated_output}"
    )

    print(
        f"Re-ID embeddings: "
        f"{reid_embeddings_dir}"
    )

    print(
        "\nJSON contains:"
    )

    print(
        "  [OK] Video metadata"
    )

    print(
        "  [OK] Frame numbers"
    )

    print(
        "  [OK] Timestamps"
    )

    print(
        "  [OK] Track IDs"
    )

    print(
        "  [OK] Global ID field"
    )

    print(
        "  [OK] Object classes"
    )

    print(
        "  [OK] Confidence scores"
    )

    print(
        "  [OK] Bounding boxes"
    )

    print(
        "  [OK] Attributes field"
    )

    print(
        "  [OK] Person Re-ID embedding IDs"
    )

    print(
        "  [OK] Vehicle Re-ID embedding IDs"
    )

    print(
        "  [OK] Relationships field"
    )

    print(
        "\nGlobal IDs, attributes and "
        "relationships are reserved for "
        "later modules."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "ARGUS - CCTV VIDEO PROCESSING"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # SELECT VIDEO
    # --------------------------------------------------------

    videos = list_videos()

    video_path = choose_from_list(
        videos,
        "videos"
    )

    if video_path is None:
        return

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    models = list_models()

    model_path = choose_from_list(
        models,
        "YOLOv8 models"
    )

    if model_path is None:
        return

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    process_video(
        video_path,
        model_path
    )


if __name__ == "__main__":
    main()