from pathlib import Path
import json
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = Path("ai-service/outputs")


# ============================================================
# LOAD JSON
# ============================================================

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# NORMALIZE JSON FORMAT
# ============================================================

def normalize_data(data):
    """
    Supports the current Argus JSON format and the older format.

    Current format:
        video metadata at root
        frames under "frames"

    Older format:
        video metadata under "video"
        frames under "detections"
    """

    # --------------------------------------------------------
    # CURRENT FORMAT
    # --------------------------------------------------------

    if "frames" in data:

        video = {
            "video_id": data["video_id"],
            "filename": data["filename"],
            "fps": data["fps"],
            "frame_count": data["frame_count"],
            "width": data["width"],
            "height": data["height"],
            "duration_sec": data["duration_sec"]
        }

        frames = data["frames"]

        return video, frames

    # --------------------------------------------------------
    # OLD FORMAT
    # --------------------------------------------------------

    if "video" in data and "detections" in data:

        return data["video"], data["detections"]

    # --------------------------------------------------------
    # UNKNOWN FORMAT
    # --------------------------------------------------------

    raise ValueError(
        "Unknown JSON format. Expected either "
        "'frames' or 'video' + 'detections'."
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze(json_path):

    data = load_json(json_path)

    video, frames = normalize_data(data)

    print("=" * 60)
    print("ARGUS DETECTION ANALYSIS")
    print("=" * 60)

    # ========================================================
    # VIDEO
    # ========================================================

    print()
    print("VIDEO")
    print("-" * 60)

    print(f"Video ID       : {video['video_id']}")
    print(f"Filename       : {video['filename']}")
    print(f"FPS            : {video['fps']:.2f}")
    print(f"Frame count    : {video['frame_count']}")
    print(f"Resolution     : {video['width']}x{video['height']}")
    print(f"Duration       : {video['duration_sec']:.2f} sec")

    # ========================================================
    # FRAME STATISTICS
    # ========================================================

    total_frames = len(frames)

    frames_with_detections = sum(
        1 for frame in frames
        if frame.get("detections", [])
    )

    empty_frames = total_frames - frames_with_detections

    print()
    print("FRAME STATISTICS")
    print("-" * 60)

    print(f"Total frames             : {total_frames}")
    print(f"Frames with detections   : {frames_with_detections}")
    print(f"Empty frames             : {empty_frames}")

    # ========================================================
    # DETECTION STATISTICS
    # ========================================================

    all_detections = []

    for frame in frames:
        all_detections.extend(
            frame.get("detections", [])
        )

    total_detections = len(all_detections)

    track_ids = {
        d["track_id"]
        for d in all_detections
        if d.get("track_id") is not None
    }

    low_confidence = [
        d for d in all_detections
        if d.get("confidence", 0) < 0.40
    ]

    average_detections = (
        total_detections / total_frames
        if total_frames > 0 else 0
    )

    low_conf_percentage = (
        len(low_confidence) / total_detections * 100
        if total_detections > 0 else 0
    )

    print()
    print("DETECTION STATISTICS")
    print("-" * 60)

    print(f"Total detections         : {total_detections}")
    print(f"Unique track IDs         : {len(track_ids)}")

    print(
        f"Low-confidence detections (< 0.40) : "
        f"{len(low_confidence)}"
    )

    print(
        f"Average detections/frame : "
        f"{average_detections:.2f}"
    )

    print(
        f"Low-confidence percentage : "
        f"{low_conf_percentage:.2f}%"
    )

    # ========================================================
    # CLASS STATISTICS
    # ========================================================

    class_detections = Counter()
    class_tracks = defaultdict(set)

    for detection in all_detections:

        class_name = detection["class"]

        class_detections[class_name] += 1

        if detection.get("track_id") is not None:
            class_tracks[class_name].add(
                detection["track_id"]
            )

    print()
    print("CLASS STATISTICS")
    print("-" * 60)

    print(
        f"{'Class':<20}"
        f"{'Detections':>12}"
        f"{'Tracks':>10}"
    )

    print("-" * 45)

    for class_name, count in class_detections.most_common():

        print(
            f"{class_name:<20}"
            f"{count:>12}"
            f"{len(class_tracks[class_name]):>10}"
        )

    # ========================================================
    # CONFIDENCE BY CLASS
    # ========================================================

    class_confidences = defaultdict(list)

    for detection in all_detections:

        class_confidences[
            detection["class"]
        ].append(
            detection["confidence"]
        )

    print()
    print("CONFIDENCE BY CLASS")
    print("-" * 60)

    print(
        f"{'Class':<20}"
        f"{'Avg':>10}"
        f"{'Min':>10}"
        f"{'<0.40':>10}"
        f"{'% <0.40':>12}"
    )

    print("-" * 65)

    for class_name in class_detections:

        confidences = class_confidences[class_name]

        average_conf = (
            sum(confidences) / len(confidences)
        )

        minimum_conf = min(confidences)

        low_count = sum(
            1
            for confidence in confidences
            if confidence < 0.40
        )

        low_percentage = (
            low_count / len(confidences) * 100
        )

        print(
            f"{class_name:<20}"
            f"{average_conf:>10.3f}"
            f"{minimum_conf:>10.3f}"
            f"{low_count:>10}"
            f"{low_percentage:>11.2f}%"
        )

    # ========================================================
    # TRACK STATISTICS
    # ========================================================

    tracks = defaultdict(list)

    for frame in frames:

        frame_idx = frame["frame_idx"]

        for detection in frame.get("detections", []):

            track_id = detection.get("track_id")

            if track_id is not None:

                tracks[track_id].append({
                    "frame_idx": frame_idx,
                    "class": detection["class"]
                })

    track_information = []

    for track_id, observations in tracks.items():

        frame_numbers = [
            observation["frame_idx"]
            for observation in observations
        ]

        first_frame = min(frame_numbers)
        last_frame = max(frame_numbers)

        track_class = Counter(
            observation["class"]
            for observation in observations
        ).most_common(1)[0][0]

        duration = (
            last_frame - first_frame
        ) / video["fps"]

        track_information.append({
            "track_id": track_id,
            "class": track_class,
            "first": first_frame,
            "last": last_frame,
            "observations": len(observations),
            "duration": duration
        })

    track_information.sort(
        key=lambda x: x["observations"],
        reverse=True
    )

    # ========================================================
    # LONGEST TRACKS
    # ========================================================

    print()
    print("LONGEST TRACKS")
    print("-" * 60)

    print(
        f"{'ID':<6}"
        f"{'Class':<20}"
        f"{'First':>7}"
        f"{'Last':>8}"
        f"{'Obs.':>8}"
        f"{'Duration':>12}"
    )

    print("-" * 65)

    for track in track_information[:20]:

        print(
            f"{track['track_id']:<6}"
            f"{track['class']:<20}"
            f"{track['first']:>7}"
            f"{track['last']:>8}"
            f"{track['observations']:>8}"
            f"{track['duration']:>10.2f}s"
        )

    # ========================================================
    # SHORT TRACKS
    # ========================================================

    short_tracks = [
        track
        for track in track_information
        if track["observations"] <= 5
    ]

    print()
    print("SHORT TRACKS")
    print("-" * 60)

    print(
        f"Tracks with <= 5 observations: "
        f"{len(short_tracks)}"
    )

    # ========================================================
    # RELATIONSHIP STATISTICS
    # ========================================================

    total_relations = 0

    relation_types = Counter()

    for frame in frames:

        relations = frame.get("relations", [])

        total_relations += len(relations)

        for relation in relations:

            relation_name = relation.get(
                "relation",
                "unknown"
            )

            relation_types[relation_name] += 1

    print()
    print("RELATIONSHIP STATISTICS")
    print("-" * 60)

    print(f"Total relationships : {total_relations}")

    if relation_types:

        for relation_name, count in relation_types.most_common():

            print(
                f"{relation_name:<25}"
                f"{count:>10}"
            )

    else:

        print("No relationships detected yet.")

    # ========================================================
    # SUMMARY
    # ========================================================

    average_observations = (
        total_detections / len(track_information)
        if track_information else 0
    )

    if track_information:

        longest = track_information[0]

        print()
        print("SUMMARY")
        print("-" * 60)

        print(
            f"Average observations/track : "
            f"{average_observations:.2f}"
        )

        print(
            f"Longest track              : "
            f"ID {longest['track_id']} "
            f"({longest['class']}) "
            f"for {longest['duration']:.2f}s"
        )

    print()
    print("Analysis complete.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not OUTPUT_DIR.exists():

        print("Output directory not found:")
        print(OUTPUT_DIR)

        raise SystemExit(1)

    json_files = sorted(
        [
            file
            for file in OUTPUT_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".json"
        ]
    )

    if not json_files:

        print("No JSON files found in:")
        print(OUTPUT_DIR)

        raise SystemExit(1)

    print("Available detection files:")
    print()

    for i, json_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"{i}. {json_file.name}"
        )

    print()

    choice = input(
        "Enter JSON number to analyze: "
    ).strip()

    try:

        choice = int(choice)

        json_path = json_files[choice - 1]

    except (ValueError, IndexError):

        print("Invalid selection.")

        raise SystemExit(1)

    print()
    print(f"Selected: {json_path.name}")
    print()

    analyze(json_path)