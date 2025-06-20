import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Any, Tuple


class VehicleTracker:
    """
    Multi-object tracking using DeepSORT
    Maintains vehicle trajectories and IDs across frames
    """

    def __init__(
        self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.5
    ):
        """
        Initialize DeepSORT tracker

        Args:
            max_age: Maximum number of frames to keep a track without detection
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IoU distance for matching detections to tracks
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
            half=False,
        )

        # Store track history for trajectory analysis
        self.track_history = {}  # {track_id: [{'bbox': [...], 'timestamp': float, 'center': (x, y)}, ...]}
        self.max_history_length = 30  # Keep last 30 positions per track

    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections

        Args:
            detections: Array of detections [[x1, y1, x2, y2, confidence], ...]
            frame: Current frame for feature extraction

        Returns:
            List of tracked objects with IDs and bounding boxes
        """
        import time

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        tracked_objects = []
        current_time = time.time()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()  # [left, top, right, bottom]

            # Calculate center point
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            center = (center_x, center_y)

            # Store in history
            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append(
                {"bbox": bbox, "timestamp": current_time, "center": center}
            )

            # Limit history length
            if len(self.track_history[track_id]) > self.max_history_length:
                self.track_history[track_id] = self.track_history[track_id][
                    -self.max_history_length :
                ]

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "bbox": bbox.astype(int),
                    "center": center,
                    "history": self.track_history[track_id].copy(),
                }
            )

        return tracked_objects

    def get_track_velocity(
        self, track_id: int, time_window: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate velocity of a track over specified time window

        Args:
            track_id: ID of the track
            time_window: Time window in seconds to calculate velocity

        Returns:
            Velocity as (vx, vy) in pixels per second
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return (0.0, 0.0)

        history = self.track_history[track_id]
        current_time = history[-1]["timestamp"]

        # Find the position from time_window seconds ago
        reference_time = current_time - time_window
        reference_pos = None

        for i in range(len(history) - 1, -1, -1):
            if history[i]["timestamp"] <= reference_time:
                reference_pos = history[i]
                break

        if reference_pos is None:
            reference_pos = history[0]

        # Calculate velocity
        current_pos = history[-1]
        dt = current_pos["timestamp"] - reference_pos["timestamp"]

        if dt <= 0:
            return (0.0, 0.0)

        dx = current_pos["center"][0] - reference_pos["center"][0]
        dy = current_pos["center"][1] - reference_pos["center"][1]

        vx = dx / dt
        vy = dy / dt

        return (vx, vy)
