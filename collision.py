import numpy as np
import time
from typing import List, Dict, Any, Tuple, Set
from tracker import VehicleTracker


class CollisionDetector:
    """
    Real-time collision detection using heuristic-based approach
    Analyzes bounding box overlaps and motion patterns
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        velocity_threshold: float = 5.0,
        time_window: float = 1.0,
        cooldown_period: float = 5.0,
    ):
        """
        Initialize collision detector

        Args:
            overlap_threshold: Minimum overlap ratio to consider collision
            velocity_threshold: Minimum velocity change to trigger collision
            time_window: Time window in seconds to analyze motion
            cooldown_period: Seconds to wait before detecting another collision for same vehicles
        """
        self.overlap_threshold = overlap_threshold
        self.velocity_threshold = velocity_threshold
        self.time_window = time_window
        self.cooldown_period = cooldown_period

        # Track recent collisions to avoid duplicate alerts
        self.recent_collisions = {}  # {frozenset([id1, id2]): timestamp}

        # Store previous velocities for sudden change detection
        self.previous_velocities = {}  # {track_id: (vx, vy)}

    def calculate_overlap_ratio(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU ratio (0.0 to 1.0)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def detect_sudden_velocity_change(
        self, track_id: int, current_velocity: Tuple[float, float]
    ) -> bool:
        """
        Detect sudden change in velocity (indicating potential collision impact)

        Args:
            track_id: ID of the track
            current_velocity: Current velocity (vx, vy)

        Returns:
            True if sudden velocity change detected
        """
        if track_id not in self.previous_velocities:
            self.previous_velocities[track_id] = current_velocity
            return False

        prev_vx, prev_vy = self.previous_velocities[track_id]
        curr_vx, curr_vy = current_velocity

        # Calculate velocity change magnitude
        velocity_change = np.sqrt((curr_vx - prev_vx) ** 2 + (curr_vy - prev_vy) ** 2)

        # Update previous velocity
        self.previous_velocities[track_id] = current_velocity

        return velocity_change > self.velocity_threshold

    def is_collision_in_cooldown(self, vehicle_ids: Set[int]) -> bool:
        """
        Check if collision between these vehicles is in cooldown period

        Args:
            vehicle_ids: Set of vehicle IDs involved in potential collision

        Returns:
            True if collision is in cooldown period
        """
        current_time = time.time()
        collision_key = frozenset(vehicle_ids)

        if collision_key in self.recent_collisions:
            time_since_collision = current_time - self.recent_collisions[collision_key]
            return time_since_collision < self.cooldown_period

        return False

    def detect_collisions(
        self, tracked_objects: List[Dict[str, Any]], tracker: VehicleTracker
    ) -> List[Dict[str, Any]]:
        """
        Detect collisions between tracked vehicles

        Args:
            tracked_objects: List of tracked vehicles with positions and IDs
            tracker: VehicleTracker instance for velocity calculation

        Returns:
            List of detected collisions with details
        """
        collisions = []
        current_time = time.time()

        # Compare each pair of vehicles
        for i in range(len(tracked_objects)):
            for j in range(i + 1, len(tracked_objects)):
                vehicle1 = tracked_objects[i]
                vehicle2 = tracked_objects[j]

                id1, id2 = vehicle1["track_id"], vehicle2["track_id"]
                bbox1, bbox2 = vehicle1["bbox"], vehicle2["bbox"]

                # Check if collision is in cooldown
                if self.is_collision_in_cooldown({id1, id2}):
                    continue

                # Calculate overlap
                overlap_ratio = self.calculate_overlap_ratio(bbox1, bbox2)

                if overlap_ratio < self.overlap_threshold:
                    continue

                # Get velocities
                vel1 = tracker.get_track_velocity(id1, self.time_window)
                vel2 = tracker.get_track_velocity(id2, self.time_window)

                # Check for sudden velocity changes
                sudden_change1 = self.detect_sudden_velocity_change(id1, vel1)
                sudden_change2 = self.detect_sudden_velocity_change(id2, vel2)

                # Collision detected if significant overlap AND (sudden velocity change OR low relative motion)
                relative_speed = np.sqrt(
                    (vel1[0] - vel2[0]) ** 2 + (vel1[1] - vel2[1]) ** 2
                )

                collision_detected = overlap_ratio >= self.overlap_threshold and (
                    sudden_change1 or sudden_change2 or relative_speed < 2.0
                )

                if collision_detected:
                    collision_info = {
                        "vehicle_ids": [id1, id2],
                        "timestamp": current_time,
                        "overlap_ratio": overlap_ratio,
                        "velocities": [vel1, vel2],
                        "bboxes": [bbox1.tolist(), bbox2.tolist()],
                        "centers": [vehicle1["center"], vehicle2["center"]],
                        "relative_speed": relative_speed,
                    }

                    collisions.append(collision_info)

                    # Add to cooldown
                    collision_key = frozenset([id1, id2])
                    self.recent_collisions[collision_key] = current_time

        # Clean up old cooldown entries
        self._cleanup_cooldown_cache(current_time)

        return collisions

    def _cleanup_cooldown_cache(self, current_time: float):
        """Clean up expired cooldown entries"""
        expired_keys = []
        for key, timestamp in self.recent_collisions.items():
            if current_time - timestamp > self.cooldown_period:
                expired_keys.append(key)

        for key in expired_keys:
            del self.recent_collisions[key]
