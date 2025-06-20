import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def draw_bounding_box(
    frame: np.ndarray,
    bbox: np.ndarray,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box with label on frame

    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        label: Text label to display
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(
        frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1
    )

    # Draw label text
    cv2.putText(
        frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    return frame


def draw_track_trail(
    frame: np.ndarray,
    history: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw track trail showing vehicle movement history

    Args:
        frame: Input frame
        history: List of track history points
        color: RGB color tuple

    Returns:
        Frame with track trail drawn
    """
    if len(history) < 2:
        return frame

    # Draw trail lines
    points = [h["center"] for h in history]
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)

    # Draw small circles at each point
    for point in points[-5:]:  # Only last 5 points
        cv2.circle(frame, point, 3, color, -1)

    return frame


def draw_collision_alert(
    frame: np.ndarray, collision_info: Dict[str, Any]
) -> np.ndarray:
    """
    Draw collision alert overlay on frame

    Args:
        frame: Input frame
        collision_info: Collision information dictionary

    Returns:
        Frame with collision alert drawn
    """
    height, width = frame.shape[:2]

    # Draw red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    # Draw alert text
    alert_text = "COLLISION DETECTED!"
    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Draw text background
    cv2.rectangle(
        frame,
        (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )

    # Draw text
    cv2.putText(
        frame,
        alert_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
    )

    # Draw vehicle ID information
    vehicle_ids = collision_info["vehicle_ids"]
    info_text = f"Vehicles: {', '.join(map(str, vehicle_ids))}"
    cv2.putText(
        frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )

    return frame


def draw_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on frame

    Args:
        frame: Input frame
        fps: Current FPS value

    Returns:
        Frame with FPS counter drawn
    """
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )
    return frame


def annotate_frame(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    collisions: List[Dict[str, Any]],
    fps: float = 0.0,
    draw_trails: bool = True,
) -> np.ndarray:
    """
    Annotate frame with all tracking and collision information

    Args:
        frame: Input frame
        tracked_objects: List of tracked vehicles
        collisions: List of detected collisions
        fps: Current FPS
        draw_trails: Whether to draw track trails

    Returns:
        Fully annotated frame
    """
    annotated_frame = frame.copy()

    # Draw FPS counter
    if fps > 0:
        annotated_frame = draw_fps_counter(annotated_frame, fps)

    # Draw tracked objects
    for obj in tracked_objects:
        track_id = obj["track_id"]
        bbox = obj["bbox"]

        # Choose color based on whether vehicle is involved in collision
        color = (0, 255, 0)  # Green by default
        for collision in collisions:
            if track_id in collision["vehicle_ids"]:
                color = (0, 0, 255)  # Red if in collision
                break

        # Draw bounding box
        label = f"Vehicle {track_id}"
        annotated_frame = draw_bounding_box(annotated_frame, bbox, label, color)

        # Draw track trail
        if draw_trails and len(obj["history"]) > 1:
            annotated_frame = draw_track_trail(annotated_frame, obj["history"], color)

    # Draw collision alerts
    for collision in collisions:
        annotated_frame = draw_collision_alert(annotated_frame, collision)

    return annotated_frame
