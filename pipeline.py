import cv2
import yaml
import time
import numpy as np
from typing import Dict, Any, Optional
import os

from detector import VehicleDetector
from tracker import VehicleTracker
from collision import CollisionDetector
from alert import AlertSystem
from utils.draw_utils import annotate_frame


class CollisionDetectionPipeline:
    """
    Main pipeline orchestrating the real-time collision detection system
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the collision detection pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.detector = VehicleDetector(
            model_path=self.config["detection"]["model_path"],
            confidence_threshold=self.config["detection"]["confidence_threshold"],
            vehicle_classes=self.config["detection"]["vehicle_classes"],
        )

        self.tracker = VehicleTracker(
            max_age=self.config["tracking"]["max_age"],
            n_init=self.config["tracking"]["n_init"],
            max_cosine_distance=self.config["tracking"]["max_cosine_distance"],
        )

        self.collision_detector = CollisionDetector(
            overlap_threshold=self.config["collision"]["overlap_threshold"],
            velocity_threshold=self.config["collision"]["velocity_threshold"],
            time_window=self.config["collision"]["time_window"],
            cooldown_period=self.config["collision"]["cooldown_period"],
        )

        self.alert_system = AlertSystem(self.config["alerts"])

        # Initialize camera
        self.cap = None
        self._initialize_camera()

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Video recording for incidents
        self.video_writer = None
        self.recording = False
        self.record_buffer = []  # Store frames before collision
        self.max_buffer_size = 300  # 10 seconds at 30 FPS

        print("Collision Detection Pipeline initialized successfully!")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Creating default config...")
            self._create_default_config(config_path)
            return self._load_config(config_path)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def _initialize_camera(self):
        """Initialize camera capture"""
        camera_source = self.config["camera"]["source"]
        print(f"DEBUG: Trying to open: {camera_source}")

        try:
            self.cap = cv2.VideoCapture(camera_source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera source: {camera_source}")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
            self.cap.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])

            print(f"Camera initialized: {camera_source}")

        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise

    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def _start_recording(self, collision_info: Dict[str, Any]):
        """Start video recording when collision is detected"""
        if not self.config["alerts"]["logging"].get("video_save", False):
            return

        try:
            # Create output filename with timestamp
            timestamp = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(collision_info["timestamp"])
            )
            vehicle_ids = "_".join(map(str, collision_info["vehicle_ids"]))
            filename = f"collision_{timestamp}_vehicles_{vehicle_ids}.mp4"

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_size = (
                self.config["camera"]["width"],
                self.config["camera"]["height"],
            )
            fps = self.config["camera"]["fps"]

            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            self.recording = True

            # Write buffered frames (before collision)
            for frame in self.record_buffer:
                self.video_writer.write(frame)

            print(f"Started recording collision video: {filename}")

        except Exception as e:
            print(f"Error starting video recording: {e}")

    def _stop_recording(self):
        """Stop video recording"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Video recording stopped")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the entire pipeline

        Args:
            frame: Input frame from camera

        Returns:
            Annotated frame with detection and tracking results
        """
        try:
            print("DEBUG: Starting frame processing")

            # Detect vehicles
            detections = self.detector.detect(frame)
            print(f"DEBUG: Got {len(detections)} detections")

            # Format detections for tracker
            detection_array = self.detector.format_detections_for_tracker(detections)
            print(f"DEBUG: Detection array length: {len(detection_array)}")

            # Update tracker
            tracked_objects = self.tracker.update(detection_array, frame)
            print(f"DEBUG: Got {len(tracked_objects)} tracked objects")

            # Detect collisions
            collisions = self.collision_detector.detect_collisions(
                tracked_objects, self.tracker
            )
            print(f"DEBUG: Got {len(collisions)} collisions")

            # Handle collision alerts
            for collision in collisions:
                print(f"COLLISION DETECTED! Vehicles: {collision['vehicle_ids']}")

                # Send alerts
                self.alert_system.send_collision_alert(collision)

                # Start video recording
                if not self.recording:
                    self._start_recording(collision)

            # Update frame buffer for video recording
            if len(self.record_buffer) >= self.max_buffer_size:
                self.record_buffer.pop(0)
            self.record_buffer.append(frame.copy())

            # Write frame to video if recording
            if self.recording and self.video_writer is not None:
                annotated_frame = annotate_frame(
                    frame,
                    tracked_objects,
                    collisions,
                    self.current_fps,
                    self.config["display"]["draw_tracks"],
                )
                self.video_writer.write(annotated_frame)

            # Annotate frame for display
            display_frame = annotate_frame(
                frame,
                tracked_objects,
                collisions,
                self.current_fps,
                self.config["display"]["draw_tracks"],
            )

            return display_frame

        except Exception as e:
            print(f"DEBUG: Error in process_frame: {e}")
            import traceback

            traceback.print_exc()
            return frame  # Return original frame if error

    def run(self):
        """Run the real-time collision detection pipeline"""
        print("Starting collision detection system...")
        print("Press 'q' to quit, 's' to save current frame")

        recording_start_time = None
        recording_duration = 10

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Update FPS
                self._update_fps()

                # Display frame if enabled
                if self.config["display"]["show_window"]:
                    cv2.imshow(self.config["display"]["window_name"], processed_frame)

                # Handle recording duration
                if self.recording:
                    if recording_start_time is None:
                        recording_start_time = time.time()
                    elif time.time() - recording_start_time > recording_duration:
                        self._stop_recording()
                        recording_start_time = None

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")

        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        cv2.destroyAllWindows()
        print("Cleanup completed")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "fps": self.current_fps,
            "recording": self.recording,
            "total_tracks": len(self.tracker.track_history),
            "recent_collisions": len(self.collision_detector.recent_collisions),
            "alerts_enabled": self.alert_system.enabled,
        }


def main():
    """Main entry point"""
    try:
        # Initialize pipeline
        pipeline = CollisionDetectionPipeline("config.yaml")

        # Run the system
        pipeline.run()

    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
