import smtplib
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from twilio.rest import Client
import os


class AlertSystem:
    """
    Emergency alert system for collision notifications
    Supports SMS notifications via Twilio
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system

        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config
        self.enabled = config.get("enabled", True)

        # SMS configuration
        sms_config = config.get("sms", {})
        self.sms_enabled = sms_config.get("enabled", False)

        if self.sms_enabled:
            self.twilio_client = Client(
                sms_config.get("account_sid"), sms_config.get("auth_token")
            )
            self.from_number = sms_config.get("from_number")
            self.to_numbers = sms_config.get("to_numbers", [])

        # Logging configuration
        logging_config = config.get("logging", {})
        self.logging_enabled = logging_config.get("enabled", True)
        self.log_file = logging_config.get("log_file", "collision_log.csv")

        # Initialize log file
        if self.logging_enabled:
            self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize CSV log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            headers = [
                "timestamp",
                "vehicle_ids",
                "overlap_ratio",
                "relative_speed",
                "velocities",
                "centers",
                "alert_sent",
                "notes",
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.log_file, index=False)

    def send_collision_alert(self, collision_info: Dict[str, Any]) -> bool:
        """
        Send collision alert via configured channels

        Args:
            collision_info: Dictionary containing collision details

        Returns:
            True if alert was sent successfully
        """
        if not self.enabled:
            return False

        # Format alert message
        message = self._format_alert_message(collision_info)

        alert_sent = False

        # Send SMS alerts
        if self.sms_enabled:
            alert_sent = self._send_sms_alert(message) or alert_sent

        # Log collision
        if self.logging_enabled:
            self._log_collision(collision_info, alert_sent)

        return alert_sent

    def _format_alert_message(self, collision_info: Dict[str, Any]) -> str:
        """Format collision information into alert message"""
        vehicle_ids = collision_info["vehicle_ids"]
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(collision_info["timestamp"])
        )
        overlap = collision_info["overlap_ratio"]

        message = f"""
🚨 VEHICLE COLLISION DETECTED 🚨

Time: {timestamp}
Vehicles Involved: {", ".join(map(str, vehicle_ids))}
Overlap Ratio: {overlap:.2f}
Location: Surveillance Camera Feed

Immediate emergency response may be required.
This is an automated alert from the Vehicle Collision Detection System.
        """.strip()

        return message

    def _send_sms_alert(self, message: str) -> bool:
        """Send SMS alert to configured numbers"""
        try:
            for to_number in self.to_numbers:
                self.twilio_client.messages.create(
                    body=message, from_=self.from_number, to=to_number
                )
            print(f"SMS alerts sent to {len(self.to_numbers)} recipients")
            return True
        except Exception as e:
            print(f"Failed to send SMS alert: {e}")
            return False

    def _log_collision(self, collision_info: Dict[str, Any], alert_sent: bool):
        """Log collision to CSV file"""
        try:
            log_entry = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(collision_info["timestamp"])
                ),
                "vehicle_ids": str(collision_info["vehicle_ids"]),
                "overlap_ratio": collision_info["overlap_ratio"],
                "relative_speed": collision_info["relative_speed"],
                "velocities": str(collision_info["velocities"]),
                "centers": str(collision_info["centers"]),
                "alert_sent": alert_sent,
                "notes": "Automated detection",
            }

            # Append to CSV
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_file, mode="a", header=False, index=False)

        except Exception as e:
            print(f"Failed to log collision: {e}")

    def get_recent_collisions(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent collisions from log file

        Args:
            hours: Number of hours to look back

        Returns:
            DataFrame with recent collision records
        """
        try:
            if not os.path.exists(self.log_file):
                return pd.DataFrame()

            df = pd.read_csv(self.log_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter recent records
            cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
            recent_df = df[df["timestamp"] >= cutoff_time]

            return recent_df.sort_values("timestamp", ascending=False)

        except Exception as e:
            print(f"Failed to retrieve recent collisions: {e}")
            return pd.DataFrame()
