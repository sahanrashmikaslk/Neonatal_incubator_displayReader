"""
Live Data Streaming Module
Sends validated OCR readings as JSON to external services
"""
import json
import requests
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LiveDataStreamer:
    """
    Streams validated incubator readings to external services.
    Supports HTTP POST, MQTT, and local file logging.
    """
    
    def __init__(self, 
                 http_endpoint: Optional[str] = None,
                 mqtt_broker: Optional[str] = None,
                 mqtt_topic: str = "incubator/readings",
                 enable_file_logging: bool = False,
                 log_file: str = "live_readings.json"):
        """
        Initialize data streamer.
        
        Args:
            http_endpoint: HTTP API endpoint (e.g., 'http://192.168.1.100:5000/api/readings')
            mqtt_broker: MQTT broker address (e.g., '192.168.1.100')
            mqtt_topic: MQTT topic for publishing
            enable_file_logging: Save readings to local JSON file
            log_file: Path to JSON log file
        """
        self.http_endpoint = http_endpoint
        self.mqtt_broker = mqtt_broker
        self.mqtt_topic = mqtt_topic
        self.enable_file_logging = enable_file_logging
        self.log_file = log_file
        
        # MQTT client (lazy initialization)
        self.mqtt_client = None
        if mqtt_broker:
            try:
                import paho.mqtt.client as mqtt
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(mqtt_broker, 1883, 60)
                self.mqtt_client.loop_start()
                logger.info(f"✅ Connected to MQTT broker: {mqtt_broker}")
            except Exception as e:
                logger.error(f"❌ MQTT connection failed: {e}")
                self.mqtt_client = None
    
    def format_reading(self, validated_readings: Dict, device_id: str = "incubator_001") -> Dict:
        """
        Format validated readings as JSON time-series data.
        
        Args:
            validated_readings: Dictionary of validated readings
            device_id: Device identifier
            
        Returns:
            Formatted JSON dict
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "readings": {}
        }
        
        # Extract values and metadata
        for param, val_data in validated_readings.items():
            data["readings"][param] = {
                "value": val_data.get('value'),
                "status": val_data.get('status', 'unknown'),
                "detection_confidence": val_data.get('detection_confidence', 0),
                "ocr_confidence": val_data.get('ocr_confidence', 0),
                "unit": val_data.get('unit', ''),
                "validated": val_data.get('status') == 'valid'
            }
        
        return data
    
    def send_http(self, data: Dict) -> bool:
        """
        Send data via HTTP POST.
        
        Args:
            data: JSON data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.http_endpoint:
            return False
        
        try:
            response = requests.post(
                self.http_endpoint,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✅ HTTP: Data sent successfully")
                return True
            else:
                logger.warning(f"⚠️ HTTP {response.status_code}: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"❌ HTTP Error: {e}")
            return False
    
    def send_mqtt(self, data: Dict) -> bool:
        """
        Send data via MQTT.
        
        Args:
            data: JSON data to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.mqtt_client:
            return False
        
        try:
            payload = json.dumps(data)
            result = self.mqtt_client.publish(self.mqtt_topic, payload)
            
            if result.rc == 0:
                logger.info(f"✅ MQTT: Data sent to {self.mqtt_topic}")
                return True
            else:
                logger.warning(f"⚠️ MQTT publish failed: {result.rc}")
                return False
        
        except Exception as e:
            logger.error(f"❌ MQTT Error: {e}")
            return False
    
    def save_to_file(self, data: Dict) -> bool:
        """
        Append data to local JSON file.
        
        Args:
            data: JSON data to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_file_logging:
            return False
        
        try:
            # Load existing data
            try:
                with open(self.log_file, 'r') as f:
                    all_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_data = []
            
            # Append new reading
            all_data.append(data)
            
            # Keep only last 1000 readings
            if len(all_data) > 1000:
                all_data = all_data[-1000:]
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            logger.info(f"✅ File: Data saved to {self.log_file}")
            return True
        
        except Exception as e:
            logger.error(f"❌ File Error: {e}")
            return False
    
    def stream(self, validated_readings: Dict, device_id: str = "incubator_001") -> Dict[str, bool]:
        """
        Stream validated readings to all configured outputs.
        
        Args:
            validated_readings: Dictionary of validated readings
            device_id: Device identifier
            
        Returns:
            Dict with status of each streaming method
        """
        # Format data
        data = self.format_reading(validated_readings, device_id)
        
        # Send to all configured outputs
        results = {
            "http": self.send_http(data) if self.http_endpoint else None,
            "mqtt": self.send_mqtt(data) if self.mqtt_client else None,
            "file": self.save_to_file(data) if self.enable_file_logging else None,
            "data": data  # Return the formatted data
        }
        
        return results
    
    def close(self):
        """Clean up connections."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("✅ MQTT connection closed")


def create_sample_data():
    """Create sample validated reading for testing."""
    return {
        "heart_rate_value": {
            "value": "120",
            "status": "valid",
            "detection_confidence": 0.95,
            "ocr_confidence": 0.92,
            "unit": "bpm"
        },
        "humidity_value": {
            "value": "65",
            "status": "valid",
            "detection_confidence": 0.88,
            "ocr_confidence": 0.85,
            "unit": "%"
        },
        "skin_temp_value": {
            "value": "36.5",
            "status": "corrected",
            "detection_confidence": 0.90,
            "ocr_confidence": 0.45,
            "unit": "°C"
        },
        "spo2_value": {
            "value": "98",
            "status": "valid",
            "detection_confidence": 0.93,
            "ocr_confidence": 0.89,
            "unit": "%"
        }
    }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize streamer
    streamer = LiveDataStreamer(
        http_endpoint="http://localhost:5000/api/readings",  # Optional
        mqtt_broker=None,  # Optional: "192.168.1.100"
        enable_file_logging=True,
        log_file="live_readings.json"
    )
    
    # Create sample data
    sample_reading = create_sample_data()
    
    # Stream data
    results = streamer.stream(sample_reading, device_id="incubator_demo")
    
    print("\n📊 Streaming Results:")
    print(f"  HTTP: {'✅ Sent' if results['http'] else '❌ Failed' if results['http'] is False else '⏭️ Not configured'}")
    print(f"  MQTT: {'✅ Sent' if results['mqtt'] else '❌ Failed' if results['mqtt'] is False else '⏭️ Not configured'}")
    print(f"  File: {'✅ Saved' if results['file'] else '❌ Failed' if results['file'] is False else '⏭️ Not configured'}")
    
    print("\n📄 JSON Data:")
    print(json.dumps(results['data'], indent=2))
    
    # Cleanup
    streamer.close()
