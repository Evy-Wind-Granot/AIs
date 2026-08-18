#!/usr/bin/env python3
"""
Tier 3: Online Real-Time Streaming Daemon
Consumes 1 Hz live messages via NATS JetStream/MQTT, applies rolling baseline calculations,
and triggers real-time alerts.
"""
import argparse
import asyncio
import json
from collections import deque
import numpy as np


class SlidingBuffer:
    def __init__(self, maxlen: int = 3600):  # 1-hour rolling buffer at 1 Hz
        self.buffer = deque(maxlen=maxlen)

    def append(self, value: float):
        self.buffer.append(value)

    def get_baseline(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.median(self.buffer))


class StreamProcessor:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.threshold = self.config.get("derived_alert_threshold_nT", 10.0)
        self.buffer = SlidingBuffer(maxlen=3600)

    def process_sample(self, payload: dict) -> dict:
        timestamp = payload.get("timestamp")
        x, y, z = payload.get("x", 0.0), payload.get("y", 0.0), payload.get("z", 0.0)
        magnitude = (x**2 + y**2 + z**2)**0.5

        self.buffer.append(magnitude)
        baseline = self.buffer.get_baseline()
        residual = abs(magnitude - baseline)

        alert = residual > self.threshold
        return {
            "timestamp": timestamp,
            "station_id": self.config.get("station_id"),
            "magnitude_nT": magnitude,
            "baseline_nT": baseline,
            "residual_nT": residual,
            "alert": alert
        }


async def mock_nats_consumer(processor: StreamProcessor):
    """Simulates subscribing to NATS JetStream at 1 Hz cadence."""
    print(f"🚀 Streaming processor active for station: {processor.config.get('station_id')}")
    print("Subscribed to topic: sensors.magnetometer.1hz\n")
    
    t = 0
    while True:
        # Simulated 1 Hz sensor feed payload
        sample = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "x": 18000.0 + np.random.normal(0, 0.5) + (50.0 if t == 10 else 0.0),
            "y": -1500.0 + np.random.normal(0, 0.5),
            "z": 52000.0 + np.random.normal(0, 0.5)
        }
        
        result = processor.process_sample(sample)
        if result["alert"]:
            print(f"🚨 [ALERT TRIPPED] Residual: {result['residual_nT']:.2f} nT (Threshold: {processor.threshold:.2f} nT)")
        else:
            print(f"[OK] Time: {result['timestamp']} | Mag: {result['magnitude_nT']:.2f} nT | Baseline: {result['baseline_nT']:.2f} nT")

        t += 1
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time streaming processor daemon.")
    parser.add_argument("--config", required=True, help="Path to tuned production_config.json")
    args = parser.parse_args()

    processor = StreamProcessor(args.config)
    
    import pandas as pd
    try:
        asyncio.run(mock_nats_consumer(processor))
    except KeyboardInterrupt:
        print("\nStopping streaming processor daemon.")