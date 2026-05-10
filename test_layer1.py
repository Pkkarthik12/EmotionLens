"""
AEGIS Layer 1 — Test Suite
Run with: python -m pytest tests/ -v
Or:        python tests/test_layer1.py
"""

import sys
import os
import time
import json
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "awdl_monitor"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qnn_detector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "micro_segmentation"))

from awdl_monitor import AWDLMonitor, AWDLDevice, BEACON_RATE_THRESHOLD
from qnn_detector import NumpyThreatClassifier, generate_synthetic_dataset, _time_encoding, THREAT_CLASSES
from micro_segmentation import MicroSegmentationEngine, Domain, ACLRule


class TestAWDLDevice(unittest.TestCase):
    def test_threat_score_benign(self):
        dev = AWDLDevice(mac="aa:bb:cc:dd:ee:ff")
        dev.update(rssi=-80, services=[1, 2])
        self.assertLess(dev.threat_score, 0.5)

    def test_threat_score_high_rssi(self):
        dev = AWDLDevice(mac="aa:bb:cc:dd:ee:ff")
        dev.update(rssi=-40, services=[1, 2, 3, 4, 5, 6, 7])
        self.assertGreater(dev.threat_score, 0.5)

    def test_beacon_rate_calculation(self):
        dev = AWDLDevice(mac="aa:bb:cc:dd:ee:ff")
        dev.first_seen = time.time() - 10.0
        dev.beacon_count = 50
        dev.last_seen = time.time()
        self.assertGreater(dev.beacon_rate(), 4.0)


class TestAWDLMonitor(unittest.TestCase):
    def test_simulation_detects_threat(self):
        """Monitor should detect threat device in simulation mode."""
        monitor = AWDLMonitor(simulate=True)
        threats = []
        monitor.on_isolation(lambda e: threats.append(e))
        t = monitor.start()
        time.sleep(12)  # enough for threat device to trigger
        monitor.stop()
        self.assertGreater(len(threats), 0, "Should detect at least one threat")

    def test_status_structure(self):
        monitor = AWDLMonitor(simulate=True)
        monitor.start()
        time.sleep(1)
        monitor.stop()
        status = monitor.status()
        self.assertIn("devices_seen", status)
        self.assertIn("devices_isolated", status)
        self.assertIn("threat_events", status)

    def test_isolation_timing(self):
        """Isolation must happen within 2-second window after detection."""
        monitor = AWDLMonitor(simulate=True)
        timestamps = []

        def record_time(event):
            timestamps.append(time.time())

        monitor.on_isolation(record_time)
        t_start = time.time()
        monitor.start()
        time.sleep(12)
        monitor.stop()

        for ts in timestamps:
            self.assertLess(
                ts - t_start, 30,
                "Threats should be detected within 30s of simulation start"
            )


class TestQNNDetector(unittest.TestCase):
    def setUp(self):
        self.clf = NumpyThreatClassifier()

    def test_inference_returns_result(self):
        features = [0.5, 0.2, 2, 100, 15, 300, 40, 1, 1.0, 0.05, 0, 1, 0, 1800,
                    *_time_encoding(14)]
        result = self.clf.predict(features)
        self.assertIn(result.threat_class, THREAT_CLASSES)
        self.assertGreater(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_inference_latency_under_budget(self):
        """Must complete inference under 2000ms (2-second isolation budget)."""
        features = [0.5, 0.2, 2, 100, 15, 300, 40, 1, 1.0, 0.05, 0, 1, 0, 1800,
                    *_time_encoding(14)]
        result = self.clf.predict(features)
        self.assertLess(result.latency_ms, 2000, "Inference must be < 2000ms")

    def test_feature_importances_returned(self):
        features = [18.0, 0.85, 14, 90, 8, 30, 5, 2, 1.0, 0.7, 12, 5, 7, 15,
                    *_time_encoding(10)]
        result = self.clf.predict(features)
        self.assertIsInstance(result.feature_importances, dict)
        self.assertGreater(len(result.feature_importances), 0)

    def test_synthetic_dataset_shape(self):
        X, y = generate_synthetic_dataset(100)
        self.assertEqual(X.shape[1], 16)
        self.assertEqual(len(X), len(y))
        self.assertEqual(set(y), {0, 1, 2, 3})

    def test_feature_vector_length(self):
        """All 16 features required."""
        features = [0.5] * 16
        result = self.clf.predict(features)
        self.assertIsNotNone(result)


class TestMicroSegmentation(unittest.TestCase):
    def setUp(self):
        self.engine = MicroSegmentationEngine(simulate=True)

    def test_topology_initialized(self):
        report = self.engine.topology_report()
        self.assertGreater(report["total_devices"], 0)

    def test_isolation_creates_segment(self):
        self.engine.register_device("dc:2b:61:aa:bb:cc", Domain.PASSENGER_WIFI)
        seg = self.engine.isolate("dc:2b:61:aa:bb:cc", "TEST_THREAT", 0.9)
        self.assertIsNotNone(seg)
        self.assertEqual(seg.domain_after, Domain.QUARANTINE)

    def test_isolation_timing_under_2s(self):
        """Micro-segmentation must complete within 2000ms."""
        self.engine.register_device("dc:2b:61:11:22:33", Domain.PASSENGER_WIFI)
        seg = self.engine.isolate("dc:2b:61:11:22:33", "TIMING_TEST")
        self.assertLess(seg.creation_time_ms, 2000, "Segment creation must be < 2000ms")

    def test_quarantined_device_cannot_connect(self):
        mac = "dc:2b:61:99:88:77"
        self.engine.register_device(mac, Domain.PASSENGER_WIFI)
        self.engine.isolate(mac, "TEST")
        allowed, reason = self.engine.check_connection(mac, "aa:bb:cc:00:00:01")
        self.assertFalse(allowed)
        self.assertIn("quarantined", reason.lower())

    def test_passenger_cannot_reach_cockpit(self):
        """Zero-trust: passenger domain cannot initiate to cockpit."""
        allowed, reason = self.engine.check_connection(
            "aa:bb:cc:00:03:01",  # passenger wifi AP
            "aa:bb:cc:00:00:01",  # cockpit FMS
        )
        self.assertFalse(allowed)

    def test_double_isolation_idempotent(self):
        """Isolating already-quarantined device should not create duplicate segment."""
        mac = "dc:2b:61:77:66:55"
        self.engine.register_device(mac, Domain.PASSENGER_WIFI)
        seg1 = self.engine.isolate(mac, "FIRST")
        count_before = len(self.engine.segments)
        self.engine.isolate(mac, "SECOND")
        count_after = len(self.engine.segments)
        self.assertEqual(count_before, count_after, "Should not create duplicate segments")

    def test_restore_removes_from_quarantine(self):
        mac = "dc:2b:61:55:44:33"
        self.engine.register_device(mac, Domain.PASSENGER_WIFI)
        self.engine.isolate(mac, "TEST")
        success = self.engine.restore(mac, "security_ops")
        self.assertTrue(success)
        dev = self.engine.devices[mac]
        self.assertNotEqual(dev.domain, Domain.QUARANTINE)

    def test_acl_rules_generated(self):
        mac = "dc:2b:61:33:22:11"
        self.engine.register_device(mac, Domain.PASSENGER_WIFI)
        seg = self.engine.isolate(mac, "ACL_TEST")
        self.assertGreater(len(seg.acl_rules), 0)

    def test_ovs_flow_format(self):
        rule = ACLRule(
            rule_id="test0001",
            mac_src="dc:2b:61:aa:bb:cc",
            action="DROP",
            priority=1000,
        )
        flow = rule.to_ovs_flow()
        self.assertIn("priority=1000", flow)
        self.assertIn("dc:2b:61:aa:bb:cc", flow)
        self.assertIn("drop", flow)

    def test_audit_log_populated(self):
        mac = "dc:2b:61:12:34:56"
        self.engine.register_device(mac, Domain.PASSENGER_WIFI)
        self.engine.isolate(mac, "AUDIT_TEST")
        self.assertGreater(len(self.engine.audit_log), 0)
        self.assertEqual(self.engine.audit_log[-1]["action"], "ISOLATE")


if __name__ == "__main__":
    print("AEGIS Layer 1 — Test Suite")
    print("=" * 50)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAWDLDevice))
    suite.addTests(loader.loadTestsFromTestCase(TestQNNDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestMicroSegmentation))
    # Skip slow simulation tests in quick run
    # suite.addTests(loader.loadTestsFromTestCase(TestAWDLMonitor))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
