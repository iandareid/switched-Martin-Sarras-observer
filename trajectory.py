"""Backward-compatible exports for trajectory utilities."""

from trajgen import TrajectoryBuffer, generate_default_demo, unwrap_yaw, wrap_angle

# Legacy alias used by older callsites/tests.
TrajectoryBuffer.demo = staticmethod(generate_default_demo)
