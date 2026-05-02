#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Odometry
from traj_utils.msg import PolyTraj


def _v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _wait_for_valid_ros_time(timeout):
    start_wall = time.time()
    rate = rospy.Rate(100.0)
    warned = False
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() <= 1.0e-6:
        if timeout > 0.0 and time.time() - start_wall > timeout:
            rospy.logwarn(
                "[poly_traj_test_pub] Timed out waiting for valid ROS time; trajectory start_time may be invalid."
            )
            return False
        if not warned:
            rospy.loginfo("[poly_traj_test_pub] Waiting for valid ROS time before building trajectory...")
            warned = True
        rate.sleep()
    return not rospy.is_shutdown()


def _hermite_cubic_coeff(p0, v0, p1, v1, dt):
    """
    Cubic Hermite in local segment time t in [0, dt]:
      p(t) = a3*t^3 + a2*t^2 + a1*t + a0
    """
    dp = _v_sub(p1, p0)
    inv_t = 1.0 / dt
    inv_t2 = inv_t * inv_t
    inv_t3 = inv_t2 * inv_t

    a0 = p0
    a1 = v0
    a2 = _v_sub(_v_mul(dp, 3.0 * inv_t2), _v_mul(_v_add(_v_mul(v0, 2.0), v1), inv_t))
    a3 = _v_add(_v_mul(dp, -2.0 * inv_t3), _v_mul(_v_add(v0, v1), inv_t2))
    return a3, a2, a1, a0


def _estimate_tangents(points, durations):
    # Endpoints set to zero velocity to avoid sudden starts from hover.
    n = len(points)
    tangents = [(0.0, 0.0, 0.0) for _ in range(n)]
    if n <= 2:
        return tangents

    for i in range(1, n - 1):
        dt_prev = max(durations[i - 1], 1.0e-3)
        dt_next = max(durations[i], 1.0e-3)
        v_prev = _v_mul(_v_sub(points[i], points[i - 1]), 1.0 / dt_prev)
        v_next = _v_mul(_v_sub(points[i + 1], points[i]), 1.0 / dt_next)
        tangents[i] = _v_mul(_v_add(v_prev, v_next), 0.5)
    return tangents


def _build_default_waypoints(num_segments):
    # Build a smooth S-like path with configurable segment count.
    # Number of waypoints = num_segments + 1.
    n = max(1, int(num_segments))
    length = 3.5
    points = []
    for i in range(n + 1):
        s = float(i) / float(n)  # normalized path progress [0, 1]
        x = length * s
        y = 0.55 * math.sin(2.0 * math.pi * s) + 0.12 * math.sin(4.0 * math.pi * s)
        z = 1.0 + 0.12 * math.sin(math.pi * s)
        points.append((x, y, z))
    return points


def _build_durations(points, cruise_speed, min_duration):
    durations = []
    for i in range(len(points) - 1):
        dist = _v_norm(_v_sub(points[i + 1], points[i]))
        dt = max(dist / max(cruise_speed, 0.05), min_duration)
        durations.append(dt)
    return durations


def make_multi_segment_msg(traj_id, start_delay, cruise_speed, min_duration, num_segments):
    msg = PolyTraj()
    msg.drone_id = 0
    msg.traj_id = traj_id
    msg.start_time = rospy.Time.now() + rospy.Duration.from_sec(start_delay)
    msg.order = 3

    points = _build_default_waypoints(num_segments=num_segments)
    durations = _build_durations(points, cruise_speed=cruise_speed, min_duration=min_duration)
    tangents = _estimate_tangents(points, durations)

    coef_x = []
    coef_y = []
    coef_z = []
    for i, dt in enumerate(durations):
        a3, a2, a1, a0 = _hermite_cubic_coeff(points[i], tangents[i], points[i + 1], tangents[i + 1], dt)
        # poly convention in this repo:
        # p(t) = c0*t^order + c1*t^(order-1) + ... + c(order), order=3.
        coef_x.extend([a3[0], a2[0], a1[0], a0[0]])
        coef_y.extend([a3[1], a2[1], a1[1], a0[1]])
        coef_z.extend([a3[2], a2[2], a1[2], a0[2]])

    msg.duration = durations
    msg.coef_x = coef_x
    msg.coef_y = coef_y
    msg.coef_z = coef_z
    return msg


class CompareRecorder:
    def __init__(self, msg):
        self.msg = msg
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)
        self.odom_topic = rospy.get_param(
            "~odom_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.output_root = rospy.get_param(
            "~output_root",
            os.path.join(package_dir, "logs", "traj_track_step"),
        )
        self.shared_log_dir_param = rospy.get_param(
            "~shared_log_dir_param",
            "/ommpc_controller/traj_track_log_dir",
        )
        self.ref_sample_dt = float(rospy.get_param("~ref_sample_dt", 0.02))
        self.max_follow_time = float(rospy.get_param("~max_follow_time", 30.0))
        self.capture_before_start = bool(rospy.get_param("~capture_before_start", False))

        self.ref_samples = []
        self.actual_samples = []
        self.stop_time = None
        self.recording = False
        self.saved = False

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=200)
        rospy.on_shutdown(self.save)

        self._build_ref_samples()
        total_dur = float(sum(self.msg.duration))
        hard_limit = self.msg.start_time.to_sec() + total_dur + 2.0
        max_limit = rospy.Time.now().to_sec() + self.max_follow_time
        self.stop_time = min(hard_limit, max_limit)
        self.recording = True
        rospy.loginfo(
            "[poly_traj_test_pub] Compare logger enabled. odom=%s total_dur=%.2f",
            self.odom_topic,
            total_dur,
        )

    @staticmethod
    def _eval_piece(coeffs, order, t):
        v = 0.0
        for i, c in enumerate(coeffs):
            v += c * (t ** (order - i))
        return v

    def _eval_ref_xyz(self, t_global):
        if t_global < 0.0:
            t_global = 0.0
        order = int(self.msg.order)
        per_piece = order + 1
        durations = list(self.msg.duration)
        if not durations:
            return None
        num_piece = len(durations)
        if len(self.msg.coef_x) < num_piece * per_piece:
            return None
        if len(self.msg.coef_y) < num_piece * per_piece:
            return None
        if len(self.msg.coef_z) < num_piece * per_piece:
            return None

        remain = t_global
        piece_idx = num_piece - 1
        local_t = durations[-1]
        for i, dt in enumerate(durations):
            if remain <= dt:
                piece_idx = i
                local_t = remain
                break
            remain -= dt

        st = piece_idx * per_piece
        ed = st + per_piece
        x = self._eval_piece(self.msg.coef_x[st:ed], order, local_t)
        y = self._eval_piece(self.msg.coef_y[st:ed], order, local_t)
        z = self._eval_piece(self.msg.coef_z[st:ed], order, local_t)
        return x, y, z

    def _build_ref_samples(self):
        total_dur = float(sum(self.msg.duration))
        if total_dur <= 0.0:
            return
        n = max(2, int(math.ceil(total_dur / self.ref_sample_dt)) + 1)
        for i in range(n):
            t = min(i * self.ref_sample_dt, total_dur)
            xyz = self._eval_ref_xyz(t)
            if xyz is None:
                continue
            self.ref_samples.append([t, xyz[0], xyz[1], xyz[2]])

    def _odom_cb(self, odom):
        if not self.recording:
            return
        now_sec = odom.header.stamp.to_sec() if odom.header.stamp.to_sec() > 1e-6 else rospy.Time.now().to_sec()
        if self.stop_time is not None and now_sec > self.stop_time:
            self.recording = False
            return
        rel_t = now_sec - self.msg.start_time.to_sec()
        if (not self.capture_before_start) and rel_t < 0.0:
            return
        ref_xyz = self._eval_ref_xyz(rel_t)
        if ref_xyz is None:
            return
        p = odom.pose.pose.position
        self.actual_samples.append([rel_t, ref_xyz[0], ref_xyz[1], ref_xyz[2], p.x, p.y, p.z])

    def _write_csv(self, path, header, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def _resolve_out_dir(self):
        shared_out_dir = rospy.get_param(self.shared_log_dir_param, "")
        if shared_out_dir:
            return shared_out_dir
        return os.path.join(self.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))

    def _combined_rows(self):
        rows = []
        if self.actual_samples:
            for r in self.actual_samples:
                err_x = r[4] - r[1]
                err_y = r[5] - r[2]
                err_z = r[6] - r[3]
                err_xy = math.sqrt(err_x * err_x + err_y * err_y)
                err_xyz = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
                rows.append([
                    "actual_vs_reference",
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    err_x,
                    err_y,
                    err_z,
                    err_xy,
                    err_xyz,
                ])
            return rows

        for r in self.ref_samples:
            rows.append([
                "reference",
                r[0],
                r[1],
                r[2],
                r[3],
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ])
        return rows

    def _plot(self, path):
        if not self.actual_samples:
            return
        t = [r[0] for r in self.actual_samples]
        rx = [r[1] for r in self.actual_samples]
        ry = [r[2] for r in self.actual_samples]
        rz = [r[3] for r in self.actual_samples]
        ax = [r[4] for r in self.actual_samples]
        ay = [r[5] for r in self.actual_samples]
        az = [r[6] for r in self.actual_samples]

        fig = plt.figure(figsize=(12, 8))
        p1 = fig.add_subplot(2, 2, 1)
        p1.plot(rx, ry, "b-", label="reference")
        p1.plot(ax, ay, "r-", label="actual")
        p1.set_title("XY trajectory")
        p1.set_xlabel("x [m]")
        p1.set_ylabel("y [m]")
        p1.grid(True)
        p1.axis("equal")
        p1.legend()

        p2 = fig.add_subplot(2, 2, 2)
        p2.plot(t, rx, "b--", label="ref x")
        p2.plot(t, ax, "r-", label="actual x")
        p2.grid(True)
        p2.legend()
        p2.set_xlabel("t [s]")
        p2.set_ylabel("x [m]")

        p3 = fig.add_subplot(2, 2, 3)
        p3.plot(t, ry, "b--", label="ref y")
        p3.plot(t, ay, "r-", label="actual y")
        p3.grid(True)
        p3.legend()
        p3.set_xlabel("t [s]")
        p3.set_ylabel("y [m]")

        p4 = fig.add_subplot(2, 2, 4)
        p4.plot(t, rz, "b--", label="ref z")
        p4.plot(t, az, "r-", label="actual z")
        p4.grid(True)
        p4.legend()
        p4.set_xlabel("t [s]")
        p4.set_ylabel("z [m]")

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save(self):
        if self.saved:
            return
        self.saved = True
        if not self.ref_samples and not self.actual_samples:
            rospy.loginfo("[poly_traj_test_pub] No comparison samples captured.")
            return
        out_dir = self._resolve_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        samples_csv = os.path.join(out_dir, "traj_track_samples.csv")
        fig_png = os.path.join(out_dir, "compare_plot.png")
        self._write_csv(
            samples_csv,
            [
                "sample_type",
                "t_s",
                "ref_x",
                "ref_y",
                "ref_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "pos_err_x",
                "pos_err_y",
                "pos_err_z",
                "pos_err_xy",
                "pos_err_xyz",
            ],
            self._combined_rows(),
        )
        if self.actual_samples:
            self._plot(fig_png)
        rospy.loginfo("[poly_traj_test_pub] Compare outputs saved to %s", out_dir)


def main():
    rospy.init_node("poly_traj_test_pub")

    topic = rospy.get_param("~topic", "/drone_0_planning/trajectory")
    start_delay = rospy.get_param("~start_delay", 1.0)
    cruise_speed = rospy.get_param("~cruise_speed", 0.8)
    min_duration = rospy.get_param("~min_duration", 1.0)
    num_segments = rospy.get_param("~num_segments", 7)
    pub_hz = rospy.get_param("~pub_hz", 1.0)
    publish_once = rospy.get_param("~publish_once", True)
    hold_node_alive = rospy.get_param("~hold_node_alive", True)
    traj_id = rospy.get_param("~traj_id", 1)
    enable_compare = rospy.get_param("~enable_compare", True)
    time_wait_timeout = float(rospy.get_param("~time_wait_timeout", 5.0))

    if not _wait_for_valid_ros_time(time_wait_timeout):
        return

    pub = rospy.Publisher(topic, PolyTraj, queue_size=10, latch=True)
    msg = make_multi_segment_msg(
        traj_id=traj_id,
        start_delay=start_delay,
        cruise_speed=cruise_speed,
        min_duration=min_duration,
        num_segments=num_segments,
    )

    rospy.loginfo("[poly_traj_test_pub] Publishing to %s", topic)
    rospy.loginfo(
        "[poly_traj_test_pub] Params: start_delay=%.2f, speed=%.2f, min_duration=%.2f, num_segments=%d, pub_hz=%.2f, publish_once=%s",
        start_delay,
        cruise_speed,
        min_duration,
        int(num_segments),
        pub_hz,
        str(publish_once),
    )
    rospy.loginfo(
        "[poly_traj_test_pub] Segments=%d, coef_len_per_axis=%d",
        len(msg.duration),
        len(msg.coef_x),
    )

    recorder = CompareRecorder(msg) if enable_compare else None
    if recorder is None:
        rospy.loginfo("[poly_traj_test_pub] Compare logger disabled.")

    if publish_once:
        pub.publish(msg)
        rospy.loginfo(
            "[poly_traj_test_pub] Sent traj_id=%d start=%.3f total_dur=%.2f order=%d",
            msg.traj_id,
            msg.start_time.to_sec(),
            sum(msg.duration),
            msg.order,
        )
        if hold_node_alive:
            rospy.spin()
        elif recorder is not None:
            rospy.sleep(min(sum(msg.duration) + 2.0, recorder.max_follow_time))
            recorder.save()
        return

    rate = rospy.Rate(pub_hz)
    while not rospy.is_shutdown():
        pub.publish(msg)
        rospy.loginfo(
            "[poly_traj_test_pub] Re-publish same traj_id=%d start=%.3f total_dur=%.2f",
            msg.traj_id,
            msg.start_time.to_sec(),
            sum(msg.duration),
        )
        rate.sleep()


if __name__ == "__main__":
    main()
