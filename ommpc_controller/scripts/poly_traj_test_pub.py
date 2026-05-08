#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Odometry
from traj_utils.msg import PolyTraj


def _v_norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _falling_factorial(power, count):
    value = 1.0
    for k in range(count):
        value *= power - k
    return value


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


def _eval_low_order_derivative(coeffs, tau):
    return sum(i * c * (tau ** (i - 1)) for i, c in enumerate(coeffs) if i > 0)


def _derivative_basis(order, derivative_order, t):
    basis = []
    for power in range(order + 1):
        if power < derivative_order:
            basis.append(0.0)
        else:
            basis.append(_falling_factorial(power, derivative_order) * (t ** (power - derivative_order)))
    return basis


def _solve_linear_system(matrix, rhs):
    n = len(rhs)
    aug = [list(map(float, matrix[i])) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            raise ValueError("minimum-snap KKT system is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1.0e-15:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _single_axis_min_snap_normalized_coeffs(p0, pm, p1, mid_fraction):
    """
    Minimize integral snap^2 for q(tau), tau in [0, 1], subject to:
    q(0)=p0, q'(0)=0, q''(0)=0, q(mid_fraction)=pm,
    q(1)=p1, q'(1)=0, q''(1)=0.
    Coefficients are returned low-order first.
    """
    if mid_fraction <= 1.0e-6 or mid_fraction >= 1.0 - 1.0e-6:
        raise ValueError("mid_fraction is too close to an endpoint")

    order = 7
    num_coeff = order + 1
    snap_order = 4

    hessian = [[0.0 for _ in range(num_coeff)] for _ in range(num_coeff)]
    for i in range(snap_order, num_coeff):
        di = _falling_factorial(i, snap_order)
        for j in range(snap_order, num_coeff):
            dj = _falling_factorial(j, snap_order)
            hessian[i][j] = 2.0 * di * dj / (i + j - 2 * snap_order + 1)

    constraints = [
        (_derivative_basis(order, 0, 0.0), p0),
        (_derivative_basis(order, 1, 0.0), 0.0),
        (_derivative_basis(order, 2, 0.0), 0.0),
        (_derivative_basis(order, 0, mid_fraction), pm),
        (_derivative_basis(order, 0, 1.0), p1),
        (_derivative_basis(order, 1, 1.0), 0.0),
        (_derivative_basis(order, 2, 1.0), 0.0),
    ]
    num_constraints = len(constraints)

    kkt = []
    rhs = []
    for i in range(num_coeff):
        kkt.append(hessian[i] + [row[i] for row, _ in constraints])
        rhs.append(0.0)
    for row, value in constraints:
        kkt.append(row + [0.0] * num_constraints)
        rhs.append(value)

    return _solve_linear_system(kkt, rhs)[:num_coeff]


def _single_axis_min_snap_seventh_coeffs(p0, pm, p1, mid_fraction, duration):
    """
    Build a single 7th-order minimum-snap p(t) with zero start/end
    velocity and acceleration, and p(mid_fraction * duration) = pm.
    Coefficients are returned high-order first for PolyTraj.
    """
    normalized = _single_axis_min_snap_normalized_coeffs(p0, pm, p1, mid_fraction)
    return [normalized[i] / (duration ** i) for i in range(7, -1, -1)]


def _build_duration(start, mid, end, mid_fraction, cruise_speed, min_duration):
    normalized_coeffs = [
        _single_axis_min_snap_normalized_coeffs(start[i], mid[i], end[i], mid_fraction)
        for i in range(3)
    ]
    max_norm_speed = 0.0
    for k in range(1001):
        tau = k / 1000.0
        vx = _eval_low_order_derivative(normalized_coeffs[0], tau)
        vy = _eval_low_order_derivative(normalized_coeffs[1], tau)
        vz = _eval_low_order_derivative(normalized_coeffs[2], tau)
        max_norm_speed = max(max_norm_speed, _v_norm((vx, vy, vz)))
    return max(max_norm_speed / max(float(cruise_speed), 0.05), float(min_duration))


def make_single_segment_msg(traj_id, start_delay, cruise_speed, min_duration):
    msg = PolyTraj()
    msg.drone_id = 0
    msg.traj_id = traj_id
    msg.start_time = rospy.Time.now() + rospy.Duration.from_sec(start_delay)
    msg.order = 7

    start = (0.0, 0.0, 1.0)
    mid = (5.0, 0.5, 1.2)
    end = (10.0, 0.0, 1.0)
    mid_fraction = 0.5
    duration = _build_duration(start, mid, end, mid_fraction, cruise_speed, min_duration)

    # PolyTraj convention in this repo:
    # p(t) = c0*t^order + c1*t^(order-1) + ... + c(order).
    msg.duration = [duration]
    msg.coef_x = _single_axis_min_snap_seventh_coeffs(start[0], mid[0], end[0], mid_fraction, duration)
    msg.coef_y = _single_axis_min_snap_seventh_coeffs(start[1], mid[1], end[1], mid_fraction, duration)
    msg.coef_z = _single_axis_min_snap_seventh_coeffs(start[2], mid[2], end[2], mid_fraction, duration)
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
        v = odom.twist.twist.linear
        speed = _v_norm((v.x, v.y, v.z))
        self.actual_samples.append([
            rel_t,
            ref_xyz[0],
            ref_xyz[1],
            ref_xyz[2],
            p.x,
            p.y,
            p.z,
            speed,
        ])

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
                    r[7],
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
        actual_speed = [r[7] for r in self.actual_samples]

        fig = plt.figure(figsize=(12, 8))
        p1 = fig.add_subplot(2, 2, 1)
        p1.plot(rx, ry, "b-", label="reference")
        speed_artist = None
        if len(ax) >= 2:
            segments = [
                [(ax[i], ay[i]), (ax[i + 1], ay[i + 1])]
                for i in range(len(ax) - 1)
            ]
            segment_speed = [
                0.5 * (actual_speed[i] + actual_speed[i + 1])
                for i in range(len(actual_speed) - 1)
            ]
            speed_max = max(segment_speed)
            if speed_max <= 1.0e-9:
                speed_max = 1.0
            speed_artist = LineCollection(segments, cmap="plasma", linewidths=2.0)
            speed_artist.set_array(segment_speed)
            speed_artist.set_clim(0.0, speed_max)
            p1.add_collection(speed_artist)
            p1.plot([], [], color="tab:orange", linewidth=2.0, label="actual speed")
        else:
            speed_max = max(actual_speed)
            if speed_max <= 1.0e-9:
                speed_max = 1.0
            speed_artist = p1.scatter(
                ax,
                ay,
                c=actual_speed,
                cmap="plasma",
                s=20,
                label="actual speed",
                vmin=0.0,
                vmax=speed_max,
            )
        if speed_artist is not None:
            cbar = fig.colorbar(speed_artist, ax=p1, pad=0.02)
            cbar.set_label("actual speed [m/s]")
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
                "actual_speed",
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
    cruise_speed = rospy.get_param("~cruise_speed", 5.0)
    min_duration = rospy.get_param("~min_duration", 1.0)
    pub_hz = rospy.get_param("~pub_hz", 1.0)
    publish_once = rospy.get_param("~publish_once", True)
    hold_node_alive = rospy.get_param("~hold_node_alive", True)
    traj_id = rospy.get_param("~traj_id", 1)
    enable_compare = rospy.get_param("~enable_compare", True)
    time_wait_timeout = float(rospy.get_param("~time_wait_timeout", 5.0))

    if not _wait_for_valid_ros_time(time_wait_timeout):
        return

    pub = rospy.Publisher(topic, PolyTraj, queue_size=10, latch=True)
    msg = make_single_segment_msg(
        traj_id=traj_id,
        start_delay=start_delay,
        cruise_speed=cruise_speed,
        min_duration=min_duration,
    )

    rospy.loginfo("[poly_traj_test_pub] Publishing to %s", topic)
    rospy.loginfo(
        "[poly_traj_test_pub] Params: start_delay=%.2f, speed=%.2f, min_duration=%.2f, pub_hz=%.2f, publish_once=%s",
        start_delay,
        cruise_speed,
        min_duration,
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
