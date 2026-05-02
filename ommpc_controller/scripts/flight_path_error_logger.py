#!/usr/bin/env python3

import csv
import math
import os
from bisect import bisect_left
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import rospy
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Int32


def _safe_stamp(msg):
    stamp = msg.header.stamp.to_sec()
    if stamp > 1.0e-6:
        return stamp
    return rospy.Time.now().to_sec()


def _vec_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pos_metrics(err_x, err_y, err_z):
    err_xy = math.sqrt(err_x * err_x + err_y * err_y)
    err_xyz = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
    return err_xy, abs(err_z), err_xyz


def _vel_metrics(err_vx, err_vy, err_vz):
    err_vxy = math.sqrt(err_vx * err_vx + err_vy * err_vy)
    err_vxyz = math.sqrt(err_vx * err_vx + err_vy * err_vy + err_vz * err_vz)
    return err_vxy, abs(err_vz), err_vxyz


def _series_stats(values):
    if not values:
        return None
    n = float(len(values))
    mean_v = sum(values) / n
    rmse_v = math.sqrt(sum(v * v for v in values) / n)
    max_v = max(values)
    return {
        "count": int(n),
        "mean": mean_v,
        "rmse": rmse_v,
        "max": max_v,
    }


class FlightPathErrorLogger:
    def __init__(self):
        rospy.init_node("flight_path_error_logger")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.package_dir = os.path.dirname(self.script_dir)
        self.default_params_yaml = os.path.join(self.package_dir, "config", "params.yaml")

        self.params_yaml = rospy.get_param("~params_yaml", self.default_params_yaml)
        self.output_root = rospy.get_param(
            "~output_root",
            os.path.join(self.package_dir, "logs", "traj_track_step"),
        )
        self.shared_log_dir_param = rospy.get_param(
            "~shared_log_dir_param",
            "/ommpc_controller/traj_track_log_dir",
        )
        self.shared_log_run_id_param = rospy.get_param(
            "~shared_log_run_id_param",
            "/ommpc_controller/traj_track_run_id",
        )
        self.actual_topic = rospy.get_param(
            "~actual_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.gt_topic = rospy.get_param("~gt_topic", "")
        self.gt_max_dt = float(rospy.get_param("~gt_max_dt", 0.05))
        self.print_period = float(rospy.get_param("~print_period", 0.0))
        self.autosave_period = float(rospy.get_param("~autosave_period", 10.0))
        self.plot_on_autosave = bool(rospy.get_param("~plot_on_autosave", False))
        self.start_trigger = rospy.get_param("~start_trigger", "first_odom")
        self.motion_threshold = float(rospy.get_param("~motion_threshold", 0.05))
        self.use_text_reference = bool(rospy.get_param("~use_text_reference", True))
        self.ref_time_offset = float(rospy.get_param("~ref_time_offset", 0.0))
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))
        self.exec_state_topic = rospy.get_param("~exec_state_topic", "/ommpc_controller/exec_traj_state")
        self.points_tracking_start_topic = rospy.get_param(
            "~points_tracking_start_topic",
            "/ommpc_controller/points_tracking_start_time",
        )
        self.points_state_code = int(rospy.get_param("~points_state_code", 12))

        self.params_data = {}
        self.controller_param_snapshot = {}
        self.ref_dt = None
        self.ref_points = []
        self.ref_times = []
        self.ref_duration = None
        self._load_params_yaml()
        if self.use_text_reference:
            self._load_text_reference()

        self.gt_buffer = deque(maxlen=4000)
        self.actual_samples = []
        self.gt_matches = []
        self.ref_matches = []

        self.node_start_wall_time = rospy.Time.now().to_sec()
        self.record_start_time = None
        self.record_start_reason = ""
        self.first_actual_position = None
        self.last_print_time = 0.0
        self.final_saved = False
        self.last_autosave_time = 0.0

        self.current_exec_state = None
        self.latest_points_tracking_start = None
        self.tracking_active = False
        self.tracking_start_time = None
        self.tracking_stop_time = None

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.output_root, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        rospy.set_param(self.shared_log_dir_param, self.out_dir)
        rospy.set_param(self.shared_log_run_id_param, self.run_id)
        self.actual_csv_path = os.path.join(self.out_dir, "actual_path.csv")
        self.gt_csv_path = os.path.join(self.out_dir, "actual_vs_gt.csv")
        self.ref_csv_path = os.path.join(self.out_dir, "actual_vs_txt_ref.csv")
        self.summary_txt_path = os.path.join(self.out_dir, "summary.txt")
        self.fig_png_path = os.path.join(self.out_dir, "flight_path_3d_and_errors.png")

        rospy.Subscriber(self.actual_topic, Odometry, self._actual_cb, queue_size=300)
        if self.gt_topic:
            rospy.Subscriber(self.gt_topic, Odometry, self._gt_cb, queue_size=300)
        rospy.Subscriber(self.exec_state_topic, Int32, self._exec_state_cb, queue_size=20)
        rospy.Subscriber(self.points_tracking_start_topic, Float64, self._points_tracking_start_cb, queue_size=20)
        if self.save_on_shutdown:
            rospy.on_shutdown(self.save)

        rospy.loginfo(
            "[flight_path_error_logger] Node started. actual_topic=%s gt_topic=%s start_trigger=%s",
            self.actual_topic,
            self.gt_topic if self.gt_topic else "<disabled>",
            self.start_trigger,
        )
        rospy.loginfo(
            "[flight_path_error_logger] Tracking topics: exec_state=%s points_start=%s state_code=%d",
            self.exec_state_topic,
            self.points_tracking_start_topic,
            self.points_state_code,
        )
        rospy.loginfo("[flight_path_error_logger] This run will save under %s", self.out_dir)
        if self.ref_points:
            rospy.loginfo(
                "[flight_path_error_logger] Loaded text reference: %d samples dt=%.4f duration=%.3f file=%s",
                len(self.ref_points),
                self.ref_dt,
                self.ref_duration if self.ref_duration is not None else 0.0,
                self._resolve_ref_path(),
            )
        else:
            rospy.loginfo("[flight_path_error_logger] Text reference disabled or unavailable.")

    def _load_params_yaml(self):
        if not os.path.exists(self.params_yaml):
            rospy.logwarn("[flight_path_error_logger] params.yaml not found: %s", self.params_yaml)
            return

        with open(self.params_yaml, "r") as f:
            self.params_data = yaml.safe_load(f) or {}

        mpc_params = self.params_data.get("MPC_params", {})
        self.controller_param_snapshot = {
            "hover_percentage": self.params_data.get("hover_percentage"),
            "ref_txt_enable": self.params_data.get("ref_txt", {}).get("enable"),
            "ref_txt_time_step": self.params_data.get("ref_txt", {}).get("time_step"),
            "ref_txt_ref_filename": self.params_data.get("ref_txt", {}).get("ref_filename"),
            "step_T": mpc_params.get("step_T"),
            "Q_pos_xy": mpc_params.get("Q_pos_xy"),
            "Q_pos_z": mpc_params.get("Q_pos_z"),
            "Q_velocity": mpc_params.get("Q_velocity"),
            "Q_attitude_rp": mpc_params.get("Q_attitude_rp"),
            "Q_attitude_yaw": mpc_params.get("Q_attitude_yaw"),
            "R_thrust": mpc_params.get("R_thrust"),
            "R_pitchroll": mpc_params.get("R_pitchroll"),
            "R_yaw": mpc_params.get("R_yaw"),
            "state_cost_exponential": mpc_params.get("state_cost_exponential"),
            "input_cost_exponential": mpc_params.get("input_cost_exponential"),
            "max_bodyrate_xy": mpc_params.get("max_bodyrate_xy"),
            "max_bodyrate_z": mpc_params.get("max_bodyrate_z"),
            "min_thrust": mpc_params.get("min_thrust"),
            "max_thrust": mpc_params.get("max_thrust"),
        }

    def _resolve_ref_path(self):
        ref_txt = self.params_data.get("ref_txt", {})
        ref_filename = ref_txt.get("ref_filename")
        if not ref_filename:
            return None
        ref_filename = str(ref_filename)
        if os.path.isabs(ref_filename) and os.path.exists(ref_filename):
            return ref_filename
        return os.path.normpath(os.path.join(self.package_dir, ref_filename.lstrip("/")))

    def _load_text_reference(self):
        ref_txt = self.params_data.get("ref_txt", {})
        if not ref_txt.get("enable", False):
            return

        self.ref_dt = float(ref_txt.get("time_step", 0.0) or 0.0)
        if self.ref_dt <= 0.0:
            rospy.logwarn("[flight_path_error_logger] Invalid ref_txt/time_step: %s", ref_txt.get("time_step"))
            return

        ref_path = self._resolve_ref_path()
        if not ref_path or not os.path.exists(ref_path):
            rospy.logwarn("[flight_path_error_logger] Reference txt not found: %s", ref_path)
            return

        with open(ref_path, "r") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                parts = text.split()
                if len(parts) < 3:
                    rospy.logwarn("[flight_path_error_logger] Skip malformed ref line %d: %s", idx + 1, text)
                    continue
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                vx = float(parts[3]) if len(parts) > 3 else 0.0
                vy = float(parts[4]) if len(parts) > 4 else 0.0
                vz = float(parts[5]) if len(parts) > 5 else 0.0
                t = len(self.ref_points) * self.ref_dt
                self.ref_points.append((t, x, y, z, vx, vy, vz))
        self.ref_times = [p[0] for p in self.ref_points]
        if self.ref_times:
            self.ref_duration = self.ref_times[-1]

    def _set_record_start(self, stamp, reason):
        if self.record_start_time is not None:
            return
        self.record_start_time = stamp
        self.record_start_reason = reason
        human_time = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
        rospy.loginfo(
            "[flight_path_error_logger] Flight recording started at %.3f (%s), reason=%s",
            stamp,
            human_time,
            reason,
        )

    def _set_tracking_start(self, stamp, reason):
        if self.tracking_active and self.tracking_start_time is not None and abs(self.tracking_start_time - stamp) < 1.0e-6:
            return
        self.tracking_start_time = stamp
        self.tracking_stop_time = None
        self.tracking_active = True
        human_time = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
        rospy.loginfo(
            "[flight_path_error_logger] POINTS tracking started at %.3f (%s), reason=%s",
            stamp,
            human_time,
            reason,
        )

    def _set_tracking_stop(self, stamp, reason):
        if not self.tracking_active:
            return
        self.tracking_active = False
        self.tracking_stop_time = stamp
        rospy.loginfo(
            "[flight_path_error_logger] POINTS tracking stopped at %.3f, reason=%s",
            stamp,
            reason,
        )

    def _gt_cb(self, msg):
        stamp = _safe_stamp(msg)
        p = msg.pose.pose.position
        self.gt_buffer.append((stamp, p.x, p.y, p.z))

    def _exec_state_cb(self, msg):
        self.current_exec_state = int(msg.data)

    def _points_tracking_start_cb(self, msg):
        self.latest_points_tracking_start = float(msg.data)

    def _find_gt_match(self, stamp):
        if not self.gt_buffer:
            return None
        best = None
        best_dt = None
        for sample in reversed(self.gt_buffer):
            dt = abs(sample[0] - stamp)
            if best_dt is None or dt < best_dt:
                best = sample
                best_dt = dt
            if sample[0] < stamp and dt > self.gt_max_dt:
                break
        if best is None or best_dt is None or best_dt > self.gt_max_dt:
            return None
        return best

    def _interp_ref(self, track_t):
        if not self.ref_points:
            return None
        if track_t <= self.ref_points[0][0]:
            t, x, y, z, vx, vy, vz = self.ref_points[0]
            return {
                "track_t_rel_s": track_t,
                "ref_index_float": 0.0,
                "ref_index_floor": 0,
                "ref_index_ceil": 0,
                "ref_alpha": 0.0,
                "ref_x": x,
                "ref_y": y,
                "ref_z": z,
                "ref_vx": vx,
                "ref_vy": vy,
                "ref_vz": vz,
            }
        if track_t >= self.ref_points[-1][0]:
            t, x, y, z, vx, vy, vz = self.ref_points[-1]
            last_idx = len(self.ref_points) - 1
            return {
                "track_t_rel_s": track_t,
                "ref_index_float": float(last_idx),
                "ref_index_floor": last_idx,
                "ref_index_ceil": last_idx,
                "ref_alpha": 0.0,
                "ref_x": x,
                "ref_y": y,
                "ref_z": z,
                "ref_vx": vx,
                "ref_vy": vy,
                "ref_vz": vz,
            }

        idx = bisect_left(self.ref_times, track_t)
        if idx <= 0:
            idx = 1

        t0, x0, y0, z0, vx0, vy0, vz0 = self.ref_points[idx - 1]
        t1, x1, y1, z1, vx1, vy1, vz1 = self.ref_points[idx]
        if abs(t1 - t0) < 1.0e-9:
            return {
                "track_t_rel_s": track_t,
                "ref_index_float": float(idx),
                "ref_index_floor": idx,
                "ref_index_ceil": idx,
                "ref_alpha": 0.0,
                "ref_x": x1,
                "ref_y": y1,
                "ref_z": z1,
                "ref_vx": vx1,
                "ref_vy": vy1,
                "ref_vz": vz1,
            }

        ratio = (track_t - t0) / (t1 - t0)
        ref_index_float = (idx - 1) + ratio
        return {
            "track_t_rel_s": track_t,
            "ref_index_float": ref_index_float,
            "ref_index_floor": idx - 1,
            "ref_index_ceil": idx,
            "ref_alpha": ratio,
            "ref_x": x0 + ratio * (x1 - x0),
            "ref_y": y0 + ratio * (y1 - y0),
            "ref_z": z0 + ratio * (z1 - z0),
            "ref_vx": vx0 + ratio * (vx1 - vx0),
            "ref_vy": vy0 + ratio * (vy1 - vy0),
            "ref_vz": vz0 + ratio * (vz1 - vz0),
        }

    def _should_start_on_motion(self, actual_xyz):
        if self.first_actual_position is None:
            self.first_actual_position = actual_xyz
            return False
        return _vec_dist(actual_xyz, self.first_actual_position) >= self.motion_threshold

    def _maybe_start_recording(self, stamp, actual_xyz):
        if self.record_start_time is not None:
            return
        if self.start_trigger == "node_start":
            self._set_record_start(self.node_start_wall_time, "node_start")
        elif self.start_trigger == "first_motion":
            if self._should_start_on_motion(actual_xyz):
                self._set_record_start(stamp, "first_motion")
        else:
            self._set_record_start(stamp, "first_odom")

    def _maybe_update_tracking_state(self, stamp):
        if self.current_exec_state == self.points_state_code and self.latest_points_tracking_start is not None:
            if (not self.tracking_active) or self.tracking_start_time is None or abs(self.tracking_start_time - self.latest_points_tracking_start) > 1.0e-6:
                self._set_tracking_start(self.latest_points_tracking_start, "controller_points_start")
        elif self.current_exec_state is not None and self.current_exec_state != self.points_state_code:
            self._set_tracking_stop(stamp, "controller_left_points")

    def _actual_cb(self, msg):
        stamp = _safe_stamp(msg)
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        actual_xyz = (p.x, p.y, p.z)

        self._maybe_start_recording(stamp, actual_xyz)
        if self.record_start_time is None:
            return

        self._maybe_update_tracking_state(stamp)

        flight_rel_t = stamp - self.record_start_time
        actual_row = {
            "stamp_s": stamp,
            "flight_t_rel_s": flight_rel_t,
            "actual_x": p.x,
            "actual_y": p.y,
            "actual_z": p.z,
            "actual_vx": v.x,
            "actual_vy": v.y,
            "actual_vz": v.z,
        }
        self.actual_samples.append(actual_row)

        gt_match = self._find_gt_match(stamp) if self.gt_topic else None
        if gt_match is not None:
            err_x = p.x - gt_match[1]
            err_y = p.y - gt_match[2]
            err_z = p.z - gt_match[3]
            err_xy, err_z_abs, err_xyz = _pos_metrics(err_x, err_y, err_z)
            self.gt_matches.append({
                "stamp_s": stamp,
                "flight_t_rel_s": flight_rel_t,
                "actual_x": p.x,
                "actual_y": p.y,
                "actual_z": p.z,
                "gt_x": gt_match[1],
                "gt_y": gt_match[2],
                "gt_z": gt_match[3],
                "pos_err_x": err_x,
                "pos_err_y": err_y,
                "pos_err_z": err_z,
                "pos_err_xy": err_xy,
                "pos_err_z_abs": err_z_abs,
                "pos_err_xyz": err_xyz,
            })

        if self.tracking_active and self.tracking_start_time is not None and self.ref_points:
            track_t = stamp - self.tracking_start_time - self.ref_time_offset
            if track_t >= 0.0:
                ref_match = self._interp_ref(track_t)
                if ref_match is not None:
                    pos_err_x = p.x - ref_match["ref_x"]
                    pos_err_y = p.y - ref_match["ref_y"]
                    pos_err_z = p.z - ref_match["ref_z"]
                    pos_err_xy, pos_err_z_abs, pos_err_xyz = _pos_metrics(pos_err_x, pos_err_y, pos_err_z)

                    vel_err_x = v.x - ref_match["ref_vx"]
                    vel_err_y = v.y - ref_match["ref_vy"]
                    vel_err_z = v.z - ref_match["ref_vz"]
                    vel_err_xy, vel_err_z_abs, vel_err_xyz = _vel_metrics(vel_err_x, vel_err_y, vel_err_z)

                    self.ref_matches.append({
                        "stamp_s": stamp,
                        "flight_t_rel_s": flight_rel_t,
                        "track_t_rel_s": ref_match["track_t_rel_s"],
                        "ref_index_float": ref_match["ref_index_float"],
                        "ref_index_floor": ref_match["ref_index_floor"],
                        "ref_index_ceil": ref_match["ref_index_ceil"],
                        "ref_alpha": ref_match["ref_alpha"],
                        "actual_x": p.x,
                        "actual_y": p.y,
                        "actual_z": p.z,
                        "actual_vx": v.x,
                        "actual_vy": v.y,
                        "actual_vz": v.z,
                        "ref_x": ref_match["ref_x"],
                        "ref_y": ref_match["ref_y"],
                        "ref_z": ref_match["ref_z"],
                        "ref_vx": ref_match["ref_vx"],
                        "ref_vy": ref_match["ref_vy"],
                        "ref_vz": ref_match["ref_vz"],
                        "pos_err_x": pos_err_x,
                        "pos_err_y": pos_err_y,
                        "pos_err_z": pos_err_z,
                        "pos_err_xy": pos_err_xy,
                        "pos_err_z_abs": pos_err_z_abs,
                        "pos_err_xyz": pos_err_xyz,
                        "vel_err_x": vel_err_x,
                        "vel_err_y": vel_err_y,
                        "vel_err_z": vel_err_z,
                        "vel_err_xy": vel_err_xy,
                        "vel_err_z_abs": vel_err_z_abs,
                        "vel_err_xyz": vel_err_xyz,
                    })

        if self.print_period > 0.0 and stamp - self.last_print_time >= self.print_period:
            self.last_print_time = stamp
            self._print_live_status()
        if self.autosave_period > 0.0 and stamp - self.last_autosave_time >= self.autosave_period:
            self.last_autosave_time = stamp
            self._save_snapshot(final=False)

    def _print_live_status(self):
        msg = "[flight_path_error_logger] flight_samples=%d tracking_samples=%d" % (
            len(self.actual_samples),
            len(self.ref_matches),
        )
        if self.ref_matches:
            row = self.ref_matches[-1]
            msg += " | REF pos_xy=%.3f pos_z=%.3f vel_xy=%.3f vel_z=%.3f" % (
                row["pos_err_xy"],
                row["pos_err_z_abs"],
                row["vel_err_xy"],
                row["vel_err_z_abs"],
            )
        rospy.loginfo(msg)

    def _write_csv(self, path, rows, header):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def _plot_3d_and_error(self, path):
        if not self.actual_samples:
            return

        fig = plt.figure(figsize=(18, 6))

        ax3d = fig.add_subplot(1, 3, 1, projection="3d")
        ax3d.plot(
            [r["actual_x"] for r in self.actual_samples],
            [r["actual_y"] for r in self.actual_samples],
            [r["actual_z"] for r in self.actual_samples],
            label="actual",
            linewidth=2.0,
        )
        if self.ref_matches:
            ax3d.plot(
                [r["ref_x"] for r in self.ref_matches],
                [r["ref_y"] for r in self.ref_matches],
                [r["ref_z"] for r in self.ref_matches],
                label="txt_ref",
                linewidth=1.8,
                linestyle="--",
            )
        ax3d.set_title("3D Flight Path")
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.legend()

        ax_pos = fig.add_subplot(1, 3, 2)
        if self.ref_matches:
            ax_pos.plot(
                [r["track_t_rel_s"] for r in self.ref_matches],
                [r["pos_err_xy"] for r in self.ref_matches],
                "--",
                label="REF xy err",
            )
            ax_pos.plot(
                [r["track_t_rel_s"] for r in self.ref_matches],
                [r["pos_err_z_abs"] for r in self.ref_matches],
                "--",
                label="REF z err",
            )
        ax_pos.set_title("Position Error")
        ax_pos.set_xlabel("tracking t [s]")
        ax_pos.set_ylabel("error [m]")
        ax_pos.grid(True)
        ax_pos.legend()

        ax_vel = fig.add_subplot(1, 3, 3)
        if self.ref_matches:
            ax_vel.plot(
                [r["track_t_rel_s"] for r in self.ref_matches],
                [r["vel_err_xy"] for r in self.ref_matches],
                "--",
                label="REF vel xy err",
            )
            ax_vel.plot(
                [r["track_t_rel_s"] for r in self.ref_matches],
                [r["vel_err_z_abs"] for r in self.ref_matches],
                "--",
                label="REF vel z err",
            )
        ax_vel.set_title("Velocity Error")
        ax_vel.set_xlabel("tracking t [s]")
        ax_vel.set_ylabel("error [m/s]")
        ax_vel.grid(True)
        ax_vel.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _save_summary(self, path):
        gt_pos_xy_stats = _series_stats([r["pos_err_xy"] for r in self.gt_matches])
        gt_pos_z_stats = _series_stats([r["pos_err_z_abs"] for r in self.gt_matches])
        gt_pos_xyz_stats = _series_stats([r["pos_err_xyz"] for r in self.gt_matches])

        ref_pos_xy_stats = _series_stats([r["pos_err_xy"] for r in self.ref_matches])
        ref_pos_z_stats = _series_stats([r["pos_err_z_abs"] for r in self.ref_matches])
        ref_pos_xyz_stats = _series_stats([r["pos_err_xyz"] for r in self.ref_matches])
        ref_vel_xy_stats = _series_stats([r["vel_err_xy"] for r in self.ref_matches])
        ref_vel_z_stats = _series_stats([r["vel_err_z_abs"] for r in self.ref_matches])
        ref_vel_xyz_stats = _series_stats([r["vel_err_xyz"] for r in self.ref_matches])

        lines = []
        lines.append("flight_path_error_logger summary")
        lines.append("")
        lines.append("Run info")
        lines.append("node_start_time: %.6f" % self.node_start_wall_time)
        lines.append("record_start_time: %s" % ("%.6f" % self.record_start_time if self.record_start_time is not None else "not_started"))
        lines.append("record_start_reason: %s" % (self.record_start_reason or "n/a"))
        lines.append("tracking_start_time: %s" % ("%.6f" % self.tracking_start_time if self.tracking_start_time is not None else "not_started"))
        lines.append("tracking_stop_time: %s" % ("%.6f" % self.tracking_stop_time if self.tracking_stop_time is not None else "active_or_not_stopped"))
        lines.append("actual_topic: %s" % self.actual_topic)
        lines.append("gt_topic: %s" % (self.gt_topic if self.gt_topic else "disabled"))
        lines.append("exec_state_topic: %s" % self.exec_state_topic)
        lines.append("points_tracking_start_topic: %s" % self.points_tracking_start_topic)
        lines.append("params_yaml: %s" % self.params_yaml)
        lines.append("ref_time_offset: %.6f" % self.ref_time_offset)
        lines.append("ref_sample_count: %d" % len(self.ref_points))
        lines.append("ref_dt: %s" % ("%.6f" % self.ref_dt if self.ref_dt is not None else "n/a"))
        lines.append("ref_duration_s: %s" % ("%.6f" % self.ref_duration if self.ref_duration is not None else "n/a"))
        lines.append("actual_sample_count: %d" % len(self.actual_samples))
        lines.append("gt_match_count: %d" % len(self.gt_matches))
        lines.append("txt_tracking_match_count: %d" % len(self.ref_matches))
        lines.append("")
        lines.append("Controller params from params.yaml")
        for key in sorted(self.controller_param_snapshot.keys()):
            lines.append("%s: %s" % (key, self.controller_param_snapshot[key]))
        lines.append("")
        lines.extend(self._format_stats_block("GT position error stats", gt_pos_xy_stats, gt_pos_z_stats, gt_pos_xyz_stats, "m"))
        lines.append("")
        lines.extend(self._format_stats_block("TXT tracking position error stats", ref_pos_xy_stats, ref_pos_z_stats, ref_pos_xyz_stats, "m"))
        lines.append("")
        lines.extend(self._format_stats_block("TXT tracking velocity error stats", ref_vel_xy_stats, ref_vel_z_stats, ref_vel_xyz_stats, "m/s"))

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _format_stats_block(self, title, xy_stats, z_stats, xyz_stats, unit):
        lines = [title]
        if xy_stats is None:
            lines.append("no data")
            return lines
        lines.append(
            "xy_error_%s: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (unit, xy_stats["mean"], xy_stats["rmse"], xy_stats["max"], xy_stats["count"])
        )
        lines.append(
            "z_error_%s: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (unit, z_stats["mean"], z_stats["rmse"], z_stats["max"], z_stats["count"])
        )
        lines.append(
            "xyz_error_%s: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (unit, xyz_stats["mean"], xyz_stats["rmse"], xyz_stats["max"], xyz_stats["count"])
        )
        return lines

    def _save_snapshot(self, final):
        if not self.actual_samples and not self.gt_matches and not self.ref_matches:
            if final:
                rospy.loginfo("[flight_path_error_logger] No samples recorded. Nothing to save.")
            return

        self._write_csv(
            self.actual_csv_path,
            self.actual_samples,
            [
                "stamp_s",
                "flight_t_rel_s",
                "actual_x",
                "actual_y",
                "actual_z",
                "actual_vx",
                "actual_vy",
                "actual_vz",
            ],
        )

        if self.gt_matches:
            self._write_csv(
                self.gt_csv_path,
                self.gt_matches,
                [
                    "stamp_s",
                    "flight_t_rel_s",
                    "actual_x",
                    "actual_y",
                    "actual_z",
                    "gt_x",
                    "gt_y",
                    "gt_z",
                    "pos_err_x",
                    "pos_err_y",
                    "pos_err_z",
                    "pos_err_xy",
                    "pos_err_z_abs",
                    "pos_err_xyz",
                ],
            )

        if self.ref_matches:
            self._write_csv(
                self.ref_csv_path,
                self.ref_matches,
                [
                    "stamp_s",
                    "flight_t_rel_s",
                    "track_t_rel_s",
                    "ref_index_float",
                    "ref_index_floor",
                    "ref_index_ceil",
                    "ref_alpha",
                    "actual_x",
                    "actual_y",
                    "actual_z",
                    "actual_vx",
                    "actual_vy",
                    "actual_vz",
                    "ref_x",
                    "ref_y",
                    "ref_z",
                    "ref_vx",
                    "ref_vy",
                    "ref_vz",
                    "pos_err_x",
                    "pos_err_y",
                    "pos_err_z",
                    "pos_err_xy",
                    "pos_err_z_abs",
                    "pos_err_xyz",
                    "vel_err_x",
                    "vel_err_y",
                    "vel_err_z",
                    "vel_err_xy",
                    "vel_err_z_abs",
                    "vel_err_xyz",
                ],
            )

        if final or self.plot_on_autosave:
            self._plot_3d_and_error(self.fig_png_path)
        self._save_summary(self.summary_txt_path)
        if final:
            rospy.loginfo("[flight_path_error_logger] Final outputs saved to %s", self.out_dir)
        else:
            rospy.loginfo_throttle(5.0, "[flight_path_error_logger] Autosaving outputs to %s", self.out_dir)

    def save(self):
        if self.final_saved:
            return
        self.final_saved = True
        self._save_snapshot(final=True)

    def spin(self):
        rospy.spin()


def main():
    node = FlightPathErrorLogger()
    node.spin()


if __name__ == "__main__":
    main()
