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


def _metrics(err_x, err_y, err_z):
    xy = math.sqrt(err_x * err_x + err_y * err_y)
    xyz = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
    return xy, abs(err_z), xyz


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
        self.output_root = rospy.get_param("~output_root", os.path.join(self.package_dir, "logs"))
        self.actual_topic = rospy.get_param(
            "~actual_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.gt_topic = rospy.get_param("~gt_topic", "")
        self.gt_max_dt = float(rospy.get_param("~gt_max_dt", 0.05))
        self.print_period = float(rospy.get_param("~print_period", 1.0))
        self.autosave_period = float(rospy.get_param("~autosave_period", 0.5))
        self.start_trigger = rospy.get_param("~start_trigger", "first_odom")
        self.motion_threshold = float(rospy.get_param("~motion_threshold", 0.05))
        self.use_text_reference = bool(rospy.get_param("~use_text_reference", True))
        self.ref_time_offset = float(rospy.get_param("~ref_time_offset", 0.0))
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))

        self.params_data = {}
        self.controller_param_snapshot = {}
        self.ref_dt = None
        self.ref_points = []
        self.ref_times = []
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

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.output_root, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.actual_csv_path = os.path.join(self.out_dir, "actual_path.csv")
        self.gt_csv_path = os.path.join(self.out_dir, "actual_vs_gt.csv")
        self.ref_csv_path = os.path.join(self.out_dir, "actual_vs_txt_ref.csv")
        self.summary_txt_path = os.path.join(self.out_dir, "summary.txt")
        self.fig_png_path = os.path.join(self.out_dir, "flight_path_3d_and_errors.png")

        rospy.Subscriber(self.actual_topic, Odometry, self._actual_cb, queue_size=300)
        if self.gt_topic:
            rospy.Subscriber(self.gt_topic, Odometry, self._gt_cb, queue_size=300)
        if self.save_on_shutdown:
            rospy.on_shutdown(self.save)

        rospy.loginfo(
            "[flight_path_error_logger] Node started. actual_topic=%s gt_topic=%s params_yaml=%s start_trigger=%s",
            self.actual_topic,
            self.gt_topic if self.gt_topic else "<disabled>",
            self.params_yaml,
            self.start_trigger,
        )
        rospy.loginfo("[flight_path_error_logger] This run will save under %s", self.out_dir)
        if self.ref_points:
            rospy.loginfo(
                "[flight_path_error_logger] Loaded text reference: %d samples dt=%.4f file=%s",
                len(self.ref_points),
                self.ref_dt,
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
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                t = len(self.ref_points) * self.ref_dt
                self.ref_points.append((t, x, y, z))
        self.ref_times = [p[0] for p in self.ref_points]

    def _set_record_start(self, stamp, reason):
        if self.record_start_time is not None:
            return
        self.record_start_time = stamp
        self.record_start_reason = reason
        human_time = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
        rospy.loginfo(
            "[flight_path_error_logger] Recording started at %.3f (%s), reason=%s",
            stamp,
            human_time,
            reason,
        )

    def _gt_cb(self, msg):
        stamp = _safe_stamp(msg)
        p = msg.pose.pose.position
        self.gt_buffer.append((stamp, p.x, p.y, p.z))

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

    def _interp_ref(self, rel_t):
        if not self.ref_points:
            return None
        if rel_t <= self.ref_points[0][0]:
            return self.ref_points[0]
        if rel_t >= self.ref_points[-1][0]:
            return self.ref_points[-1]

        idx = bisect_left(self.ref_times, rel_t)
        if idx <= 0:
            return self.ref_points[0]
        if idx >= len(self.ref_points):
            return self.ref_points[-1]

        t0, x0, y0, z0 = self.ref_points[idx - 1]
        t1, x1, y1, z1 = self.ref_points[idx]
        if abs(t1 - t0) < 1.0e-9:
            return self.ref_points[idx]
        ratio = (rel_t - t0) / (t1 - t0)
        x = x0 + ratio * (x1 - x0)
        y = y0 + ratio * (y1 - y0)
        z = z0 + ratio * (z1 - z0)
        return rel_t, x, y, z

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

    def _actual_cb(self, msg):
        stamp = _safe_stamp(msg)
        p = msg.pose.pose.position
        actual_xyz = (p.x, p.y, p.z)

        self._maybe_start_recording(stamp, actual_xyz)
        if self.record_start_time is None:
            return

        rel_t = stamp - self.record_start_time
        self.actual_samples.append((stamp, rel_t, p.x, p.y, p.z))

        gt_match = self._find_gt_match(stamp) if self.gt_topic else None
        if gt_match is not None:
            err_x = p.x - gt_match[1]
            err_y = p.y - gt_match[2]
            err_z = p.z - gt_match[3]
            xy_err, z_err, xyz_err = _metrics(err_x, err_y, err_z)
            self.gt_matches.append(
                (stamp, rel_t, p.x, p.y, p.z, gt_match[1], gt_match[2], gt_match[3], err_x, err_y, err_z, xy_err, z_err, xyz_err)
            )

        ref_match = None
        if self.ref_points:
            ref_match = self._interp_ref(rel_t - self.ref_time_offset)
        if ref_match is not None:
            err_x = p.x - ref_match[1]
            err_y = p.y - ref_match[2]
            err_z = p.z - ref_match[3]
            xy_err, z_err, xyz_err = _metrics(err_x, err_y, err_z)
            self.ref_matches.append(
                (stamp, rel_t, p.x, p.y, p.z, ref_match[1], ref_match[2], ref_match[3], err_x, err_y, err_z, xy_err, z_err, xyz_err)
            )

        if stamp - self.last_print_time >= self.print_period:
            self.last_print_time = stamp
            self._print_live_status()
        if self.autosave_period > 0.0 and stamp - self.last_autosave_time >= self.autosave_period:
            self.last_autosave_time = stamp
            self._save_snapshot(final=False)

    def _print_live_status(self):
        msg = "[flight_path_error_logger] samples=%d" % len(self.actual_samples)
        if self.gt_matches:
            row = self.gt_matches[-1]
            msg += " | GT xy=%.3f z=%.3f xyz=%.3f" % (row[11], row[12], row[13])
        if self.ref_matches:
            row = self.ref_matches[-1]
            msg += " | REF xy=%.3f z=%.3f xyz=%.3f" % (row[11], row[12], row[13])
        rospy.loginfo(msg)

    def _write_csv(self, path, header, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def _plot_3d_and_error(self, path):
        if not self.actual_samples:
            return

        fig = plt.figure(figsize=(14, 6))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")

        ax3d.plot(
            [r[2] for r in self.actual_samples],
            [r[3] for r in self.actual_samples],
            [r[4] for r in self.actual_samples],
            label="actual",
            linewidth=2.0,
        )

        if self.gt_matches:
            ax3d.plot(
                [r[5] for r in self.gt_matches],
                [r[6] for r in self.gt_matches],
                [r[7] for r in self.gt_matches],
                label="gt",
                linewidth=1.8,
            )

        if self.ref_matches:
            ax3d.plot(
                [r[5] for r in self.ref_matches],
                [r[6] for r in self.ref_matches],
                [r[7] for r in self.ref_matches],
                label="txt_ref",
                linewidth=1.8,
                linestyle="--",
            )

        ax3d.set_title("3D Flight Path")
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        if self.gt_matches:
            ax2.plot([r[1] for r in self.gt_matches], [r[11] for r in self.gt_matches], label="GT xy err")
            ax2.plot([r[1] for r in self.gt_matches], [r[12] for r in self.gt_matches], label="GT z err")
        if self.ref_matches:
            ax2.plot([r[1] for r in self.ref_matches], [r[11] for r in self.ref_matches], "--", label="REF xy err")
            ax2.plot([r[1] for r in self.ref_matches], [r[12] for r in self.ref_matches], "--", label="REF z err")
        ax2.set_title("Tracking Error")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("error [m]")
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _save_summary(self, path):
        gt_xy_stats = _series_stats([r[11] for r in self.gt_matches])
        gt_z_stats = _series_stats([r[12] for r in self.gt_matches])
        gt_xyz_stats = _series_stats([r[13] for r in self.gt_matches])
        ref_xy_stats = _series_stats([r[11] for r in self.ref_matches])
        ref_z_stats = _series_stats([r[12] for r in self.ref_matches])
        ref_xyz_stats = _series_stats([r[13] for r in self.ref_matches])

        lines = []
        lines.append("flight_path_error_logger summary")
        lines.append("")
        lines.append("Run info")
        lines.append("node_start_time: %.6f" % self.node_start_wall_time)
        lines.append("record_start_time: %s" % ("%.6f" % self.record_start_time if self.record_start_time is not None else "not_started"))
        lines.append("record_start_reason: %s" % (self.record_start_reason or "n/a"))
        lines.append("actual_topic: %s" % self.actual_topic)
        lines.append("gt_topic: %s" % (self.gt_topic if self.gt_topic else "disabled"))
        lines.append("params_yaml: %s" % self.params_yaml)
        lines.append("ref_time_offset: %.6f" % self.ref_time_offset)
        lines.append("actual_sample_count: %d" % len(self.actual_samples))
        lines.append("gt_match_count: %d" % len(self.gt_matches))
        lines.append("txt_ref_match_count: %d" % len(self.ref_matches))
        lines.append("")
        lines.append("Controller params from params.yaml")
        for key in sorted(self.controller_param_snapshot.keys()):
            lines.append("%s: %s" % (key, self.controller_param_snapshot[key]))
        lines.append("")
        lines.extend(self._format_stats_block("GT error stats", gt_xy_stats, gt_z_stats, gt_xyz_stats))
        lines.append("")
        lines.extend(self._format_stats_block("TXT reference error stats", ref_xy_stats, ref_z_stats, ref_xyz_stats))

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _format_stats_block(self, title, xy_stats, z_stats, xyz_stats):
        lines = [title]
        if xy_stats is None:
            lines.append("no data")
            return lines
        lines.append(
            "xy_error_m: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (xy_stats["mean"], xy_stats["rmse"], xy_stats["max"], xy_stats["count"])
        )
        lines.append(
            "z_error_m: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (z_stats["mean"], z_stats["rmse"], z_stats["max"], z_stats["count"])
        )
        lines.append(
            "xyz_error_m: mean=%.4f rmse=%.4f max=%.4f count=%d"
            % (xyz_stats["mean"], xyz_stats["rmse"], xyz_stats["max"], xyz_stats["count"])
        )
        return lines

    def _save_snapshot(self, final):
        if not self.actual_samples and not self.gt_matches and not self.ref_matches:
            if final:
                rospy.loginfo("[flight_path_error_logger] No samples recorded. Nothing to save.")
            return

        self._write_csv(
            self.actual_csv_path,
            ["stamp_s", "t_rel_s", "actual_x", "actual_y", "actual_z"],
            self.actual_samples,
        )
        if self.gt_matches:
            self._write_csv(
                self.gt_csv_path,
                [
                    "stamp_s", "t_rel_s",
                    "actual_x", "actual_y", "actual_z",
                    "gt_x", "gt_y", "gt_z",
                    "err_x", "err_y", "err_z",
                    "err_xy", "err_z_abs", "err_xyz",
                ],
                self.gt_matches,
            )
        if self.ref_matches:
            self._write_csv(
                self.ref_csv_path,
                [
                    "stamp_s", "t_rel_s",
                    "actual_x", "actual_y", "actual_z",
                    "ref_x", "ref_y", "ref_z",
                    "err_x", "err_y", "err_z",
                    "err_xy", "err_z_abs", "err_xyz",
                ],
                self.ref_matches,
            )

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
