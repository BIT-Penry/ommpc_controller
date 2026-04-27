#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


def _wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_diff(target, actual):
    return _wrap_pi(target - actual)


def _yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _series_stats(values):
    if not values:
        return None
    n = float(len(values))
    mean_v = sum(values) / n
    rmse_v = math.sqrt(sum(v * v for v in values) / n)
    max_abs_v = max(abs(v) for v in values)
    return {
        "count": int(n),
        "mean": mean_v,
        "rmse": rmse_v,
        "max_abs": max_abs_v,
    }


def _parse_targets_deg(value):
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


class HoverYawStepTest:
    def __init__(self):
        rospy.init_node("hover_yaw_step_test")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)

        self.yaw_topic = rospy.get_param("~yaw_topic", "/drone_0_planning/hover_yaw")
        self.odom_topic = rospy.get_param(
            "~odom_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.targets_deg = _parse_targets_deg(
            rospy.get_param("~targets_deg", [60, 120, 180, 240, 300, 360])
        )
        self.dwell_time = float(rospy.get_param("~dwell_time", 10.0))
        self.start_delay = float(rospy.get_param("~start_delay", 2.0))
        self.mode = str(rospy.get_param("~mode", "ramp")).strip().lower()
        if self.mode not in ("step", "ramp"):
            rospy.logwarn("[hover_yaw_step_test] Unknown mode=%s, falling back to ramp.", self.mode)
            self.mode = "ramp"
        self.pub_hz = float(rospy.get_param("~pub_hz", 20.0))
        self.max_yaw_rate_deg_s = float(rospy.get_param("~max_yaw_rate_deg_s", 15.0))
        self.command_repeat = int(rospy.get_param("~command_repeat", 3))
        self.command_repeat_dt = float(rospy.get_param("~command_repeat_dt", 0.1))
        self.settle_tail_s = float(rospy.get_param("~settle_tail_s", 1.0))
        self.settle_threshold_deg = float(rospy.get_param("~settle_threshold_deg", 5.0))
        self.output_root = rospy.get_param(
            "~output_root",
            os.path.join(package_dir, "logs", "hover_yaw_step"),
        )

        self.pub = rospy.Publisher(self.yaw_topic, Float64, queue_size=10, latch=True)
        self.samples = []
        self.commands = []
        self.latest_odom = None
        self.active_segment = -1
        self.active_target_rad = None
        self.active_target_deg = None
        self.active_goal_rad = None
        self.active_goal_deg = None
        self.active_ref_yaw_rate_rad_s = 0.0
        self.active_segment_start_stamp = None
        self.active_segment_start_pos = None
        self.ref_yaw_unwrapped_rad = None
        self.last_yaw = None
        self.last_yaw_stamp = None
        self.last_yaw_unwrapped = None
        self.saved = False

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.output_root, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.samples_csv_path = os.path.join(self.out_dir, "hover_yaw_samples.csv")
        self.commands_csv_path = os.path.join(self.out_dir, "hover_yaw_commands.csv")
        self.summary_path = os.path.join(self.out_dir, "summary.txt")
        self.fig_path = os.path.join(self.out_dir, "hover_yaw_step_response.png")

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=300)
        rospy.on_shutdown(self.save)

        rospy.loginfo("[hover_yaw_step_test] yaw_topic=%s", self.yaw_topic)
        rospy.loginfo("[hover_yaw_step_test] odom_topic=%s", self.odom_topic)
        rospy.loginfo(
            "[hover_yaw_step_test] targets_deg=%s mode=%s dwell=%.2fs max_yaw_rate=%.1fdeg/s output=%s",
            self.targets_deg,
            self.mode,
            self.dwell_time,
            self.max_yaw_rate_deg_s,
            self.out_dir,
        )

    def _odom_cb(self, msg):
        stamp = msg.header.stamp.to_sec()
        if stamp <= 1.0e-6:
            stamp = rospy.Time.now().to_sec()
        self.latest_odom = msg

        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        yaw_rate = 0.0
        if self.last_yaw is not None and self.last_yaw_stamp is not None:
            dt = stamp - self.last_yaw_stamp
            if dt > 1.0e-4:
                dyaw = _angle_diff(yaw, self.last_yaw)
                yaw_rate = dyaw / dt
                self.last_yaw_unwrapped += dyaw
        else:
            self.last_yaw_unwrapped = yaw

        self.last_yaw = yaw
        self.last_yaw_stamp = stamp

        if self.active_segment < 0 or self.active_target_rad is None:
            return

        p = msg.pose.pose.position
        start_pos = self.active_segment_start_pos or (p.x, p.y, p.z)
        dx = p.x - start_pos[0]
        dy = p.y - start_pos[1]
        dz = p.z - start_pos[2]
        drift_xy = math.sqrt(dx * dx + dy * dy)
        yaw_err = _angle_diff(self.active_target_rad, yaw)

        self.samples.append({
            "stamp_s": stamp,
            "test_t_s": stamp - self.test_start_stamp,
            "segment_t_s": stamp - self.active_segment_start_stamp,
            "segment_id": self.active_segment,
            "target_yaw_deg": self.active_target_deg,
            "target_yaw_rad": self.active_target_rad,
            "goal_yaw_deg": self.active_goal_deg,
            "goal_yaw_rad": self.active_goal_rad,
            "target_yaw_rate_deg_s": math.degrees(self.active_ref_yaw_rate_rad_s),
            "target_yaw_rate_rad_s": self.active_ref_yaw_rate_rad_s,
            "actual_yaw_deg": math.degrees(yaw),
            "actual_yaw_rad": yaw,
            "actual_yaw_unwrapped_deg": math.degrees(self.last_yaw_unwrapped),
            "yaw_error_deg": math.degrees(yaw_err),
            "yaw_error_rad": yaw_err,
            "actual_yaw_rate_deg_s": math.degrees(yaw_rate),
            "actual_yaw_rate_rad_s": yaw_rate,
            "yaw_rate_error_deg_s": math.degrees(self.active_ref_yaw_rate_rad_s - yaw_rate),
            "yaw_rate_error_rad_s": self.active_ref_yaw_rate_rad_s - yaw_rate,
            "actual_x": p.x,
            "actual_y": p.y,
            "actual_z": p.z,
            "drift_xy_m": drift_xy,
            "drift_z_m": dz,
        })

    def _wait_for_odom(self):
        rospy.loginfo("[hover_yaw_step_test] Waiting for odom...")
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and self.latest_odom is None:
            rate.sleep()
        rospy.loginfo("[hover_yaw_step_test] Odom ready.")

    def _publish_ref(self, segment_id, ref_rad, ref_rate_rad_s, publish_log=False):
        self.active_target_rad = ref_rad
        self.active_target_deg = math.degrees(ref_rad)
        self.active_ref_yaw_rate_rad_s = ref_rate_rad_s
        self.pub.publish(Float64(data=ref_rad))

        if publish_log:
            now = rospy.Time.now().to_sec()
            self.commands.append({
                "stamp_s": now,
                "test_t_s": now - self.test_start_stamp,
                "segment_id": int(segment_id),
                "mode": self.mode,
                "target_yaw_deg": self.active_target_deg,
                "target_yaw_rad": ref_rad,
                "target_yaw_rate_deg_s": math.degrees(ref_rate_rad_s),
                "target_yaw_rate_rad_s": ref_rate_rad_s,
                "goal_yaw_deg": self.active_goal_deg,
                "goal_yaw_rad": self.active_goal_rad,
                "start_x": self.active_segment_start_pos[0],
                "start_y": self.active_segment_start_pos[1],
                "start_z": self.active_segment_start_pos[2],
            })

    def _start_segment(self, segment_id, target_deg):
        target_rad = math.radians(target_deg)
        now = rospy.Time.now().to_sec()
        odom = self.latest_odom
        p = odom.pose.pose.position

        self.active_segment = int(segment_id)
        self.active_goal_deg = float(target_deg)
        self.active_goal_rad = target_rad
        self.active_segment_start_stamp = now
        self.active_segment_start_pos = (p.x, p.y, p.z)

        if self.ref_yaw_unwrapped_rad is None:
            self.ref_yaw_unwrapped_rad = self.last_yaw_unwrapped

        if self.mode == "step":
            self.ref_yaw_unwrapped_rad = target_rad
            self._publish_ref(segment_id, target_rad, 0.0, publish_log=True)
            msg = Float64(data=target_rad)
            for _ in range(max(0, self.command_repeat - 1)):
                rospy.sleep(self.command_repeat_dt)
                self.pub.publish(msg)
        else:
            self._publish_ref(segment_id, self.ref_yaw_unwrapped_rad, 0.0, publish_log=True)

        rospy.loginfo(
            "[hover_yaw_step_test] Segment %d goal %.1f deg started.",
            segment_id,
            target_deg,
        )

    def _run_ramp_segment(self, segment_id, target_deg):
        self._start_segment(segment_id, target_deg)
        rate = rospy.Rate(max(1.0, self.pub_hz))
        max_rate = math.radians(max(0.1, self.max_yaw_rate_deg_s))
        hold_end_time = None
        last_t = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt = max(1.0e-3, now - last_t)
            last_t = now

            remaining = self.active_goal_rad - self.ref_yaw_unwrapped_rad
            if abs(remaining) <= max_rate * dt:
                self.ref_yaw_unwrapped_rad = self.active_goal_rad
                ref_rate = 0.0
                if hold_end_time is None:
                    hold_end_time = now + self.dwell_time
            else:
                direction = 1.0 if remaining > 0.0 else -1.0
                self.ref_yaw_unwrapped_rad += direction * max_rate * dt
                ref_rate = direction * max_rate

            self._publish_ref(segment_id, self.ref_yaw_unwrapped_rad, ref_rate, publish_log=True)

            if hold_end_time is not None and now >= hold_end_time:
                break
            rate.sleep()

    def _run_step_segment(self, segment_id, target_deg):
        self._start_segment(segment_id, target_deg)
        end_time = rospy.Time.now().to_sec() + self.dwell_time
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < end_time:
            rate.sleep()

    def run(self):
        self._wait_for_odom()
        rospy.sleep(self.start_delay)
        self.test_start_stamp = rospy.Time.now().to_sec()
        self.ref_yaw_unwrapped_rad = self.last_yaw_unwrapped

        for idx, target_deg in enumerate(self.targets_deg):
            if rospy.is_shutdown():
                break
            if self.mode == "step":
                self._run_step_segment(idx, target_deg)
            else:
                self._run_ramp_segment(idx, target_deg)

        self.active_segment = -1
        self.active_target_rad = None
        self.active_target_deg = None
        self.active_goal_rad = None
        self.active_goal_deg = None
        self.active_ref_yaw_rate_rad_s = 0.0
        self.save()
        rospy.loginfo("[hover_yaw_step_test] Done.")

    def _write_csv(self, path, rows, header):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def _segment_rows(self, segment_id):
        return [r for r in self.samples if int(r["segment_id"]) == int(segment_id)]

    def _settling_time(self, rows):
        if not rows:
            return None
        threshold = self.settle_threshold_deg
        for i, row in enumerate(rows):
            tail = rows[i:]
            if tail and all(abs(r["yaw_error_deg"]) <= threshold for r in tail):
                return row["segment_t_s"]
        return None

    def _tail_rows(self, rows):
        if not rows:
            return []
        end_t = rows[-1]["segment_t_s"]
        start_t = max(0.0, end_t - self.settle_tail_s)
        return [r for r in rows if r["segment_t_s"] >= start_t]

    def _save_summary(self):
        lines = []
        lines.append("hover_yaw_step_test summary")
        lines.append("")
        lines.append("Run info")
        lines.append("yaw_topic: %s" % self.yaw_topic)
        lines.append("odom_topic: %s" % self.odom_topic)
        lines.append("targets_deg: %s" % ", ".join("%.1f" % v for v in self.targets_deg))
        lines.append("mode: %s" % self.mode)
        lines.append("pub_hz: %.3f" % self.pub_hz)
        lines.append("max_yaw_rate_deg_s: %.3f" % self.max_yaw_rate_deg_s)
        lines.append("dwell_time_s: %.3f" % self.dwell_time)
        lines.append("start_delay_s: %.3f" % self.start_delay)
        lines.append("settle_threshold_deg: %.3f" % self.settle_threshold_deg)
        lines.append("settle_tail_s: %.3f" % self.settle_tail_s)
        lines.append("sample_count: %d" % len(self.samples))
        lines.append("")
        lines.append("Per-segment stats")

        for idx, target_deg in enumerate(self.targets_deg):
            rows = self._segment_rows(idx)
            yaw_stats = _series_stats([r["yaw_error_deg"] for r in rows])
            yaw_rate_err_stats = _series_stats([r["yaw_rate_error_deg_s"] for r in rows])
            yaw_rate_stats = _series_stats([r["actual_yaw_rate_deg_s"] for r in rows])
            drift_xy_stats = _series_stats([r["drift_xy_m"] for r in rows])
            drift_z_stats = _series_stats([r["drift_z_m"] for r in rows])
            tail = self._tail_rows(rows)
            tail_abs_err = [abs(r["yaw_error_deg"]) for r in tail]
            settling_time = self._settling_time(rows)

            lines.append("")
            lines.append("segment_%d_target_deg: %.1f" % (idx, target_deg))
            lines.append("samples: %d" % len(rows))
            if yaw_stats is None:
                lines.append("no data")
                continue
            lines.append(
                "yaw_error_deg: mean=%.4f rmse=%.4f max_abs=%.4f"
                % (yaw_stats["mean"], yaw_stats["rmse"], yaw_stats["max_abs"])
            )
            lines.append(
                "tail_abs_yaw_error_deg: mean=%.4f max=%.4f"
                % (
                    (sum(tail_abs_err) / len(tail_abs_err)) if tail_abs_err else float("nan"),
                    max(tail_abs_err) if tail_abs_err else float("nan"),
                )
            )
            lines.append(
                "actual_yaw_rate_deg_s: mean=%.4f rmse=%.4f max_abs=%.4f"
                % (yaw_rate_stats["mean"], yaw_rate_stats["rmse"], yaw_rate_stats["max_abs"])
            )
            lines.append(
                "yaw_rate_error_deg_s: mean=%.4f rmse=%.4f max_abs=%.4f"
                % (yaw_rate_err_stats["mean"], yaw_rate_err_stats["rmse"], yaw_rate_err_stats["max_abs"])
            )
            lines.append(
                "drift_xy_m: mean=%.4f rmse=%.4f max_abs=%.4f"
                % (drift_xy_stats["mean"], drift_xy_stats["rmse"], drift_xy_stats["max_abs"])
            )
            lines.append(
                "drift_z_m: mean=%.4f rmse=%.4f max_abs=%.4f"
                % (drift_z_stats["mean"], drift_z_stats["rmse"], drift_z_stats["max_abs"])
            )
            lines.append(
                "settling_time_s: %s"
                % ("%.4f" % settling_time if settling_time is not None else "not_settled")
            )

        with open(self.summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _plot(self):
        if not self.samples:
            return

        t = [r["test_t_s"] for r in self.samples]
        target = [r["target_yaw_deg"] for r in self.samples]
        target_rate = [r["target_yaw_rate_deg_s"] for r in self.samples]
        actual = [r["actual_yaw_deg"] for r in self.samples]
        actual_unwrapped = [r["actual_yaw_unwrapped_deg"] for r in self.samples]
        error = [r["yaw_error_deg"] for r in self.samples]
        yaw_rate = [r["actual_yaw_rate_deg_s"] for r in self.samples]
        yaw_rate_error = [r["yaw_rate_error_deg_s"] for r in self.samples]
        drift_xy = [r["drift_xy_m"] for r in self.samples]
        drift_z = [r["drift_z_m"] for r in self.samples]

        fig = plt.figure(figsize=(14, 9))

        ax1 = fig.add_subplot(2, 2, 1)
        if self.mode == "step":
            ax1.step(t, target, where="post", label="target yaw", linewidth=2.0)
        else:
            ax1.plot(t, target, label="target yaw", linewidth=2.0)
        ax1.plot(t, actual, label="actual yaw wrapped", linewidth=1.2)
        ax1.plot(t, actual_unwrapped, label="actual yaw unwrapped", linewidth=1.2, alpha=0.75)
        ax1.set_title("Yaw Reference Response")
        ax1.set_xlabel("t [s]")
        ax1.set_ylabel("yaw [deg]")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t, error, label="yaw error", linewidth=1.5)
        ax2.axhline(self.settle_threshold_deg, color="k", linestyle="--", linewidth=0.8)
        ax2.axhline(-self.settle_threshold_deg, color="k", linestyle="--", linewidth=0.8)
        ax2.set_title("Yaw Error")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("error [deg]")
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, target_rate, label="target yaw rate", linewidth=1.2)
        ax3.plot(t, yaw_rate, label="actual yaw rate", linewidth=1.5)
        ax3.plot(t, yaw_rate_error, label="yaw rate error", linewidth=1.0, alpha=0.75)
        ax3.set_title("Yaw Rate")
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("yaw rate [deg/s]")
        ax3.grid(True)
        ax3.legend()

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, drift_xy, label="xy drift", linewidth=1.5)
        ax4.plot(t, drift_z, label="z drift", linewidth=1.5)
        ax4.set_title("Hover Position Drift")
        ax4.set_xlabel("t [s]")
        ax4.set_ylabel("drift [m]")
        ax4.grid(True)
        ax4.legend()

        fig.tight_layout()
        fig.savefig(self.fig_path, dpi=160)
        plt.close(fig)

    def save(self):
        if self.saved:
            return
        self.saved = True

        if self.commands:
            self._write_csv(
                self.commands_csv_path,
                self.commands,
                [
                    "stamp_s",
                    "test_t_s",
                    "segment_id",
                    "mode",
                    "target_yaw_deg",
                    "target_yaw_rad",
                    "target_yaw_rate_deg_s",
                    "target_yaw_rate_rad_s",
                    "goal_yaw_deg",
                    "goal_yaw_rad",
                    "start_x",
                    "start_y",
                    "start_z",
                ],
            )
        if self.samples:
            self._write_csv(
                self.samples_csv_path,
                self.samples,
                [
                    "stamp_s",
                    "test_t_s",
                    "segment_t_s",
                    "segment_id",
                    "target_yaw_deg",
                    "target_yaw_rad",
                    "goal_yaw_deg",
                    "goal_yaw_rad",
                    "target_yaw_rate_deg_s",
                    "target_yaw_rate_rad_s",
                    "actual_yaw_deg",
                    "actual_yaw_rad",
                    "actual_yaw_unwrapped_deg",
                    "yaw_error_deg",
                    "yaw_error_rad",
                    "actual_yaw_rate_deg_s",
                    "actual_yaw_rate_rad_s",
                    "yaw_rate_error_deg_s",
                    "yaw_rate_error_rad_s",
                    "actual_x",
                    "actual_y",
                    "actual_z",
                    "drift_xy_m",
                    "drift_z_m",
                ],
            )
            self._plot()
            self._save_summary()
            rospy.loginfo("[hover_yaw_step_test] Outputs saved to %s", self.out_dir)
        else:
            rospy.loginfo("[hover_yaw_step_test] No samples captured. Output dir: %s", self.out_dir)


if __name__ == "__main__":
    HoverYawStepTest().run()
