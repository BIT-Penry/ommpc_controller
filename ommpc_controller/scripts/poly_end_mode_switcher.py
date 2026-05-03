#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from traj_utils.msg import PolyTraj


POLY_TRAJ_STATE = 11


def _bool_param(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _stamp_to_sec(stamp):
    sec = stamp.to_sec()
    if sec > 1.0e-6:
        return sec
    return rospy.Time.now().to_sec()


def _eval_piece(coeffs, order, t):
    value = 0.0
    for i, c in enumerate(coeffs):
        value += c * (t ** (order - i))
    return value


def _poly_end_xyz(msg):
    order = int(msg.order)
    if order < 3:
        return None

    durations = list(msg.duration)
    if not durations:
        return None

    piece_count = len(durations)
    coeff_count = order + 1
    required_len = piece_count * coeff_count
    if len(msg.coef_x) < required_len or len(msg.coef_y) < required_len or len(msg.coef_z) < required_len:
        return None

    start = (piece_count - 1) * coeff_count
    end = start + coeff_count
    local_t = float(durations[-1])
    return (
        _eval_piece(msg.coef_x[start:end], order, local_t),
        _eval_piece(msg.coef_y[start:end], order, local_t),
        _eval_piece(msg.coef_z[start:end], order, local_t),
    )


def _distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class PolyEndModeSwitcher:
    def __init__(self):
        self.traj_topic = rospy.get_param("~traj_topic", "/drone_0_planning/trajectory")
        self.odom_topic = rospy.get_param(
            "~odom_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.exec_state_topic = rospy.get_param("~exec_state_topic", "/ommpc_controller/exec_traj_state")
        self.switch_distance = float(rospy.get_param("~switch_distance", 0.25))
        self.min_time_after_start = float(rospy.get_param("~min_time_after_start", 0.2))
        self.require_poly_state = _bool_param("~require_poly_state", True)
        self.poly_state_code = int(rospy.get_param("~poly_state_code", POLY_TRAJ_STATE))
        self.set_mode_service = rospy.get_param("~set_mode_service", "/mavros/set_mode")
        self.set_mode_timeout = float(rospy.get_param("~set_mode_timeout", 1.0))
        self.abort_poly_after_switch = _bool_param("~abort_poly_after_switch", True)
        self.abort_publish_count = max(1, int(rospy.get_param("~abort_publish_count", 5)))
        self.abort_publish_period = max(0.0, float(rospy.get_param("~abort_publish_period", 0.05)))
        self.set_command_or_hover_false = _bool_param("~set_command_or_hover_false", True)
        self.dynamic_reconfigure_node = rospy.get_param("~dynamic_reconfigure_node", "/ommpc_controller")
        self.dynamic_reconfigure_timeout = float(rospy.get_param("~dynamic_reconfigure_timeout", 0.5))
        self.shutdown_after_switch = _bool_param("~shutdown_after_switch", False)

        # 实机
        # self.target_px4_mode = rospy.get_param("~target_px4_mode", "POSCTL")
        # gazebo
        self.target_px4_mode = rospy.get_param("~target_px4_mode", "AUTO.LOITER")

        self.current_exec_state = None
        self.active_key = None
        self.active_drone_id = 0
        self.active_traj_id = 1
        self.active_start_time = None
        self.active_end_time = None
        self.target_xyz = None
        self.switched = False

        self.abort_pub = rospy.Publisher(self.traj_topic, PolyTraj, queue_size=10)
        self.set_mode_client = rospy.ServiceProxy(self.set_mode_service, SetMode)
        rospy.Subscriber(self.traj_topic, PolyTraj, self._traj_cb, queue_size=20)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=100)
        rospy.Subscriber(self.exec_state_topic, Int32, self._exec_state_cb, queue_size=20)

        rospy.loginfo(
            "[poly_end_mode_switcher] Watching traj=%s odom=%s threshold=%.3fm target_px4_mode=%s",
            self.traj_topic,
            self.odom_topic,
            self.switch_distance,
            self.target_px4_mode,
        )

    def _traj_cb(self, msg):
        target_xyz = _poly_end_xyz(msg)
        if target_xyz is None:
            return

        start_sec = msg.start_time.to_sec()
        total_duration = float(sum(msg.duration))
        active_key = (int(msg.traj_id), start_sec, total_duration)
        if active_key == self.active_key:
            return

        self.active_key = active_key
        self.active_drone_id = int(msg.drone_id)
        self.active_traj_id = max(1, int(msg.traj_id))
        self.active_start_time = start_sec
        self.active_end_time = start_sec + total_duration
        self.target_xyz = target_xyz
        self.switched = False

        rospy.loginfo(
            "[poly_end_mode_switcher] New poly traj id=%d end=(%.3f, %.3f, %.3f) total_dur=%.2fs",
            self.active_traj_id,
            target_xyz[0],
            target_xyz[1],
            target_xyz[2],
            total_duration,
        )

    def _exec_state_cb(self, msg):
        self.current_exec_state = int(msg.data)

    def _odom_cb(self, msg):
        if self.switched or self.target_xyz is None:
            return

        if self.require_poly_state and self.current_exec_state != self.poly_state_code:
            return

        now_sec = _stamp_to_sec(msg.header.stamp)
        if self.active_start_time is not None and now_sec < self.active_start_time + self.min_time_after_start:
            return

        p = msg.pose.pose.position
        actual_xyz = (p.x, p.y, p.z)
        dist = _distance(actual_xyz, self.target_xyz)
        rospy.loginfo_throttle(
            1.0,
            "[poly_end_mode_switcher] Distance to poly end: %.3fm target=(%.2f, %.2f, %.2f)",
            dist,
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2],
        )

        if dist <= self.switch_distance:
            self._switch_to_position_mode(dist, now_sec)

    def _switch_to_position_mode(self, dist, now_sec):
        self.switched = True
        rospy.logwarn(
            "[poly_end_mode_switcher] Within %.3fm of target at t=%.3f. Switching PX4 mode to %s.",
            dist,
            now_sec,
            self.target_px4_mode,
        )

        mode_sent = self._set_px4_mode(self.target_px4_mode)
        if mode_sent:
            rospy.loginfo("[poly_end_mode_switcher] PX4 mode switch request accepted.")
        else:
            rospy.logwarn("[poly_end_mode_switcher] PX4 mode switch request was not accepted.")

        if mode_sent and self.abort_poly_after_switch:
            self._publish_poly_abort()

        if mode_sent and self.set_command_or_hover_false:
            self._set_command_or_hover(False)

        if self.shutdown_after_switch:
            rospy.signal_shutdown("poly endpoint mode switch completed")

    def _set_px4_mode(self, mode):
        try:
            rospy.wait_for_service(self.set_mode_service, timeout=self.set_mode_timeout)
            response = self.set_mode_client(0, mode)
            return bool(response.mode_sent)
        except Exception as exc:
            rospy.logwarn(
                "[poly_end_mode_switcher] Failed to call %s custom_mode=%s: %s",
                self.set_mode_service,
                mode,
                str(exc),
            )
            return False

    def _publish_poly_abort(self):
        abort_msg = PolyTraj()
        abort_msg.drone_id = self.active_drone_id
        abort_msg.traj_id = self.active_traj_id
        abort_msg.start_time = rospy.Time.now()
        abort_msg.order = 0
        abort_msg.coef_x = []
        abort_msg.coef_y = []
        abort_msg.coef_z = []
        abort_msg.duration = []

        for _ in range(self.abort_publish_count):
            self.abort_pub.publish(abort_msg)
            if self.abort_publish_period > 0.0:
                rospy.sleep(self.abort_publish_period)

    def _set_command_or_hover(self, enabled):
        try:
            import dynamic_reconfigure.client

            client = dynamic_reconfigure.client.Client(
                self.dynamic_reconfigure_node,
                timeout=self.dynamic_reconfigure_timeout,
            )
            client.update_configuration({"command_or_hover": bool(enabled)})
            rospy.loginfo(
                "[poly_end_mode_switcher] Set %s command_or_hover=%s",
                self.dynamic_reconfigure_node,
                str(bool(enabled)),
            )
        except Exception as exc:
            rospy.logwarn(
                "[poly_end_mode_switcher] Failed to set command_or_hover via dynamic_reconfigure: %s",
                str(exc),
            )


def main():
    rospy.init_node("poly_end_mode_switcher")
    PolyEndModeSwitcher()
    rospy.spin()


if __name__ == "__main__":
    main()
