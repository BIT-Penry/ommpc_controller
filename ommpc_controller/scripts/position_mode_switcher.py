#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry


def _distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class PositionModeSwitcher:
    def __init__(self):
        self.odom_topic = rospy.get_param(
            "~odom_topic",
            rospy.get_param("/ommpc_controller/odom_topic", "/some_object_name_vrpn_client/estimated_odometry"),
        )
        self.target_xyz = [10.0, 0.0, 1.0]
        self.switch_distance = float(rospy.get_param("~switch_distance", 0.25))
        self.set_mode_service = rospy.get_param("~set_mode_service", "/mavros/set_mode")
        self.set_mode_timeout = float(rospy.get_param("~set_mode_timeout", 1.0))
        self.shutdown_after_switch = bool(rospy.get_param("~shutdown_after_switch", False))
        
        # 实机
        self.target_px4_mode = rospy.get_param("~target_px4_mode", "POSCTL")


        self.switched = False
        self.set_mode_client = rospy.ServiceProxy(self.set_mode_service, SetMode)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=100)

        rospy.loginfo(
            "[position_mode_switcher] Watching odom=%s target=(%.3f, %.3f, %.3f) threshold=%.3fm target_px4_mode=%s",
            self.odom_topic,
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2],
            self.switch_distance,
            self.target_px4_mode,
        )

    def _odom_cb(self, msg):
        if self.switched:
            return

        p = msg.pose.pose.position
        actual_xyz = (p.x, p.y, p.z)
        dist = _distance(actual_xyz, self.target_xyz)
        rospy.loginfo_throttle(
            1.0,
            "[position_mode_switcher] Distance to target: %.3fm target=(%.2f, %.2f, %.2f)",
            dist,
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2],
        )

        if dist <= self.switch_distance:
            self._switch_mode(dist)

    def _switch_mode(self, dist):
        self.switched = True
        rospy.logwarn(
            "[position_mode_switcher] Within %.3fm of target. Switching PX4 mode to %s.",
            dist,
            self.target_px4_mode,
        )

        mode_sent = self._set_px4_mode(self.target_px4_mode)
        if mode_sent:
            rospy.loginfo("[position_mode_switcher] PX4 mode switch request accepted.")
        else:
            rospy.logwarn("[position_mode_switcher] PX4 mode switch request was not accepted.")

        if self.shutdown_after_switch:
            rospy.signal_shutdown("target position mode switch completed")

    def _set_px4_mode(self, mode):
        try:
            rospy.wait_for_service(self.set_mode_service, timeout=self.set_mode_timeout)
            response = self.set_mode_client(0, mode)
            return bool(response.mode_sent)
        except Exception as exc:
            rospy.logwarn(
                "[position_mode_switcher] Failed to call %s custom_mode=%s: %s",
                self.set_mode_service,
                mode,
                str(exc),
            )
            return False


def main():
    rospy.init_node("position_mode_switcher")
    PositionModeSwitcher()
    rospy.spin()


if __name__ == "__main__":
    main()
