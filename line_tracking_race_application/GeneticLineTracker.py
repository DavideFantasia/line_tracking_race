#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

class GeneticLineTracker(Node):
    def __init__(self):
        super().__init__('genetic_line_tracker')
        
        # Publisher per velocità
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber odometria (posizione)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Timer loop di controllo
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Stato attuale del robot [x, y, theta]
        self.current_state = np.zeros(3)
        
        # Pesi MPC
        self.weight_cross_track_error = 1.0 # Peso distanza dalla linea
        self.weight_steering_effort = 0.1   # Penalizzatore sterzate brusche
        
        self.get_logger().info("MPC Node Initialized. Waiting for Odometry...")

    def odom_callback(self, msg):
        """Aggiornamento dello stato del robot."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Conversione da quaternion ad Euler
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        #remapping in +-PI che se no Daniele mi mangia vivo
        theta = math.atan2(siny_cosp, cosy_cosp)
        
        self.current_state = np.array([x, y, theta])

    def solve_mpc(self):
        """
        Ritorna velocità lineare (v) e angolare (omega) ottimali.
        Minimizzazione dei pesi del MPC tramite Algoritmo Genetico: 
        """
        # TODO: Implementare l'ottimizzatore MPC
        v_opt = 0.5  # Valore di test
        w_opt = 0.0  # Valore di test
        return v_opt, w_opt

    def control_loop(self):
        """Loop eseguito a frequenza fissa che manda i comandi."""
        v, w = self.solve_mpc()
        
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GeneticLineTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()