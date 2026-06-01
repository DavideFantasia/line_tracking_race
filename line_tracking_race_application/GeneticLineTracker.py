#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from model import Model

class GeneticLineTracker(Node):
    def __init__(self):
        super().__init__('genetic_line_tracker')
        
        # publisher per velocità
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # subscriber odometria (posizione)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # subscriber telecamera
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # timer loop di controllo
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # stato attuale del robot [x, y, theta]
        self.current_state = np.zeros(3)
        
        # target line temporanea y=0
        self.target_line_data = [0.0, 0.0, 0.0]
        self.last_valid_poly = [0.0, 0.0, 0.0]

        # pesi MPC
        self.mpc = Model()
        #trovati lanciando optimizer.py
        #self.mpc.set_weights([3.0, 3.0, 0.1, 0.5])  # w_d == w_psi, bilanciati
        self.mpc.set_weights([1.687, 7.124, 0.373, 0.007])  # w_d == w_psi, bilanciati
        
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

    def image_callback(self, msg):
        """Elabora l'immagine per trovare l'equazione della curva gialla"""
        try:
            # messaggio ROS -> immagine OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore cv_bridge: {e}")
            return

        # filtro colore giallo
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # estrazione punti
        h, w = mask.shape
        crop_start = int(h * 0.6)  # meta' alta dei punti e' del cielo pressapoco
        
        real_world_x = []
        real_world_y = []
        
        # sampling ogni 20 punti
        for y_img in range(crop_start, h, 20):
            row = mask[y_img, :]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                # centro della striscia gialla in pixel
                x_img = int(np.mean(white_pixels))
                
                # trasformazione in metri, assumendo:
                # base dell'immagine (y_img = h) a circa 0.2m davanti
                # meta' dell'immagine (y_img = crop_start) a circa 3.0m
                x_robot = 0.2 + (3.0 - 0.2) * (h - y_img) / (h - crop_start)
                
                # fov orizzontale della camera è ~2.0 rad -> math.tan(1.0)
                y_robot = - (x_img - w/2) / (w/2) * (x_robot * math.tan(1.0)) 
                
                real_world_x.append(x_robot)
                real_world_y.append(y_robot)

        # curve fitting (y = ax^2 + bx + c)
        unique_x = np.unique(real_world_x)

        if len(unique_x) >= 3:
            poly = np.polyfit(real_world_x, real_world_y, 2)
            # clamp del coefficiente quadratico
            poly[0] = np.clip(poly[0], -0.3, 0.3)
            self.target_line_data = poly
            self.last_valid_poly = poly
        elif len(unique_x) == 2:
            # con pochi punti approssimiamo a una retta di grado 1 e riempiamo lo zero per a
            poly1 = np.polyfit(real_world_x, real_world_y, 1)
            self.target_line_data = [0.0, poly1[0], poly1[1]]
            self.last_valid_poly = self.target_line_data
        else:
            # nessuna linea vista: segui l'ultima traiettoria valida
            self.target_line_data = self.last_valid_poly

        # DEBUG: filtro visivo
        cv2.imshow("Yellow Mask", mask)
        cv2.waitKey(1)

    def solve_mpc(self):
        """
        Ritorna velocità lineare (v) e angolare (omega) ottimali.
        Minimizzazione dei pesi del MPC tramite Algoritmo Genetico: 
        """
        # Se poly è in frame robot, lo stato iniziale è l'errore rispetto alla linea
        y_current = np.polyval(self.target_line_data, 0.0)  # dove si trova la linea a x=0
        dy_dx = np.polyval(np.polyder(self.target_line_data), 0.0)
        theta_line = math.atan2(dy_dx, 1.0)
        local_state = np.array([0.0, -y_current, -theta_line])  # errore reale
        return self.mpc.solve(local_state, self.target_line_data)

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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()