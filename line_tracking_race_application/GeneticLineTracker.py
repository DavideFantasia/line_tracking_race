#!/usr/bin/env python3
"""
GeneticLineTracker.py
=====================
Nodo ROS 2 con due modalità operative selezionabili via parametro ROS:

  RUN (default)
    Comportamento normale: legge i pesi MPC da parametro ROS,
    segue la linea gialla in autonomia.

  TRAIN
    Modalità training online per il GA in optimizer.py.
    Protocollo:
      1. Attende i pesi su /mpc_weights  (Float64MultiArray)
      2. Li carica nel Model, resetta lo stato interno
      3. Resetta la posizione del robot in Gazebo via /set_entity_state
      4. Gira per eval_duration secondi accumulando errore reale
      5. Pubblica la fitness su /mpc_score  (Float64)
      6. Torna al passo 1
________________________________________________________________________
Parametri ROS dichiarati
    mode            : "run" | "train"                  (default: "run")
    weights         : [w_d, w_psi, w_effort, w_v]      (default: [7.895, 2.486, 0.053, 0.028])
    eval_duration   : secondi per evaluation in TRAIN  (default: 15.0)
    weights_topic   : topic pesi in TRAIN               (default: "/mpc_weights")
    score_topic     : topic fitness in TRAIN            (default: "/mpc_score")
    reset_x, reset_y, reset_yaw : posizione di reset    (default: 0.0, 0.0, 0.0)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64, Float64MultiArray
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Quaternion, Vector3

from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

from model import Model


# PARAMETRI PER IL MASKING DELLA CAMERA
LOWER_YELLOW = [20, 50, 50]
UPPER_YELLOW = [30, 255, 255]

class GeneticLineTracker(Node):

    def __init__(self):
        super().__init__("genetic_line_tracker")

        self._node_ready = False 
        self.should_visualize = True

        self.declare_parameter("mode",          "run")
        #self.declare_parameter("weights",       [7.895, 2.486, 0.053, 0.028])
        self.declare_parameter("weights",       [3.0, 3.0, 0.3, 1.0])
        self.declare_parameter("eval_duration", 15.0)
        self.declare_parameter("weights_topic", "/mpc_weights")
        self.declare_parameter("score_topic",   "/mpc_score")
        self.declare_parameter("reset_x",       0.0)
        self.declare_parameter("reset_y",       0.0)
        self.declare_parameter("reset_yaw",     0.0)

        self._mode          = self.get_parameter("mode").value
        self._eval_duration = self.get_parameter("eval_duration").value
        self._reset_pose    = (
            self.get_parameter("reset_x").value,
            self.get_parameter("reset_y").value,
            self.get_parameter("reset_yaw").value,
        )

        # Publisher/subscriber comuni ad entrambe le modalità
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.error_pub = self.create_publisher(Vector3, "/tracking_error", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self._image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self._camera_info_callback, 10)


        self.camera_height = 0.57  
        self.camera_pitch  = math.pi / 6 
        self.camera_info_msg = None
        self.prev_centroid = None
        self.current_state = np.zeros(3)

        # Variabili di stato Frenet-Serret estratte dalla visione
        self.d = 0.0
        self.psi = 0.0
        self.gamma = 0.0
        self.traj_poly = np.zeros(3)

        # ultima soluzione valida
        self.last_valid_d = 0.0
        self.last_valid_psi = 0.0
        self.last_valid_gamma = 0.0

        # Modello MPC
        self.mpc = Model(0.1, 20)
        self.timer_update_frequency = 0.1   #ogni 0.1 secondi, come il framerate di aggiornamento della camera

        # Setup per modalità specifica
        if self._mode == "train":
            self._setup_train_mode()
        else:
            self._setup_run_mode()

        self.get_logger().info(f"GeneticLineTracker avviato in modalità: {self._mode.upper()}")

    def _setup_run_mode(self):
        import ast
        weights = self.get_parameter("weights").value
        if isinstance(weights, str):
            weights = ast.literal_eval(weights)
        self.mpc.set_weights(weights)
        self.get_logger().info(f"Pesi MPC caricati: {[round(w, 3) for w in weights]}")

        self.timer = self.create_timer(self.timer_update_frequency, self._control_loop)

    def _setup_train_mode(self):
        """
        Modalità training: non avvia il loop autonomo,
        ma reagisce ai pesi ricevuti via topic e pubblica la fitness.
        """
        weights_topic = self.get_parameter("weights_topic").value
        score_topic   = self.get_parameter("score_topic").value

        self.weights_sub = self.create_subscription(Float64MultiArray, weights_topic, self._on_new_weights, 1)
        self.score_pub = self.create_publisher(Float64, score_topic, 1)
        self._reset_client = self.create_client(SetEntityState, "/set_entity_state")

        self._eval_active       = False   
        self._eval_start_time   = None
        self._accumulated_error = 0.0
        self._eval_steps        = 0

        self.timer = self.create_timer(self.timer_update_frequency, self._train_loop)
        self.get_logger().info(f"Modalità TRAIN in attesa su '{weights_topic}' ...")

    # Callback odometria (comune ad entrambe le modalita')
    def _odom_callback(self, msg):
        self._first_odom_received = True
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        self.current_state = np.array([x, y, theta])

    def _camera_info_callback(self, msg):
        if self.camera_info_msg is None:
            self.camera_info_msg = msg
        self.K = [
            [msg.k[0], msg.k[1], msg.k[2]],
            [msg.k[3], msg.k[4], msg.k[5]],
            [msg.k[6], msg.k[7], msg.k[8]],
        ]

    # Credits: https://github.com/GioZerini
    def plan(self, img_msg, camera_info_msg, robot_angle):
        # Convert the image
        image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        height, width, _ = image.shape

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))

        # Compute centroid
        M = cv.moments(mask)
        if M["m00"] != 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            self.prev_centroid = centroid
        else:
            if self.prev_centroid is None:
                self.get_logger().warn("No centroid found and no previous to reuse. Using center.")
                centroid = (width // 2, height // 2)
            else:
                self.get_logger().warn("No centroid found, reusing previous waypoint.")
                centroid = self.prev_centroid

        # Conversion matrix
        K = np.array(camera_info_msg)
        crosshair = (width // 2, height // 2)

        # Call the estimate_d_psi_gamma function
        result = self.estimate_d_psi_gamma(
            mask=mask,
            crosshair_px=crosshair,
            robot_angle=robot_angle,
            K=K,
            image=image
        )

        if result is not None:
            # Linea vista! Aggiorno i valori e la memoria
            d, psi, gamma, traj_poly = result
            self.last_valid_d = d
            self.last_valid_psi = psi
            self.last_valid_gamma = gamma
            self.last_valid_traj_poly = traj_poly
        else:
            # Linea persa di vista
            self.get_logger().warn("Linea persa, uso di precedenti valori")
            d = self.last_valid_d
            psi = self.last_valid_psi
            gamma = self.last_valid_gamma
            self.traj_poly = self.last_valid_traj_poly

        # Visualization part
        if self.should_visualize:
            cv.circle(image, crosshair, 5, (0, 255, 0), 2)   
            cv.circle(image, centroid, 5, (0, 0, 255), 2)    
            cv.putText(image, f"d={d:.3f} m", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv.putText(image, f"psi={psi:.3f} rad", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv.putText(image, f"gamma={gamma:.3f} 1/m", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv.imshow("Line Following Visualization", image)
            cv.waitKey(1)

        return {"d": d, "psi": psi, "gamma": gamma}
    
    # Credits: https://github.com/GioZerini
    def estimate_d_psi_gamma(self, mask, crosshair_px, robot_angle, K, image):
        """
        Estrae d, psi, gamma proiettando i pixel sul piano 3D del suolo
        usando la geometria esatta della telecamera Pinhole.
        """
        # Parametri intrinseci della telecamera estratti da K
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Parametri fisici dal file URDF/Xacro
        H = self.camera_height         # 0.57 m
        pitch = self.camera_pitch      # 30 gradi (pi/6)
        cam_offset_x = 0.2             # Offset X della telecamera rispetto al base_link
        
        h, w = mask.shape
        pts_x = []
        pts_y = []
        
        # Scansiona l'immagine dal basso (vicino al robot) verso metà altezza
        crop_start = int(h * 0.4)
        for v in range(h - 1, crop_start, -15):
            row = mask[v, :]
            whites = np.where(row > 0)[0]
            
            if len(whites) > 0:
                # Trova il centro della linea in pixel (u)
                u = np.mean(whites)
                
                # PROIEZIONE 3D: PIXEL -> METRI SUL SUOLO
                # Vettore del raggio normalizzato nel frame ottico
                nx = (u - cx) / fx
                ny = (v - cy) / fy
                
                # Denominatore per l'intersezione col suolo (Z = -H)
                denom = math.sin(pitch) + ny * math.cos(pitch)
                if denom <= 0.01:
                    continue  # Il raggio punta verso il cielo o all'orizzonte
                    
                # Profondità s scalata per raggiungere il suolo
                s = H / denom
                
                # Coordinate nel frame della telecamera proiettate a terra
                x_ground = s * (math.cos(pitch) - ny * math.sin(pitch))
                y_ground = -s * nx  # Negativo perché Y_robot va a sinistra
                
                # Coordinate finali nel frame del ROBOT (chassis)
                x_robot = cam_offset_x + x_ground
                y_robot = y_ground
                
                pts_x.append(x_robot)
                pts_y.append(y_robot)
                
        # Se non ci sono abbastanza punti, ritorna 0
        if len(pts_x) < 3:
            return None
            
        # FIT POLINOMIALE LOCALE IN METRI
        # Ora i punti sono (X, Y) reali a terra. Il fit è y = ax^2 + bx + c
        poly = np.polyfit(pts_x, pts_y, 2)
        a, b, c = poly
        
        # ESTRAZIONE FRENET-SERRET (al muso del robot, X=0)
        # d: Distanza laterale del robot dalla linea. d = -y(0)
        d = float(-c)
        
        # psi: Errore angolare locale. psi = -atan(y'(0))
        psi = float(-math.atan(b))
        
        # gamma: Curvatura locale. gamma = y''(0) / (1 + y'(0)^2)^1.5
        gamma = float((2 * a) / math.pow(1 + b**2, 1.5))
        
        # plot di info utili
        cx_px, cy_px = crosshair_px
        cv.circle(image, (int(cx_px), int(cy_px)), 5, (0, 255, 0), 2)
        # Mostra dove il robot crede che sia la linea
        for px, py in zip(pts_x, pts_y):
            # Proiezione rudimentale indietro per il display
            disp_y = int(cy + fy * (math.cos(pitch) * H - math.sin(pitch) * (px - cam_offset_x)) / (math.sin(pitch) * H + math.cos(pitch) * (px - cam_offset_x)))
            disp_x = int(cx - fx * py / (math.sin(pitch) * H + math.cos(pitch) * (px - cam_offset_x)))
            if 0 <= disp_x < w and 0 <= disp_y < h:
                cv.circle(image, (disp_x, disp_y), 3, (0, 128, 255), -1)

        return d, psi, gamma, poly

    # Function to normalize an angle
    def _wrap_pi(self, a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a
    
    # Callback immagine - Centroid strategy (comune)
    def _image_callback(self, msg):
        if self.camera_info_msg is None:
            self.get_logger().warn("Waiting for camera_info...")
            return

        if self.K is None:
            self.get_logger().warn("Camera matrix K not yet received — skipping frame.")
            return
            
        robot_angle = self.current_state[2]
        res = self.plan(msg, self.K, robot_angle)
        
        if res is None:
            self.get_logger().error("[IMAGE CALLBACK] res value is None")
            return

        self.d = res["d"]
        self.psi = res["psi"]
        self.gamma = res["gamma"]

    # Calcolo MPC (comune)
    def _solve_mpc(self):
        """
        Calcola (v, w) ottimali.
        Lo stato locale codifica solo l'errore angolare - l'errore laterale
        è già nel termine costante del polinomio (frame robot).
        """
        # Frenet-Serret state: [sigma, d, psi]
        local_state = np.array([0.0, self.d, self.psi])
        return self.mpc.solve(local_state, self.gamma, self.traj_poly)

    def _publish_cmd(self, v: float, w: float):
        if not getattr(self, '_first_odom_received', False):
            return
        if math.isnan(v) or math.isnan(w):
            self.get_logger().error("ATTENZIONE: Comandi NaN rilevati! Forzo il robot a fermarsi.")
            v, w = 0.0, 0.0

        cmd             = Twist()
        cmd.linear.x    = float(v)
        cmd.angular.z   = float(w)
        self.cmd_vel_pub.publish(cmd)


    # Modalità RUN - loop di controllo normale
    def _control_loop(self):
        if not self._node_ready:
            return
        
        err_msg = Vector3()
        err_msg.x = float(self.d)      # Errore laterale
        err_msg.y = float(self.psi)    # Errore angolare
        err_msg.z = float(self.gamma)  # Curvatura
        self.error_pub.publish(err_msg)

        v, w = self._solve_mpc()
        self._publish_cmd(v, w)


    # Modalità TRAIN - gestione valutazione online
    def _on_new_weights(self, msg: Float64MultiArray):
        """
        Riceve i nuovi pesi dal GA.
        Carica i pesi, resetta Gazebo e avvia una nuova valutazione.
        """
        weights = list(msg.data)
        self.get_logger().info(f"[TRAIN] Nuovi pesi ricevuti: {[round(w, 3) for w in weights]}")

        self.mpc.set_weights(weights)

        self._reset_gazebo_pose()

        # Avvia la valutazione
        self._accumulated_error = 0.0
        self._eval_steps        = 0
        self._eval_start_time   = self.get_clock().now()
        self._eval_active       = True

    def _reset_gazebo_pose(self):
        """
        Resetta la posizione del robot in Gazebo via SetEntityState.
        Gestisce il caso in cui il servizio non sia disponibile senza bloccare.
        """
        if not self._reset_client.service_is_ready():
            self.get_logger().info("[TRAIN] /set_entity_state non disponibile — reset saltato")
            return

        rx, ry, ryaw = self._reset_pose

        # Conversione yaw -> quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(ryaw / 2.0)
        q.w = math.cos(ryaw / 2.0)

        pose = Pose()
        pose.position.x = rx
        pose.position.y = ry
        pose.position.z = 0.45
        pose.orientation= q

        req = SetEntityState.Request()
        req.state        = EntityState()
        req.state.name   = "car"   # nome dell'entità in Gazebo
        req.state.pose   = pose

        # Chiamata asincrona: non blocca il loop ROS
        future = self._reset_client.call_async(req)
        future.add_done_callback(self._on_reset_done)

    def _on_reset_done(self, future):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().info(f"[TRAIN] Reset Gazebo fallito: {result.status_message}")
        except Exception as e:
            self.get_logger().error(f"[TRAIN] Errore reset Gazebo: {e}")

    def _train_loop(self):
        """
        Eseguito ogni 0.05s in modalità TRAIN.
        Se una valutazione è attiva, guida il robot e accumula l'errore.
        Quando il tempo è scaduto, pubblica la fitness e aspetta nuovi pesi.
        """
        if not self._eval_active:
            # In attesa di nuovi pesi: robot fermo
            self._publish_cmd(0.0, 0.0)
            return

        # Controlla se la valutazione è terminata
        elapsed = (self.get_clock().now() - self._eval_start_time).nanoseconds * 1e-9
        if elapsed >= self._eval_duration:
            self._publish_eval_score()
            self._eval_active = False
            self._publish_cmd(0.0, 0.0)
            return

        # Guida il robot e accumula l'errore reale
        v, w = self._solve_mpc()
        self._publish_cmd(v, w)
        
        # L'errore accumulato si basa direttamente sulle stime dalla visione
        self._accumulated_error += abs(self.d) + abs(self.psi)
        self._eval_steps        += 1

    def _publish_eval_score(self):
        """Calcola la fitness media e la pubblica su /mpc_score."""
        if self._eval_steps == 0:
            fitness = 0.0001
        else:
            mean_error = self._accumulated_error / self._eval_steps
            fitness    = 1.0 / (mean_error + 0.001) if mean_error < 1000 else 0.0001

        msg      = Float64()
        msg.data = fitness
        self.score_pub.publish(msg)

        self.get_logger().info(
            f"[TRAIN] Valutazione completata - steps: {self._eval_steps} | "
            f"errore medio: {self._accumulated_error / max(self._eval_steps, 1):.4f} | "
            f"fitness: {fitness:.4f}"
        )


# ==================================================

def main(args=None):
    rclpy.init(args=args)
    node = GeneticLineTracker()

    node._node_ready = True

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()