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
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Quaternion

from cv_bridge import CvBridge
import cv2
import numpy as np

from model import Model


class GeneticLineTracker(Node):

    def __init__(self):
        super().__init__("genetic_line_tracker")

        # Callback group separato per la camera — non viene bloccato dal timer MPC
        self._camera_cb_group  = MutuallyExclusiveCallbackGroup()
        self._odom_cb_group    = MutuallyExclusiveCallbackGroup()
        self._control_cb_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameter("mode",          "run")
        self.declare_parameter("weights",       [7.895, 2.486, 0.053, 0.028])
        self.declare_parameter("eval_duration", 15.0)
        self.declare_parameter("weights_topic", "/mpc_weights")
        self.declare_parameter("score_topic",   "/mpc_score")
        self.declare_parameter("reset_x",       0.0)
        self.declare_parameter("reset_y",       0.0)
        self.declare_parameter("reset_yaw",     0.0)

        self._mode          = self.get_parameter("mode").value
        self._eval_duration = self.get_parameter("eval_duration").value
        self._reset_pose = (
                                self.get_parameter("reset_x").value,
                                self.get_parameter("reset_y").value,
                                self.get_parameter("reset_yaw").value,
                            )

        # Publisher/subscriber comuni ad entrambe le modalità
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
                                                    Odometry, "/odom", self._odom_callback, 10,
                                                    callback_group=self._odom_cb_group
                                                )
        self.bridge   = CvBridge()
        self.image_sub = self.create_subscription(
                                                    Image, "/camera/image_raw", self._image_callback, 10,
                                                    callback_group=self._camera_cb_group   # thread dedicato
                                                )
        # Stato corrente del robot [x, y, theta] -- aggiornato da odom
        self.current_state = np.zeros(3)
        # Traiettoria target [a, b, c] per y = ax^2 + bx + c -- aggiornata dalla camera
        self.target_line_data = [0.0, 0.0, 0.0]
        self.last_valid_poly  = [0.0, 0.0, 0.0]

        # Modello MPC
        self.mpc = Model()

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

        # Timer nel gruppo di controllo — gira in thread separato dalla camera
        self.timer = self.create_timer(
            0.05, self._control_loop,
            callback_group=self._control_cb_group
        )

    def _setup_train_mode(self):
        """
        Modalità training: non avvia il loop autonomo,
        ma reagisce ai pesi ricevuti via topic e pubblica la fitness.
        """
        weights_topic = self.get_parameter("weights_topic").value
        score_topic   = self.get_parameter("score_topic").value

        # Subscriber: riceve i nuovi pesi dal GA
        self.weights_sub = self.create_subscription(
            Float64MultiArray,
            weights_topic,
            self._on_new_weights,
            1,  # QoS depth 1 - interessa solo l'ultimo messaggio
        )

        # Publisher: invia la fitness al GA
        self.score_pub = self.create_publisher(Float64, score_topic, 1)

        # Client per resettare la posizione in Gazebo
        self._reset_client = self.create_client(SetEntityState, "/set_entity_state")

        # Stato interno della valutazione
        self._eval_active    = False   # True durante una valutazione
        self._eval_start_time = None
        self._accumulated_error = 0.0
        self._eval_steps     = 0

        # Loop di controllo attivo solo durante una valutazione
        self.timer = self.create_timer(
            0.05, self._train_loop,
            callback_group=self._control_cb_group
        )

        self.get_logger().info(
            f"Modalità TRAIN — in attesa di pesi su '{weights_topic}' ..."
        )

    # Callback odometria (comune ad entrambe le modalita')
    def _odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        self.current_state = np.array([x, y, theta])


    # Callback immagine - CenterLine strategy (comune)
    def _image_callback(self, msg):
        self.get_logger().warn("immagine ricevuta")  # DEBUG
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        height, width, _ = cv_image.shape

        # Filtro colore HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask= cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))

        # Contorno del tracciato
        track_outline = np.zeros((height, width), dtype=np.uint8)
        contours, _   = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(track_outline, contours, 0, 255, 1)

        # crop: elimina cielo e bordi laterali rumorosi
        crop_y0, crop_y1 = int(height / 2), height - 10
        crop_x0, crop_x1 = 100, width - 100
        cropped = track_outline[crop_y0:crop_y1, crop_x0:crop_x1]

        # Componenti connesse -> bordo sinistro (label 1) e destro (label 2)
        _, labels = cv2.connectedComponents(cropped)
        left_rows,  left_cols  = np.where(labels == 1)
        right_rows, right_cols = np.where(labels == 2)

        if left_rows.size == 0 or right_rows.size == 0:
            self.target_line_data = self.last_valid_poly
            return

        left_pts  = np.column_stack((left_cols,  left_rows))[::10]
        right_pts = np.column_stack((right_cols, right_rows))[::10]

        # Centerline: media dei due bordi
        real_world_x, real_world_y = [], []
        for (x1, y1), (x2, y2) in zip(left_pts, right_pts):
            x_img = int((x1 + x2) / 2) + crop_x0
            y_img = int((y1 + y2) / 2) + crop_y0

            # Proiezione pixel → frame robot
            x_robot = 0.2 + (3.0 - 0.2) * (height - y_img) / (height - crop_y0)
            y_robot = -(x_img - width / 2) / (width / 2) * (x_robot * math.tan(1.0))

            real_world_x.append(x_robot)
            real_world_y.append(y_robot)

        # Polyfit
        unique_x = np.unique(real_world_x)
        if len(unique_x) >= 3:
            poly    = np.polyfit(real_world_x, real_world_y, 2)
            poly[0] = np.clip(poly[0], -0.3, 0.3)
            self.target_line_data = poly
            self.last_valid_poly  = poly
        elif len(unique_x) == 2:
            poly1 = np.polyfit(real_world_x, real_world_y, 1)
            self.target_line_data = [0.0, poly1[0], poly1[1]]
            self.last_valid_poly  = self.target_line_data
        else:
            self.target_line_data = self.last_valid_poly

        cv2.imshow("Yellow Mask", mask)
        cv2.waitKey(1)


    # Calcolo MPC (comune)
    def _solve_mpc(self):
        """
        Calcola (v, w) ottimali.
        Lo stato locale codifica solo l'errore angolare - l'errore laterale
        è già nel termine costante del polinomio (frame robot).
        """
        dy_dx      = np.polyval(np.polyder(self.target_line_data), 0.0)
        theta_line = math.atan2(dy_dx, 1.0)
        local_state = np.array([0.0, 0.0, -theta_line])
        return self.mpc.solve(local_state, self.target_line_data)

    def _publish_cmd(self, v: float, w: float):
        cmd             = Twist()
        cmd.linear.x    = float(v)
        cmd.angular.z   = float(w)
        self.cmd_vel_pub.publish(cmd)


    # Modalità RUN — loop di controllo normale
    def _control_loop(self):
        v, w = self._solve_mpc()
        self._publish_cmd(v, w)


    # Modalità TRAIN — gestione valutazione online
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
            self.get_logger().warn("[TRAIN] /set_entity_state non disponibile — reset saltato")
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
                self.get_logger().warn(f"[TRAIN] Reset Gazebo fallito: {result.status_message}")
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

        # Errore laterale: polyval a x=0 è l'offset della linea al muso del robot
        y_err = abs(np.polyval(self.target_line_data, 0.0))

        # Errore angolare: termine lineare del polinomio
        dy_dx        = np.polyval(np.polyder(self.target_line_data), 0.0)
        theta_target = math.atan2(dy_dx, 1.0)
        psi_err      = abs((-theta_target + math.pi) % (2 * math.pi) - math.pi)

        self._accumulated_error += y_err + psi_err
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

    # 2 thread: uno per camera/odom, uno per il loop di controllo MPC
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()