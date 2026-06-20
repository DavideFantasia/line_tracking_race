"""
Pattern Strategy per la valutazione della fitness del GA.

Due implementazioni:
  - OfflineFitnessStrategy  : simulatore analitico puro (nessuna dipendenza ROS)
  - OnlineGazeboFitnessStrategy : valutazione reale su Gazebo via ROS 2 topics

Il GA in optimizer.py non conosce quale strategia sta usando — chiama solo
strategy.evaluate(weights) e riceve un float.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import math
import time
import json
import threading
from typing import List

import numpy as np
from model import Model

# Interfaccia base astratta
class FitnessStrategy(ABC):
    """
    Interfaccia unica che il GA chiama per valutare un individuo.
    Ogni implementazione può avere parametri propri, ma espone sempre:
      - evaluate(weights: List[float]) -> float
      - name: str (per logging)
    """

    @abstractmethod
    def evaluate(self, weights: List[float]) -> float:
        """
        Valuta la fitness di un set di pesi MPC.
        ________________________________________
        Parametri
            weights : [w_d, w_psi, w_effort, w_v, qf_mult_d, qf_mult_psi, gamma_decay_start]
        Returna
            float : fitness (valore più alto = robot migliore). Mai negativo.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome della strategia, usato nei log del GA."""

    def log(self, msg: str):
        """Metodo di logging base. Effettua un flush immediato sul terminale."""
        print(msg, flush=True)

# Strategia - Simulatore analitico offline
class OfflineFitnessStrategy(FitnessStrategy):
    """
    Valuta i pesi simulando la cinematica del robot analiticamente.
    Non richiede ROS né Gazebo — eseguibile standalone.
    __________________________________________________________________________________
    Parametri configurabili:
        noise_std        : (std_a, std_b, std_c) — rumore sui coefficienti del polyfit
        n_noise_samples  : quante realizzazioni di rumore per individuo
        cmd_delay        : latenza in step tra calcolo e applicazione del comando
        training_steps   : step di simulazione per scenario
        track_half_width : larghezza massima prima della penalità di bordo
        soft_margin      : larghezza oltre cui inizia la penalità progressiva
        out_penalty      : penalità per ogni step rimanente dopo uscita
    """

    def __init__(
        self,
        noise_std:        tuple  = (0.005, 0.02, 0.05),
        n_noise_samples:  int    = 3,
        cmd_delay:        int    = 2,
        training_steps:   int    = 300,
        track_half_width: float  = 1.0,
    ):
        self.noise_std_a, self.noise_std_b, self.noise_std_c = noise_std
        self.n_noise_samples  = n_noise_samples
        self.cmd_delay        = cmd_delay
        self.training_steps   = training_steps
        self.track_half_width = track_half_width

        # Scenari: (target_poly, initial_state)
        # Definiti qui una volta sola — aggiungi/rimuovi scenari in questo punto.
        self._scenarios = [
            # Scenario 1 — Rettilineo con offset laterale
            ([0.0, 0.0, 0.0], np.array([0.0,  0.5, 0.0])),
            # Scenario 2 — Solo errore angolare (forza w_psi alto)
            ([0.0, 0.0, 0.0], np.array([0.0,  0.0, 0.4])),
            # Scenario 3 — Curva parabolica dolce con offset e angolo
            ([0.05, 0.0, 0.0], np.array([0.0,  0.3, 0.2])),
            # Scenario 4 — Curva stretta (stress test)
            ([0.15, 0.0, 0.0], np.array([0.0,  0.2, 0.1])),
            # Scenario 5 — Curva parabolica dolce con offset e angolo
            ([0.05, 0.0, 0.0], np.array([0.0,  -0.3, -0.2])),
            # Scenario 6 — Curva stretta (stress test)
            ([0.15, 0.0, 0.0], np.array([0.0,  -0.2, -0.1])),
        ]

    @property
    def name(self) -> str:
        return "OfflineSimulator"

    def evaluate(self, weights: List[float]) -> float:
        mpc = Model()
        mpc.set_weights(weights)
        
        total_error = 0.0
        total_expected_steps = 0

        for _ in range(self.n_noise_samples):
            na = np.random.normal(0, self.noise_std_a)
            nb = np.random.normal(0, self.noise_std_b)
            nc = np.random.normal(0, self.noise_std_c)

            for target_poly, init_state in self._scenarios:
                noisy_poly = [
                    target_poly[0] + na,
                    target_poly[1] + nb,
                    target_poly[2] + nc,
                ]
                err, expected_steps = self._simulate_scenario(mpc, init_state, target_poly, noisy_poly)
                total_error += err
                total_expected_steps += expected_steps

        mean_error = total_error / total_expected_steps
        # controllo sull'esplosione del risultato
        if math.isnan(mean_error) or mean_error > 1000:
            return 0.0001

        return 1.0 / (mean_error + 0.001)

    def _simulate_scenario(
        self,
        mpc:         Model,
        init_state:  np.ndarray,
        target_poly: List[float],
        noisy_poly:  List[float],
    ) -> tuple[float, int]:
        """
        Simula un singolo scenario con latenza dei comandi e rumore.
        Il robot percepisce 'noisy_poly', si muove con cinematica reale,
        l'errore è misurato su 'target_poly'.
        """
        state           = init_state.copy()
        cmd_buffer      = [(0.0, 0.0)] * self.cmd_delay
        scenario_error  = 0.0
        actual_steps    = 0
        expected_steps  = self.training_steps

        target_poly_der = np.polyder(target_poly)

        for step in range(self.training_steps):
            y_noisy     = np.polyval(noisy_poly, state[0])
            dy_dx_noisy = np.polyval(np.polyder(noisy_poly), state[0])
            theta_noisy = math.atan2(dy_dx_noisy, 1.0)
            
            d_noisy     = state[1] - y_noisy
            psi_noisy   = (state[2] - theta_noisy + math.pi) % (2*math.pi) - math.pi
            gamma_noisy = (2 * noisy_poly[0]) / math.pow(1 + dy_dx_noisy**2, 1.5)

            v_cmd, w_cmd = mpc.solve(np.array([0.0, d_noisy, psi_noisy]), gamma_noisy, np.array(noisy_poly))

            cmd_buffer.append((v_cmd, w_cmd))
            v_applied, w_applied = cmd_buffer.pop(0)

            # kinematica diretta
            state[0] += mpc.dt * math.cos(state[2]) * v_applied
            state[1] += mpc.dt * math.sin(state[2]) * v_applied
            state[2] += mpc.dt * w_applied

            # ==== ERRORE SENZA RUMORE PER LA FITNESS ====
            y_target     = np.polyval(target_poly, state[0])
            dy_dx        = np.polyval(target_poly_der, state[0])
            theta_target = math.atan2(dy_dx, 1.0)

            y_err   = abs(state[1] - y_target)
            psi_err = abs((state[2] - theta_target + math.pi) % (2 * math.pi) - math.pi)
            
            scenario_error += (y_err + psi_err)
            actual_steps   += 1

            # terminazione anticipata se si cade dal rettilinio
            if abs(state[1]) > self.track_half_width:
                break

        # Penalità unificata per gli step mancanti
        PENALTY_PER_MISSED_STEP = 5.0
        missed_steps = expected_steps - actual_steps
        total_scenario_error = scenario_error + (missed_steps * PENALTY_PER_MISSED_STEP)

        return total_scenario_error, expected_steps


# Strategia 2 — Valutazione online su Gazebo

class OnlineGazeboFitnessStrategy(FitnessStrategy):
    """
    Valuta i pesi MPC facendo girare il robot reale in Gazebo.

    Protocollo di comunicazione con il nodo ROS (GeneticLineTracker in modalità TRAIN):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  optimizer.py                     GeneticLineTracker (TRAIN mode)    │
    │                                                                      │
    │  1. pubblica /mpc_weights ──────────────────────────────────────►    │
    │     [w_d, w_psi, w_effort, ...]   carica i pesi nel Model            │
    │                                   resetta stato interno              │
    │                                   resetta la posizione in Gazebo     │
    │                                   gira per eval_duration secondi     │
    │                                   accumula errore reale              │
    │  2. attende /mpc_score  ◄──────────────────────────────────────      │
    │     1 / (mean_error + 0.001)      pubblica fitness                   │
    │                                                                      │
    |                                                                      |
    │  3. ritorna fitness al GA                                            │
    └──────────────────────────────────────────────────────────────────────┘
    _______________________________________________________________________________
    Parametri configurabili
        eval_duration   : secondi di simulazione Gazebo per individuo
        score_timeout   : secondi massimi di attesa per /mpc_score prima di penalizzare
        weights_topic   : topic su cui pubblicare i pesi
        score_topic     : topic da cui leggere la fitness
    """

    def __init__(
        self,
        eval_duration:  float = 20.0,
        score_timeout:  float = 40.0,
        weights_topic:  str   = "/mpc_weights",
        score_topic:    str   = "/mpc_score",
    ):
        self.eval_duration  = eval_duration
        self.score_timeout  = score_timeout
        self.weights_topic  = weights_topic
        self.score_topic    = score_topic

        # ROS2 viene inizializzato lazy al primo evaluate()
        self._ros_ready   = False
        self._node        = None
        self._pub_weights = None
        self._last_score  = None
        self._score_lock  = threading.Lock()
        self._score_event = threading.Event()

        self._init_ros()

    @property
    def name(self) -> str:
        return "OnlineGazebo"
    
    def log(self, msg: str):
        """Sovrascrive il log standard usando il logger nativo di ROS 2."""
        if self._node is not None:
            self._node.get_logger().info(msg)
        else:
            print(msg, flush=True)

    def _init_ros(self):
        """Inizializza il nodo ROS 2 minimalista per la comunicazione."""
        if self._ros_ready:
            return

        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64, Float64MultiArray

        if not rclpy.ok():
            rclpy.init()

        class _OptimizerBridge(Node):
            def __init__(bridge_self):
                super().__init__("ga_optimizer_bridge")

                bridge_self.pub = bridge_self.create_publisher(
                    Float64MultiArray, self.weights_topic, 1
                )
                bridge_self.sub = bridge_self.create_subscription(
                    Float64,
                    self.score_topic,
                    bridge_self._on_score,
                    1,
                )

            def _on_score(bridge_self, msg):
                with self._score_lock:
                    self._last_score = msg.data
                self._score_event.set()

        self._node        = _OptimizerBridge()
        self._pub_weights = self._node.pub
        self._spin_thread = threading.Thread(
            target=self._spin_forever, daemon=True
        )
        self._spin_thread.start()
        self._ros_ready = True

        # Attesa di 2.5 secondi per permettere ai topic ROS 2 di agganciarsi
        import time
        time.sleep(2.5)

    def _spin_forever(self):
        import rclpy
        rclpy.spin(self._node)

    def evaluate(self, weights: List[float]) -> float:
        """
        Invia i pesi al nodo ROS, attende la fitness reale da Gazebo.
        Se il nodo non risponde entro score_timeout, ritorna penalità minima.
        """
        self._init_ros()

        from std_msgs.msg import Float64MultiArray

        # Resetta l'evento prima di pubblicare
        self._score_event.clear()
        with self._score_lock:
            self._last_score = None

        # Pubblica i pesi al nodo ROS
        msg      = Float64MultiArray()
        msg.data = [float(w) for w in weights]
        self._pub_weights.publish(msg)

        # Attendi la risposta entro score_timeout
        received = self._score_event.wait(timeout=self.score_timeout)

        if not received or self._last_score is None:
            # Timeout: individuo penalizzato come se fosse uscito subito
            return 0.0001

        return float(self._last_score)