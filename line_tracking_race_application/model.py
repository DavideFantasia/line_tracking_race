import numpy as np
from scipy.optimize import minimize
import math

class Model:
    def __init__(self, dt=0.05, horizon=10):
        self.dt = dt # Ts, Time di Sampling
        self.horizon = horizon # N passi di previsione nel futuro
        # pesi da ottimizzare tramite Genetica
        self.weight_d = 2.0     # peso errore distanza dalla linea
        self.weight_psi = 1.5   # peso errore orientamento
        self.weight_effort = 0.0# penalità per sterzate sgravate

        self.weight_v = 1.0     # penalità se il robot non va alla velocità desiderata
        self.v_target = 0.5     # velocità di crociera desiderata (m/s)

        # limiti fisici del veicolo (da car_gazebo_control.xacro)
        self.v_max = 0.5   # m/s
        self.w_max = 1.0   # rad/s

        self._last_solution = None
        
    def set_weights(self, weights):
        self.weight_d, self.weight_psi, self.weight_effort, self.weight_v = weights
        # Reset del warm-start quando i pesi cambiano (per il training)
        self._last_solution = None

    def cost_function(self, controls, current_state, target_line):
        """
        Simula 'horizon' passi nel futuro con Forward Euler e calcola il costo.
 
        Parametri
        ---------
        controls        : array [v0, w0, v1, w1, ..., vN, wN]
        current_state   : [x, y, theta] — stato attuale nel frame robot
        target_line     : coefficienti polinomio [a, b, c] per y = ax² + bx + c
        target_line_der : derivata precalcolata di target_line (evita ricalcolo nel loop)
        """
        cost = 0.0
        
        x, y, theta = current_state
        
        # scomposizione in coppie (v, w)
        v_seq = controls[0::2]
        w_seq = controls[1::2]

        # precalcolo della derivata della target line
        target_line_der = np.polyder(target_line)
        
        for i in range(self.horizon):
            v = v_seq[i]
            w = w_seq[i]
            
            # Modello Discretizzato (Forward Euler)
            # q(k + 1) = q(k) + Ts * S(q) * nu(k)
            x = x + self.dt * math.cos(theta) * v
            y = y + self.dt * math.sin(theta) * v
            theta = theta + self.dt * w
            
            # calcolo dell'errore (Path Following)
            # y_target = a*x^2 + b*x + c
            y_target = np.polyval(target_line, x)
            d = y - y_target

            # derivata del polinomio per trovare l'angolo della curva in quel punto
            # dy/dx = 2a*x + b
            dy_dx = np.polyval(target_line_der, x)
            theta_target = math.atan2(dy_dx,1.0)
            psi = (theta - theta_target + math.pi) % (2 * math.pi) - math.pi
            
            cost += self.weight_d * (d   ** 2)
            cost += self.weight_psi * (psi ** 2)
            cost += self.weight_effort * (w   ** 2)
            cost += self.weight_v * ((v - self.v_target) ** 2)
            
        return cost

    def solve(self, current_state, target_line):
        """Usa scipy.minimize per trovare i controlli (v, omega) ottimali"""
        # Guess iniziale dei controlli futuri
        target_line_der = np.polyder(target_line)  # precalcolato una volta sola
 
        # ------------------------------------------------------------------
        # Warm-start: se esiste una soluzione precedente, shiftala di 1 passo
        # [v0,w0, v1,w1, ..., vN,wN] → [v1,w1, ..., vN,wN, vN,wN]
        # ------------------------------------------------------------------
        if self._last_solution is not None and len(self._last_solution) == self.horizon * 2:
            controls0 = np.roll(self._last_solution, -2)
            # L'ultimo passo ripete l'ultimo comando (assunzione stazionaria)
            controls0[-2] = self._last_solution[-2]
            controls0[-1] = self._last_solution[-1]
        else:
            controls0          = np.zeros(self.horizon * 2)
            controls0[0::2]    = self.v_target
 
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0,          self.v_max))   # v ∈ [0, v_max]
            bounds.append((-self.w_max,  self.w_max))   # ω ∈ [-w_max, w_max]
 
        res = minimize(
            self.cost_function,
            controls0,
            args=(current_state, target_line),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-5}
        )
 
        self._last_solution = res.x  # salva per il warm-start al prossimo step
 
        return res.x[0], res.x[1]   # v_opt, w_opt