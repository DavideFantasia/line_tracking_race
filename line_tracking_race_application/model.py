import numpy as np
from scipy.optimize import minimize
import math

class Model:
    def __init__(self, dt=1.0/10, horizon=20):
        self.dt = dt # Ts, Time di Sampling
        self.horizon = horizon # p passi di previsione nel futuro
        self.traj_poly = np.zeros(3) #polynomio della traiettoria, per calcolare la curvatura
        
        # pesi da ottimizzare tramite Genetica (Q e R)
        self.weight_d = (1/0.1)**2      # Q_d
        self.weight_psi = (1/0.1)**2    # Q_psi
        self.weight_effort = 1.0        # R_w
        self.weight_v = 0               # R_v
        self.k_curve = 2.0              # costante di frenata in curva

        self.Qf_multiplier_d   = 1.0
        self.Qf_multiplier_psi = 1.0
        
        self.v_target = 2.0             # velocità di crociera desiderata (m/s)

        self.gamma_decay_start = 1.5    # metri da cui iniziamo a non fidarci più di gamma_0
        self.gamma_decay_end   = 3.0    # oltre, assumiamo rettilineo 

        # limiti fisici del veicolo
        self.v_max = 2.0                # m/s
        self.w_max = 1.0                # rad/s

        self._last_solution = None
        
    def set_weights(self, weights):
        (self.weight_d, 
          self.weight_psi,
          self.weight_effort,
          self.k_curve,
          self.weight_v,
          self.Qf_multiplier_d,
          self.Qf_multiplier_psi,
          self.gamma_decay_start
        ) = weights
 
        # Margine fisso tra decay_start e decay_end: evita di esporre
        # un ottavo gene e garantisce sempre end > start per costruzione.
        self.gamma_decay_end = self.gamma_decay_start + 1.5

        # Reset del warm-start quando i pesi cambiano (per il training)
        self._last_solution = None
    
    @property
    def Qf_d(self):
        return self.Qf_multiplier_d * self.weight_d
 
    @property
    def Qf_psi(self):
        return self.Qf_multiplier_psi * self.weight_psi
 
    # Approssimazione di γ(σ)= ∂C(σ)​/∂σ
    def gamma_function(self, sigma, gamma_0):
        """
        Calcola la curvatura esatta basata sul polinomio estratto dalla visione.
        Decade a 0 solo oltre i limiti di visibilità reali.
        """
        a,b,c = self.traj_poly

        if sigma <= self.gamma_decay_start:
            return gamma_0

        if sigma >= self.gamma_decay_end:
            return 0.0
            
        # Calcolo della curvatura usando i coefficienti a, b
        gamma = (2 * a) / math.pow(1 + (2 * a * sigma + b)**2, 1.5)
        
        # Oltre un certo limite (es. 1.5m), ci si fida meno del fit e sfuma verso 0
        t = (sigma - self.gamma_decay_start) / (self.gamma_decay_end - self.gamma_decay_start)
        return gamma * (1.0 - t)


    def cost_function(self, controls, current_state, gamma_0):
        """
        Calcola il costo J = l_f(x_{k+p+1}) + sum(l(x_i, u_i))
        usando il modello cinematico Path Following (sigma, d, psi).
        """
        cost = 0.0
        sigma, d, psi = current_state
        gamma = gamma_0
        
        # scomposizione in coppie (v, w)
        v_seq = controls[0::2]
        w_seq = controls[1::2]

        for i in range(self.horizon):
            v = v_seq[i]
            w = w_seq[i]
            
            # Denominatore: 1 - (d * gamma)
            # per evitare singolarità (d * gamma == 1)
            denom = 1.0 - d * gamma
            #if abs(denom) < 1e-3:
            #    denom = 1e-3 if denom >= 0 else -1e-3
            denom = max(0.01, denom)

            # Modello Discretizzato (Forward Euler: q(k+1) = q(k) + Ts·q̇(k))
            # in Frenet-Serret
            sigma_next = sigma + self.dt * (v * math.cos(psi) / denom)
            d_next     = d     + self.dt * (v * math.sin(psi))

            psi_next   = psi   + self.dt * (w - (v * math.cos(psi) * gamma) / denom)
            
            sigma, d, psi = sigma_next, d_next, psi_next
            # approssimazione di x(σ)≈σ⋅cos(ψ)
            gamma = self.gamma_function(sigma*math.cos(psi), gamma_0) #aggiornamento di gamma nella previsione (col valore aggiornato di sigma)

            # Calcolo del running cost l(x, u)
            # l(x,u) = 1/2 (x_ref - x)^T Q (x_ref - x) + 1/2 u^T R u  + 1/2 W_v (v_target - v)^2
            #                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
            #                                                         Termine Extra per far 
            #                                                           accellerare il Robot
            # x_ref = [_, 0, 0] (vogliamo annullare l'errore laterale e angolare)
            cost += 0.5 * self.weight_d * (d ** 2)
            cost += 0.5 * self.weight_psi * (psi ** 2)
            cost += 0.5 * self.weight_effort * (w ** 2)

            break_factor = 1/(1.0 + self.k_curve * abs(gamma))
            cost += 0.5 * self.weight_v * ((self.v_target*break_factor - v) ** 2) # termine extra
            
        # Calcolo del terminal cost l_f(x_f)
        # Usiamo pesi maggiorati (Q_f) per penalizzare l'errore a fine orizzonte
        cost += 0.5 * self.Qf_d * (d ** 2)
        cost += 0.5 * self.Qf_psi * (psi ** 2)
            
        return cost

    def solve(self, current_state, gamma_0, traj_poly):
        """Trova i controlli (v, omega) ottimali minimizzando J"""

        # warm-start
        if self._last_solution is not None and len(self._last_solution) == self.horizon * 2:
            controls0 = np.roll(self._last_solution, -2)
            controls0[-2] = self._last_solution[-2]
            controls0[-1] = self._last_solution[-1]
        else:
            controls0 = np.zeros(self.horizon * 2)
            controls0[0::2] = self.v_target
 
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-0.5, 0.5))    # v in [-v_max, v_max]
            #bounds.append((-self.v_max, self.v_max))    # v in [-v_max, v_max]
            bounds.append((-self.w_max, self.w_max))    # w in [-w_max, w_max]
 
        self.traj_poly = traj_poly
        res = minimize(
            self.cost_function,
            controls0,
            args=(current_state, gamma_0),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 10, 'ftol': 1e-5}
        )

        if np.any(np.isnan(res.x)):
            self._last_solution = None  # Reset del warm-start corrotto
            return 0.0, 0.0

        self._last_solution = res.x  # salva per il warm-start al prossimo step
        return res.x[0], res.x[1]   # v_opt, w_opt