import pygad
import numpy as np
import math
from model import Model
import json

RANDOM_SEED = 42

# deviazioni standard del rumore che simula l'incertezza della camera.
#   - coefficiente quadratico   (a): rumore piccolo, raramente sbaglia molto
#   - coefficiente lineare      (b): rumore medio, dipende dall'angolo di vista
#   - termine noto              (c): rumore più alto, offset laterale più incerto
NOISE_STD_A = 0.005
NOISE_STD_B = 0.02
NOISE_STD_C = 0.05
 
# ogni individuo viene valutato N_NOISE_SAMPLES volte con rumore diverso
# e si usa la media degli errori come fitness. Questo forza il GA a trovare
# pesi ROBUSTI invece di pesi ottimali solo in condizioni ideali.
# valore alto => migliora la robustezza | moltiplica il tempo di training.
N_NOISE_SAMPLES = 3
 
# simula la latenza ROS (topic /cmd_vel, scheduling del timer).
# Con dt=0.05s e CMD_DELAY=2, il robot esegue il comando calcolato 100ms prima.
CMD_DELAY = 2
 
TRAINING_STEPS = 25  # step per scenario — 25*0.05s = 1.25s di simulazione

# parametri per simulare e penalizzare l'uscita dal terreno
TRACK_HALF_WIDTH = 1.0
SOFT_MARGIN = 0.7       # inizia a penalizzare già oltre questo valore
OUT_PENALTY = 2.0
 
def simulate_scenario(mpc, initial_state, target_poly, noisy_poly):
    """
    Simula uno scenario con buffer di latenza e percezione rumorosa.
 
    Il robot calcola i comandi usando 'noisy_poly' (quello che "vede" dalla camera),
    ma si muove seguendo la cinematica reale. L'errore viene misurato rispetto
    a 'target_poly' (la linea reale, senza rumore).
 
    Ritorna l'errore totale accumulato (y_err + psi_err) sullo scenario.
    """
    state = initial_state.copy()
    
    # FIFO queue per la latenza dei comandi: i primi CMD_DELAY step
    # il robot sta fermo (comandi zero), poi inizia ad eseguire con ritardo.
    cmd_buffer = [(0.0, 0.0)] * CMD_DELAY
 
    scenario_error = 0.0
    target_poly_der = np.polyder(target_poly)  # derivata precalcolata della linea REALE
 
    for step in range(TRAINING_STEPS):
        # il robot "vede" la linea con rumore e calcola il comando
        v_cmd, w_cmd = mpc.solve(state, target_line=noisy_poly)
        
        # aggiunge il comando al buffer e preleva quello in ritardo
        cmd_buffer.append((v_cmd, w_cmd))
        v_applied, w_applied = cmd_buffer.pop(0)  # esegue il comando di CMD_DELAY step fa
 
        # kinematica con il comando ritardato
        state[0] += mpc.dt * math.cos(state[2]) * v_applied
        state[1] += mpc.dt * math.sin(state[2]) * v_applied
        state[2] += mpc.dt * w_applied
 
        # errore misurato sulla linea non rumorosa
        y_target    = np.polyval(target_poly, state[0])
        dy_dx       = np.polyval(target_poly_der, state[0])
        theta_target = math.atan2(dy_dx, 1.0)
 
        y_err   = abs(state[1] - y_target)
        psi_err = abs((state[2] - theta_target + math.pi) % (2 * math.pi) - math.pi)
        
        # penalita progressiva dalla soft margin in poi
        lateral_excess = max(0.0, abs(state[1]) - SOFT_MARGIN)

        scenario_error += y_err + psi_err + 5.0 * lateral_excess

        # terminazione anticipata oltre il bordo fisico
        if abs(state[1]) > TRACK_HALF_WIDTH:
            #errore proporzionale a quanto in fretta si butta di sotto
            remaining = TRAINING_STEPS - step - 1
            scenario_error += OUT_PENALTY * remaining
            break
 
    return scenario_error
 
 
def fitness_func(ga_instance, solution, solution_idx):
    """
    Funzione di fitness. 'solution' è l'array dei 4 pesi [w_d, w_psi, w_effort, w_v].
 
    Ogni individuo viene valutato su 4 scenari per N_NOISE_SAMPLES realizzazioni di rumore.
    La fitness è l'inverso dell'errore medio
    l'errore medio valutato su tutti gli scenari e su tutte le realizzazioni di rumore.
    """
    mpc = Model()
    mpc.set_weights(solution)
 
    total_error = 0.0
 
    for _ in range(N_NOISE_SAMPLES):
        
        # campiona una realizzazione di rumore per questa valutazione.
        noise_a = np.random.normal(0, NOISE_STD_A)
        noise_b = np.random.normal(0, NOISE_STD_B)
        noise_c = np.random.normal(0, NOISE_STD_C)
 
        def add_noise(poly):
            """Aggiunge rumore ai coefficienti [a, b, c] del polinomio."""
            return [poly[0] + noise_a, poly[1] + noise_b, poly[2] + noise_c]
 
        # ===========================================
        # SCENARIO 1 — rettilineo con offset laterale
        # -------------------------------------------
        # testa la capacità di tornare sulla linea 
        #           da una posizione sfasata.
        # ===========================================
        straight_poly = [0.0, 0.0, 0.0]
        total_error += simulate_scenario(
            mpc,
            initial_state=np.array([0.0, 0.5, 0.0]),
            target_poly=straight_poly,
            noisy_poly=add_noise(straight_poly)
        )
 
        # =================================================
        # SCENARIO 2 — curva parabolica con offset e angolo
        # -------------------------------------------------
        #   Testa il tracking su una curva con condizioni 
        #               iniziali non ideali.
        # =================================================
        curve_poly = [0.05, 0.0, 0.0]
        total_error += simulate_scenario(
            mpc,
            initial_state=np.array([0.0, 0.3, 0.2]),
            target_poly=curve_poly,
            noisy_poly=add_noise(curve_poly)
        )
 
        # =========================================
        # SCENARIO 3 — recupero con errore angolare
        # -----------------------------------------
        #   Testa la capacità di correggere 
        # l'orientamento senza offset laterale.
        # forza il GA a non trascurare w_psi.
        # =========================================
        total_error += simulate_scenario(
            mpc,
            initial_state=np.array([0.0, 0.0, 0.4]),
            target_poly=straight_poly,
            noisy_poly=add_noise(straight_poly)
        )
 
        # ===================================
        # SCENARIO 4 — Curva stretta (a=0.15)
        # ===================================
        tight_curve_poly = [0.15, 0.0, 0.0]
        total_error += simulate_scenario(
            mpc,
            initial_state=np.array([0.0, 0.2, 0.1]),
            target_poly=tight_curve_poly,
            noisy_poly=add_noise(tight_curve_poly)
        )
 
    # errore medio su tutti i sample e tutti gli scenari
    mean_error = total_error / N_NOISE_SAMPLES
 
    # per evitare casi di divisioni instabili
    if math.isnan(mean_error) or mean_error > 1000:
        return 0.0001
 
    return 1.0 / (mean_error + 0.001)
 
 
# =============================================================
# rilevamento stagnazione e iniezione di diversità
# =============================================================
_stagnation_counter = 0
_last_best_fitness  = 0.0
 
def on_generation(ga_instance):
    global _stagnation_counter, _last_best_fitness
 
    gen = ga_instance.generations_completed
    best_solution, best_fitness, _ = ga_instance.best_solution()
 
    pop = ga_instance.population
    std_devs = [float(round(np.std(pop[:, g]), 4)) for g in range(4)]
    formatted_weights = [float(round(x, 3)) for x in best_solution]
    print(f"[Gen {gen}] Fitness: {best_fitness:.4f} | Pesi: {formatted_weights} | StdDev: {std_devs}")
 
    # dopo un tot di generazioni senza miglioramento, inietta individui casuali
    # per rompere la convergenza prematura mantenendo i 2 migliori (keep_elitism).
    if abs(best_fitness - _last_best_fitness) < 1e-4:
        _stagnation_counter += 1
    else:
        _stagnation_counter = 0
    _last_best_fitness = best_fitness
 
    if _stagnation_counter >= 6:
        n_replace = 3  # rimpiazza 3 individui, partendo dai peggiori
        for i in range(1, n_replace + 1):
            for g, space in enumerate(gene_space):
                pop[-i, g] = np.random.uniform(space['low'], space['high'])
        _stagnation_counter = 0
        print(f"  --> Iniezione diversità attivata (stagnazione rilevata)")
 
 
# =============================================================
# Setup GA
# =============================================================
gene_space = [
    {'low': 0.5, 'high': 8.0},  # w_d
    {'low': 1.0, 'high': 8.0},  # w_psi — floor a 1.0 per evitare ratio w_d/w_psi >> 1
    {'low': 0.0, 'high': 0.5},  # w_effort
    {'low': 0.0, 'high': 2.0},  # w_v
]
 
ga_instance = pygad.GA(
    num_generations=50,
    keep_elitism=2,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=4,
    gene_space=gene_space,
    crossover_type="uniform",
    mutation_type="adaptive",
    mutation_num_genes=(2, 1),
    on_generation=on_generation,
    parallel_processing=["process", 8],
)
 
if __name__ == '__main__':
    print("Avvio GA — 4 scenari")
    print(f"  CMD_DELAY={CMD_DELAY} step | N_NOISE_SAMPLES={N_NOISE_SAMPLES} | TRAINING_STEPS={TRAINING_STEPS}")

    np.random.seed(RANDOM_SEED)

    ga_instance.run()
 
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"\n== ALLENAMENTO COMPLETATO ==")
    print(f"Fitness finale: {solution_fitness:.4f}")
    print(f"Pesi ottimali [w_d, w_psi, w_effort, w_v]:")
    print([float(round(x, 3)) for x in solution])
    
    model_weights_filename = "GeneticAlgorithmWeights"
    ga_instance.save(model_weights_filename)
    run_metadata = {
        "seed": RANDOM_SEED,
        "solution": [float(round(x, 3)) for x in solution],
        "solution_fitness": float(solution_fitness),
        "num_generations_completed": ga_instance.generations_completed,
        "cmd_delay": CMD_DELAY,
        "n_noise_samples": N_NOISE_SAMPLES,
        "training_steps": TRAINING_STEPS,
    }
    with open("ga_checkpoint_meta.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"saving the model parameters in {model_weights_filename} and ga_checkpoint_meta.json")