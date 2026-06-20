#!/usr/bin/env python3
"""
optimizer.py
============
Algoritmo Genetico per il tuning dei pesi MPC.

Uso
---
# Training offline (default, nessun Gazebo necessario):
    python3 optimizer.py

# Training online su Gazebo (richiede Gazebo + nodo ROS in modalità TRAIN):
    python3 optimizer.py --strategy online

# Riprendi da un checkpoint:
    python3 optimizer.py --resume GeneticAlgorithmWeights

Opzioni
-------
--strategy  offline | online        (default: offline)
--resume    <percorso .pkl>         carica checkpoint PyGAD e continua
--seed      <int>                   seed NumPy per riproducibilità (default: 42)
--output    <nome file senza ext>   nome base per .pkl e .json (default: GeneticAlgorithmWeights)
"""

import argparse
import json
import sys

import numpy as np
import pygad

from fitness_strategy import FitnessStrategy, OfflineFitnessStrategy, OnlineGazeboFitnessStrategy
# Configurazione GA: indipendente dalla strategia
GENE_SPACE = [
    {"low": 1.0,  "high": 15.0},  # 0: w_d              - errore posizione laterale
    {"low": 1.0,  "high": 15.0},  # 1: w_psi            - errore orientamento (floor 1.0)
    {"low": 0.01, "high": 0.5},   # 2: w_effort         - penalità sterzata
    {"low": 5.0,  "high": 25.0},  # 3: k_curve          - costante di frenata in curva
    {"low": 0.01, "high": 1.0},   # 4: w_v              - penalità velocità
    {"low": 1.0,  "high": 5.0},   # 5: qf_mult_d        - moltiplicatore terminal cost su d
    {"low": 1.0,  "high": 5.0},   # 6: qf_mult_psi      - moltiplicatore terminal cost su psi
    {"low": 0.5,  "high": 2.0},   # 7: gamma_decay_start- metri di fiducia sulla curvatura
    {"low": 10.0, "high": 40.0}   # 8: horizon          - passi di previsione nel futuro
]

NUMBER_OF_THREAD = 4

class _FitnessWrapper:
    """
    Wrapper picklable attorno a una FitnessStrategy.
    Definita a top-level così multiprocessing può serializzarla su Windows
    (spawn) oltre che su Linux (fork).
    """
    def __init__(self, strategy: FitnessStrategy, total_gens: int, pop_size: int):
        self.strategy = strategy
        self.total_gens = total_gens
        self.pop_size = pop_size
 
    def __call__(self, ga_instance, solution, solution_idx):
        fitness = self.strategy.evaluate(solution.tolist())

        #log della fitness calcolata
        gen = ga_instance.generations_completed + 1
        ind = solution_idx + 1 if isinstance(solution_idx, int) else "?"
        pesi = [round(x, 3) for x in solution]
        
        # 3. Creiamo una SINGOLA stringa completa (il \n alla fine è obbligatorio)
        msg = f"[Gen {gen}/{self.total_gens} | Ind {ind:2}/{self.pop_size}]\n\tPesi: {pesi} -> Fitness: {fitness:.4f}\n"
        
        # 4. Scrittura atomica thread-safe
        if hasattr(self.strategy, "_node") and getattr(self.strategy, "_node", None) is not None:
            # Se siamo online su Gazebo, usiamo il logger ROS 2 (senza il \n finale)
            self.strategy.log(msg.strip())
        else:
            # Se siamo offline, usiamo la scrittura atomica su terminale
            sys.stdout.write(msg)
            sys.stdout.flush()

        return fitness


GA_PARAMS = dict(
    num_generations     = 50,
    keep_elitism        = 2,
    num_parents_mating  = 4,
    sol_per_pop         = 10,
    num_genes           = len(GENE_SPACE),
    gene_space          = GENE_SPACE,
    crossover_type      = "uniform",
    mutation_type       = "adaptive",
    mutation_num_genes  = (3, 2),
    # parallel_processing viene aggiunto solo per la strategia offline
    # perché OnlineGazebo non è parallelizzabile (un solo Gazebo per processo)
)


# Stato stagnazione: condiviso tra generazioni via closure
class _StagnationGuard:
    """Rileva stagnazione e inietta diversità nella popolazione."""

    def __init__(self, strategy: FitnessStrategy, patience: int = 6, n_inject: int = 3, total_gens: int = 50):
        self.strategy   = strategy
        self.patience   = patience
        self.n_inject   = n_inject
        self.total_gens = total_gens
        self._counter   = 0
        self._last_best = 0.0

    def __call__(self, ga_instance):
        gen = ga_instance.generations_completed
        best_sol, best_fit, _ = ga_instance.best_solution()

        pop      = ga_instance.population
        std_devs = [round(float(np.std(pop[:, g])), 4) for g in range(len(GENE_SPACE))]
        weights  = [round(float(x), 3) for x in best_sol]

        # Stampa il riassunto a fine generazione in maniera sincrona
        msg = f">>> FINE GENERAZIONE {gen}/{self.total_gens} <<<\n\t| Miglior Fitness: {best_fit:.4f} |\n\t| Pesi ottimali: {weights} |\n\t| StdDev: {std_devs} |"
        self.strategy.log(msg)

        if abs(best_fit - self._last_best) < 1e-4:
            self._counter += 1
        else:
            self._counter = 0
        self._last_best = best_fit

        if self._counter >= self.patience:
            # Rimpiazza i peggiori individui (lascia intatti i keep_elitism migliori)
            for i in range(1, self.n_inject + 1):
                for g, space in enumerate(GENE_SPACE):
                    pop[-i, g] = np.random.uniform(space["low"], space["high"])
            self._counter = 0
            self.strategy.log(f"  --> [Stagnazione] Iniezione diversità")


# Builder del GA
def build_ga(strategy: FitnessStrategy) -> pygad.GA:
    """
    Costruisce l'istanza PyGAD configurata per la strategia scelta.
    La OnlineGazebo disabilita il parallel_processing perché un solo
    Gazebo non può valutare più individui contemporaneamente.
    """
    total_gens = GA_PARAMS["num_generations"]
    pop_size = GA_PARAMS["sol_per_pop"]

    stagnation = _StagnationGuard(strategy, patience=6, n_inject=3, total_gens=total_gens)

    fitness_wrapper = _FitnessWrapper(strategy, total_gens, pop_size)

    params = dict(GA_PARAMS)  # copia per non mutare il dict globale
    params["fitness_func"]    = fitness_wrapper
    params["on_generation"]   = stagnation

    if isinstance(strategy, OfflineFitnessStrategy):
        params["parallel_processing"] = ["process", NUMBER_OF_THREAD]
        print(f"[OffLine Strategy] Setted {(params['parallel_processing'])[1]} parallel threads")
    # OnlineGazebo: nessun parallel_processing : il GA gira sequenzialmente

    return pygad.GA(**params)


# Salvataggio checkpoint
def save_checkpoint(ga: pygad.GA, strategy: FitnessStrategy, output_name: str, seed: int):
    ga.save(output_name)

    solution, fitness, _ = ga.best_solution()
    metadata = {
        "strategy":                 strategy.name,
        "seed":                     seed,
        "solution":                 [round(float(x), 3) for x in solution],
        "solution_fitness":         float(fitness),
        "num_generations_completed": ga.generations_completed,
    }
    # parametri specifici della strategia offline per riproducibilità
    if isinstance(strategy, OfflineFitnessStrategy):
        metadata.update({
            "cmd_delay":        strategy.cmd_delay,
            "n_noise_samples":  strategy.n_noise_samples,
            "training_steps":   strategy.training_steps,
        })
    elif isinstance(strategy, OnlineGazeboFitnessStrategy):
        metadata.update({
            "eval_duration": strategy.eval_duration,
        })

    meta_path = output_name + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCheckpoint salvato: {output_name}.pkl + {meta_path}")
    return metadata


# Entry point
def main():
    parser = argparse.ArgumentParser(description="GA optimizer per pesi MPC")
    parser.add_argument(
        "--strategy", choices=["offline", "online"], default="offline",
        help="offline = simulatore analitico | online = Gazebo reale (default: offline)"
    )
    parser.add_argument(
        "--resume", metavar="CHECKPOINT",
        help="Percorso al file .pkl PyGAD da cui riprendere il training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed NumPy per riproducibilità (default: 42)"
    )
    parser.add_argument(
        "--output", default="GeneticAlgorithmWeights",
        help="Nome base per i file di output .pkl e _meta.json"
    )
    parser.add_argument(
        "--eval_duration", type=float, default=20.0,
        help="Durata della valutazione su Gazebo (secondi)"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Costruisci la strategia scelta
    if args.strategy == "offline":
        strategy = OfflineFitnessStrategy()
    else:
        strategy = OnlineGazeboFitnessStrategy(
            eval_duration=args.eval_duration,
            score_timeout=args.eval_duration * 3.0 + 30.0
        )

    strategy.log(f"Strategia: {strategy.name}")
    strategy.log(f"Seed: {args.seed} | Output: {args.output}")

    # Carica da checkpoint o costruisci da zero
    if args.resume:
        strategy.log(f"Ripresa da checkpoint: {args.resume}")
        ga = pygad.load(args.resume)
        
        total_gens = GA_PARAMS["num_generations"]
        pop_size = GA_PARAMS["sol_per_pop"]
        
        ga.fitness_func   = _FitnessWrapper(strategy, total_gens, pop_size)
        ga.on_generation  = _StagnationGuard(patience=6, n_inject=3, total_gens=total_gens)
    else:
        ga = build_ga(strategy)

    # Training
    strategy.log(f"\nAvvio GA - strategia '{strategy.name}'")
    ga.run()

    solution, fitness, _ = ga.best_solution()
    strategy.log(f"\n== ALLENAMENTO COMPLETATO ==")
    strategy.log(f"Fitness finale:  {fitness:.4f}")
    strategy.log(f"Pesi ottimali [w_d, w_psi, w_effort, k_curve, w_v, qf_d, qf_psi, decay_start]:")
    strategy.log([round(float(x), 3) for x in solution])

    save_checkpoint(ga, strategy, args.output, args.seed)


if __name__ == "__main__":
    main()