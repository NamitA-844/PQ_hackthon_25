import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from divi.qprog import VQE, HartreeFockAnsatz, UCCSDAnsatz, GenericLayerAnsatz
from divi.qprog.optimizers import *
from divi.backends import *
from divi.qprog.workflows import *
from divi.qprog.algorithms import *
import time
import matplotlib.pyplot as plt


# first define the molecule via geometry 
mol = qml.qchem.Molecule(
    symbols=["H", "H"],
    coordinates=np.array([(0, 0, -0.74), (0, 0, 0.74)]),
    unit = 'bohr'
)

# construct the molecular hamiltonian with the molecule as an input
# for H2, each hydrogen atom has 1s orbital, and form bonding and anti bonding orbital 
# This is H(x)
hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
    mol, 
    active_electrons = 2,
    active_orbitals = 2
)

# Create molecule transformer for bond length variations
transformer = MoleculeTransformer(
    base_molecule=mol,
    bond_modifiers=np.linspace(-0.3, 0.3, 10)
)

# Defining the possible optmizers 
mc_optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3, keep_best_params=True)
l_bfgs_optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
nm_optimizer = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)
cmaes_optimizer = PymooOptimizer(method=PymooMethod.CMAES)
coby_optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

# optimizers list
optimizer_sweep_list = [mc_optimizer, nm_optimizer, l_bfgs_optimizer, cmaes_optimizer]

# Defining the different Ansatz which creates the first ground state psi(theta) 
# theta is the parameter 
# first the most used and grdaient free ansatz
base_ansatze = [UCCSDAnsatz, HartreeFockAnsatz]  

# now Generic Layer Ansatz
layer_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ, qml.RX],
    entangler=qml.CNOT,
    entangling_layout="all_to_all"
)

minimalist_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY],
    entangler=qml.CNOT,
    entangling_layout="linear"
)

balanced_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ],
    entangler=qml.CNOT,
    entangling_layout="brick"
)

expressive_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ, qml.RX],
    entangler=qml.CNOT,
    entangling_layout="all_to_all"
)

layer_ansatze = [minimalist_ansatz, balanced_ansatz, expressive_ansatz]

# Define which ansätze and which number of layers to test
# classes, not instances
layer_list = [1, 2, 3]  # sweep over this

# for Layered ansatz, defining a function to instantiate 

def build_layered_ansatze():
    """Create a list of instantiated ansatze for all layer counts."""
    instantiated = []
    for AnsatzClass in base_ansatze:
        for n in layer_list:
            try:
                # Some ansätze (like HF) may ignore n_layers
                ans = AnsatzClass(n_layers=n)
            except TypeError:
                # fallback: ansatz does not support layers
                ans = AnsatzClass()
            instantiated.append(ans)

    for ansatz in layer_ansatze:
        for n in layer_list:
            # Create a copy with repeated layers
            # GenericLayerAnsatz supports n_layers as a kwarg in build, but the object itself doesn't need n_layers
            instantiated.append(ansatz)
    return instantiated

# defining a function to sweep the bond_length, ansatz and optimizers 
def vqe_sweep_gen(optimizer):
    vqe_sweep = VQEHyperparameterSweep(
        molecule_transformer=transformer,
        ansatze=build_layered_ansatze(),
        optimizer_list=optimizer,
        max_iterations=10,
        backend=ParallelSimulator(shots=200),
        grouping_strategy="wires"
    )
    return vqe_sweep

all_sweep_results = []
times_list = []

ti = time.time()

for optim in optimizer_sweep_list:
    print(f"Running sweep for optimizer: {type(optim).__name__}")

    vqe_sweep = vqe_sweep_gen(optimizer=optim)

    vqe_sweep.create_programs()
    vqe_sweep.run(blocking=True)
    vqe_sweep.aggregate_results()

    # Aggregate results and store best params for each ansatz/layer
    ansatz_results = []
    for program_id, program in vqe_sweep._programs.items():
        ansatz_type, n_layers = program_id

        ansatz_results.append({
            "ansatz": ansatz_type,
            "layers": n_layers,
            "best_params": program.best_params,
            "best_energy": min(program.losses_history[-1].values()) 
                           if program.losses_history else None
        })

    sweep_result = {
        "optimizer_name": type(optim).__name__,
        "aggregated_results": vqe_sweep.aggregate_results(),
        "bond_modifiers": transformer.bond_modifiers,
        "ansatz_results": ansatz_results,
        "total_circuits": vqe_sweep.total_circuit_count
    }

    all_sweep_results.append(sweep_result)

    tf = time.time()
    times_list.append(tf - ti)

    vqe_sweep.visualize_results()
    print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")

# all_sweep_results

#plotting these results in one
plt.figure(figsize=(10, 6))

for sweep in all_sweep_results:

    optimizer_name = sweep["optimizer_name"]
    results = sweep["ansatz_results"]

    # Extract bond modifier (misnamed "layers") and energy
    x_vals = [entry["layers"] for entry in results]
    y_vals = [entry["best_energy"] for entry in results]

    # Sort so lines are drawn correctly
    x_sorted, y_sorted = zip(*sorted(zip(x_vals, y_vals)))

    plt.plot(x_sorted, y_sorted, marker='o', label=optimizer_name)

plt.xlabel("Bond modifier (sweep variable)")
plt.ylabel("Best VQE Energy (Hartree)")
plt.title("VQE Sweep Across Bond Modifiers — All Optimizers")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# doing the same with now warm setting where parameters are used for next iteration 
# but now we know the best Ansatz as the UCCSDAnsatz
layer_list = [1, 2, 3]
base_ansatze = [UCCSDAnsatz()]

all_sweep_results = []
times_list = []

prev_params = None   # warm-start memory

ti = time.time()

def vqe_sweep_ucc(optimizer):
    vqe_sweep = VQEHyperparameterSweep(
        molecule_transformer=transformer,
        ansatze=base_ansatze,
        optimizer_list=[optimizer],
        max_iterations=10,
        backend=ParallelSimulator(shots=200),
        grouping_strategy="wires"
    )
    return vqe_sweep

for optim in optimizer_sweep_list:
    print(f"Running sweep for optimizer: {type(optim).__name__}")

    vqe_sweep = vqe_sweep_ucc(optimizer=optim)

    # --------------------------------------------------------
    #(A) Warm start: assign initial parameters to programs
    # --------------------------------------------------------
    init_param_dict = {}
    if prev_params is not None:
        # every program_id gets the same initial guess
        for program_id, _ in vqe_sweep._programs.items():
            init_param_dict[program_id] = prev_params

    vqe_sweep.initial_params = init_param_dict
    # --------------------------------------------------------

    # run the sweep
    vqe_sweep.create_programs()
    vqe_sweep.run(blocking=True)
    vqe_sweep.aggregate_results()

    # Aggregate results and store best params for each ansatz/layer
    ansatz_results = []
    for program_id, program in vqe_sweep._programs.items():
        ansatz_type, n_layers = program_id
        ansatz_results.append({
            "ansatz": ansatz_type,
            "layers": n_layers,
            "best_params": program.best_params,
            "best_energy": program.best_loss
        })

    # --------------------------------------------------------
    # (B) Extract best parameters from this sweep
    # --------------------------------------------------------
    best_program = min(vqe_sweep._programs.values(), key=lambda p: p.best_loss)
    prev_params = best_program.best_params
    # --------------------------------------------------------

    sweep_result = {
        "optimizer_name": type(optim).__name__,
        "aggregated_results": vqe_sweep.aggregate_results(),
        "bond_modifiers": transformer.bond_modifiers,
        "ansatz_results": ansatz_results,
        "total_circuits": vqe_sweep.total_circuit_count
    }

    all_sweep_results.append(sweep_result)

    tf = time.time()
    times_list.append(tf - ti)

    vqe_sweep.visualize_results()
    print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for sweep in all_sweep_results:

    name = sweep["optimizer_name"]
    results = sweep["ansatz_results"]

    # Extract x = bond modifier, y = best energy
    x = [entry["layers"] for entry in results]      # misnamed "layers"
    y = [entry["best_energy"] for entry in results]

    # Sort by bond modifier so the line is smooth
    xy = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*xy)

    plt.plot(x_sorted, y_sorted, marker='o', label=name)

plt.xlabel("Bond modifier")
plt.ylabel("Best VQE Energy (Hartree)")
plt.title("VQE Sweep Across Bond Modifiers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

