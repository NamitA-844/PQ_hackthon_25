import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from divi.qprog import *
from divi.qprog.optimizers import *
from divi.backends import *
from divi.qprog.workflows import *
from divi.qprog.algorithms import *

# Define the active space for a 12-qubit simulation
# This freezes core 1s electrons and focuses on valence electrons
active_electrons = 8
active_orbitals = 6

# The two degenerate configurations of Ammonia (coordinates)
nh3_config1_coords = np.array(
    [
        (0, 0, 0), # N  
        (1.01, 0, 0), # H₁  
        (-0.5, 0.87, 0), # H₂  
        (-0.5, -0.87, 0) # H₃
    ]  
)

nh3_config2_coords = np.array(
    [
        (0, 0, 0),  # N (inverted)
        (-1.01, 0, 0),  # H₁
        (0.5, -0.87, 0),  # H₂
        (0.5, 0.87, 0),  # H₃
    ]
)

# Create molecule objects
nh3_molecule1 = qml.qchem.Molecule(
    symbols=["N", "H", "H", "H"],
    coordinates=nh3_config1_coords,
)

nh3_molecule2 = qml.qchem.Molecule(
    symbols=["N", "H", "H", "H"],
    coordinates=nh3_config2_coords,
)

# Build Hamiltonians with active space parameters
hamiltonian1, qubits = qml.qchem.molecular_hamiltonian(
    nh3_molecule1,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals,
)

hamiltonian2, qubits = qml.qchem.molecular_hamiltonian(
    nh3_molecule2,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals,
)

mol_transformer1 = MoleculeTransformer(
    base_molecule=nh3_molecule1, 
    bond_modifiers=[1], 
    atom_connectivity=[(0,1), (0,2), (0, 3)]
)

mol_transformer2 = MoleculeTransformer(
    base_molecule=nh3_molecule2, 
    bond_modifiers=[1], 
    atom_connectivity=[(0,1), (0,2), (0, 3)]
)

# Minimalist: few gates
minimalist_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY],
    entangler=qml.CNOT,
    entangling_layout="linear"
)

# Balanced: moderate complexity
balanced_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ],
    entangler=qml.CNOT,
    entangling_layout="brick"
)

balanced_ansatz2 = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ],
    entangler=qml.CNOT,
    entangling_layout="circular"
)

# Expressive: many gates
expressive_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ, qml.RX],
    entangler=qml.CNOT,
    entangling_layout="all_to_all"
)

n_layers = 2
n_qubits = qubits

ansatz_objects = [minimalist_ansatz, balanced_ansatz, balanced_ansatz2, expressive_ansatz]

# for the first molecular configuration 
optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
backend = ParallelSimulator(shots=200)
all_sweep_results = []
times_list = []

ti = time.time()

for ansatz in ansatz_objects:
    print(f"Running VQE with ansatz: {ansatz.name}")

    sweep = VQEHyperparameterSweep(
        hamiltonian=hamiltonian1,
        molecule=nh3_molecule1,
        molecule_transformer=mol_transformer1, 
        ansatze=[ansatz],       # list OK
        n_layers=1,           # sweep over layers
        optimizer_list=[optimizer],
        backend=ParallelSimulator(shots=200),
        max_iterations=20,
    )
    sweep.create_programs()
    sweep.run(blocking=True)
    #sweep.aggregate_results()

    # Collect ansatz type + layer metadata
    ansatz_metadata = []
    for a in sweep.ansatze:
        layer_attr = getattr(a, "n_layers", None)
        ansatz_metadata.append({
            "ansatz": type(a).__name__,
            "layers": layer_attr
        })

    sweep_result = {
        "optimizer_name": type(optimizer).__name__,
        "aggregated_results": sweep.aggregate_results(),
        "bond_modifiers": mol_transformer1.bond_modifiers,
        "ansatz_metadata": ansatz_metadata,
        "total_circuits": sweep.total_circuit_count
    }

    all_sweep_results.append(sweep_result)

    tf = time.time()
    times_list.append(tf - ti)

#all_sweep_results

def estimate_gate_count_generic(
    n_qubits: int,
    n_layers: int,
    n_single_gates_per_qubit_per_layer: int,
    layout: str,
):
    """Very rough gate-count estimate for layered hardware-efficient ansätze."""
    oneq_per_layer = n_qubits * n_single_gates_per_qubit_per_layer

    if layout == "linear":
        twoq_per_layer = max(0, n_qubits - 1)
    elif layout == "brick":
        # One CNOT per qubit on average
        twoq_per_layer = n_qubits
    elif layout == "circular":
        twoq_per_layer = n_qubits
    elif layout == "all_to_all":
        # Denser entangling (rough upper bound)
        twoq_per_layer = n_qubits * 2
    else:
        twoq_per_layer = n_qubits

    oneq = oneq_per_layer * n_layers
    twoq = twoq_per_layer * n_layers

    return {
        "1q": oneq,
        "2q": twoq,
        "total": oneq + twoq,
    }


def get_gate_info(ansatz_name, n_qubits, n_layers):
    """Provide (approximate) gate counts for efficiency ranking."""
    if ansatz_name == "MinimalRY":
        return estimate_gate_count_generic(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_single_gates_per_qubit_per_layer=1,
            layout="linear",
        )
    if ansatz_name == "BalancedRYRZ":
        return estimate_gate_count_generic(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_single_gates_per_qubit_per_layer=2,
            layout="brick",
        )
    if ansatz_name == "QCC_like":
        # More expressive rotations + denser entangling = QCC-flavored
        return estimate_gate_count_generic(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_single_gates_per_qubit_per_layer=2,
            layout="all_to_all",
        )

    # For UCCSD we skip detailed counting – it's large and chemistry-accurate
    return {"1q": None, "2q": None, "total": None}

# for the second molecular configuration 
optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
backend = ParallelSimulator(shots=200)
all_sweep_results2 = []
times_list = []

ti = time.time()

for ansatz in ansatz_objects:
    print(f"Running VQE with ansatz: {ansatz.name}")

    sweep = VQEHyperparameterSweep(
        hamiltonian=hamiltonian2,
        molecule=nh3_molecule2,
        molecule_transformer=mol_transformer2, 
        ansatze=[ansatz],       # list OK
        n_layers=1,           # sweep over layers
        optimizer_list=[optimizer],
        backend=ParallelSimulator(shots=200),
        max_iterations=20,
    )
    sweep.create_programs()
    sweep.run(blocking=True)
    #sweep.aggregate_results()

    # Collect ansatz type + layer metadata
    ansatz_metadata = []
    for a in sweep.ansatze:
        layer_attr = getattr(a, "n_layers", None)
        ansatz_metadata.append({
            "ansatz": type(a).__name__,
            "layers": layer_attr
        })

    sweep_result = {
        "optimizer_name": type(optimizer).__name__,
        "aggregated_results": sweep.aggregate_results(),
        "bond_modifiers": mol_transformer2.bond_modifiers,
        "ansatz_metadata": ansatz_metadata,
        "total_circuits": sweep.total_circuit_count
    }

    all_sweep_results2.append(sweep_result)

    tf = time.time()
    times_list.append(tf - ti)

#all_sweep_results2
