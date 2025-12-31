"""
JAX implementation of non-bond potential.

see detail @ https://github.com/reaxnet/jax-nb
"""

from functools import partial

import jax
import numpy as np
from ase import Atoms
from jax import numpy as jnp
from jax_md import partition, space  # type: ignore

from .jax_nb import LAMBDA, nonbond_potential, pqeq_fori_loop
from .param_pqeq import pqeq_parameters


def nonbonded(
    atoms: Atoms,
    iforce: bool = False,
    ienergy: bool = False,
    r_cutoff: float = 12.5,
    pqeq_iterations: int = 2,
    capacity_multiplier: float = 2.0,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    if iforce:
        ienergy = True

    positions = jnp.array(atoms.get_scaled_positions())
    cell = jnp.array(atoms.cell.complete().array.T)
    symbols = atoms.get_chemical_symbols()

    displacement_fn, _ = space.periodic_general(
        box=cell,
        fractional_coordinates=True,
    )
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=cell,
        r_cutoff=r_cutoff,
        format=partition.Sparse,
        fractional_coordinates=True,
        capacity_multiplier=capacity_multiplier,
    )
    nbr = neighbor_fn.allocate(positions)

    rad = jnp.array([pqeq_parameters[s]["rad"] for s in symbols])
    alpha = 0.5 * LAMBDA / rad / rad
    alpha = jnp.sqrt(
        alpha.reshape(-1, 1)
        * alpha.reshape(1, -1)
        / (alpha.reshape(-1, 1) + alpha.reshape(1, -1))
    )
    chi0 = jnp.array([pqeq_parameters[s]["chi0"] for s in symbols])
    eta0 = jnp.array([pqeq_parameters[s]["eta0"] for s in symbols])
    z = jnp.array([pqeq_parameters[s]["Z"] for s in symbols])
    Ks = jnp.array([pqeq_parameters[s]["Ks"] for s in symbols])
    charges_fn = partial(
        pqeq_fori_loop,
        displacement_fn,
        alpha=alpha,
        cutoff=r_cutoff,
        iterations=pqeq_iterations,
        net_charge=atoms.get_initial_charges().sum(),
        eta0=eta0,
        chi0=chi0,
        z=z,
        Ks=Ks,
    )
    if not ienergy and not iforce:
        charges = jax.jit(charges_fn)(positions=positions, neighbor=nbr)[0]
        return np.asarray(charges), None, None  # type: ignore

    energy_fn_nb = partial(
        nonbond_potential,
        displacement_fn,
        # key args pqeq
        alpha=alpha,
        cutoff=r_cutoff,
        eta0=eta0,
        chi0=chi0,
        z=z,
        Ks=Ks,
        # key args d3
        atomic_numbers=jnp.array(atoms.numbers),
        compute_d3=False,
        # PBE zero damping parameters
        d3_params={
            "s6": 1.0,
            "rs6": 1.217,
            "s18": 0.722,
            "rs18": 1.0,
            "alp": 14.0,
        },
        damping="zero",
        smooth_fn=None,
    )

    def energy_fn(positions, nbr, **displ_kwargs):
        nbr = nbr.update(positions, **displ_kwargs)
        charges, r_shell = charges_fn(jax.lax.stop_gradient(positions), nbr)
        pe_nb = energy_fn_nb(positions, nbr, r_shell, charges, **displ_kwargs)
        return pe_nb, (charges, r_shell)

    value_and_grad_fn = jax.jit(
        jax.value_and_grad(
            partial(energy_fn, nbr=nbr),
            argnums=0,
            has_aux=True,
        )
    )
    results = value_and_grad_fn(positions, box=cell)
    charges = np.asarray(results[0][1][0])
    forces = np.asarray(-results[1])
    pe = np.asarray(results[0][0])
    return charges, float(pe), forces
