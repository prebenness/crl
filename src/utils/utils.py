import jax.numpy as jnp

# Compute mean more efficiently
def jax_mean(l: list):
    return float(jnp.mean(jnp.array(l)))