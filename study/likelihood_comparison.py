"""
Here we compare the Likelihood of the original and the new MultivariatePSD likelihood for the same data
"""
import jax.numpy as jnp
from jaxtyping import Array, Float
from GW150914_basic import original_likelihood
from GW1509814_multivarpsd import multivar_psd_likelihood


# parameter = [1,0,0,0]
# data = loda_data()
# assert Likelihood(data, parameter) == MultiavarPSDLikelihood(data, parameter)   

