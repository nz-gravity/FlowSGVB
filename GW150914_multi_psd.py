
import jax
import jax.numpy as jnp
import numpy as np
from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior
)
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
#from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################


# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = [H1, L1]

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomD(f_ref=20)

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior
M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
s1_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

# Defining Transforms

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
    BoundToUnbound(name_mapping = (["s1_z"], ["s1_z_unbounded"]) , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = (["s2_z"], ["s2_z_unbounded"]) , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
]


from jaxtyping import Array, Float
from jimgw.single_event.detector import Detector
import h5py

class multivariate_psd:
     def __init__(self, detectors: list[Detector], diagonal_psd: bool = True):
         
         if diagonal_psd:
            psd_values = jnp.stack([detector.psd for detector in detectors], axis=1)
            self.psd_matrix = jax.vmap(jnp.diag)(psd_values)

         else:
            self.psd_matrix = 0 # noise psd matrices estimated by SGVB

         self.inv_psd_matrix = jnp.linalg.inv(self.psd_matrix) 
         


def multivar_psd_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Float[Array, " n_dim"]],
    detectors: list[Detector],
    freqs: Float[Array, " n_dim"],
    align_time: Float,
    inverse_psd: multivariate_psd,
    **kwargs,
) -> Float:
    
    data_matrix = jnp.stack([detector.data for detector in detectors], axis=1)
    h_dec_matrix = jnp.stack(
        [detector.fd_response(freqs, h_sky, params) * align_time for detector in detectors],
        axis=1
    ) 
    
    residual_matrix = data_matrix - h_dec_matrix
    
    inv_psd_matrices = inverse_psd.inv_psd_matrix
    likelihood_contributions = jax.vmap(
        lambda res, inv_psd: (res.conj().T @ inv_psd @ res).real
    )(residual_matrix, inv_psd_matrices)
    
    energy_terms = jax.vmap(
        lambda data, inv_psd: -0.5 * (data.conj().T @ inv_psd @ data).real
    )(data_matrix, inv_psd_matrices)
    
    log_likelihood = -0.5 * jnp.sum(likelihood_contributions) - jnp.sum(energy_terms)
    
    return log_likelihood
    


class MultivariatePSDTransientLikelihoodFD(TransientLikelihoodFD):
    def __init__(self, *args, diagonal_psd=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.multivariate_psd = multivariate_psd(
            detectors=self.detectors,
            diagonal_psd=diagonal_psd
        )

        self.likelihood_function = lambda *args, **kwargs: multivar_psd_likelihood(
            *args, inverse_psd=self.multivariate_psd, **kwargs
        )



likelihood = MultivariatePSDTransientLikelihoodFD(
    ifos,
    prior=prior,
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_steps=5,
    popsize=10,
    diagonal_psd=True
)


mass_matrix = jnp.eye(prior.n_dim)
# mass_matrix = mass_matrix.at[1, 1].set(1e-3)
# mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

import optax

n_epochs = 20
n_loop_training = 100
total_epochs = n_epochs * n_loop_training
start = total_epochs // 10
learning_rate = optax.polynomial_schedule(
    1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_flow_sample=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    # strategies=[Adam_optimizer,"default"],
)


jim.sample(jax.random.PRNGKey(42))

samples = jim.get_samples()

parameter = np.column_stack([v for v in samples.values()])
labels = list(samples.keys())

with h5py.File("GW150914samples2.h5", "w") as hf:
    hf.create_dataset("parameter", data=parameter)
    hf.create_dataset("labels", data=np.string_(labels))





