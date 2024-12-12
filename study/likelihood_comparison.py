"""
Here we compare the Likelihood of the original and the new MultivariatePSD likelihood for the same data
"""


# parameter = [1,0,0,0]
# data = loda_data()
# assert Likelihood(data, parameter) == MultiavarPSDLikelihood(data, parameter)   



import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior, PowerLawPrior
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import ComponentMassesToChirpMassSymmetricMassRatioTransform, SkyFrameToDetectorFrameSkyPositionTransform, ComponentMassesToChirpMassMassRatioTransform
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
post_trigger_duration = 2
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
fmin = 20.0
fmax = 1024.0


ifos = [H1, L1]

# # TODO: dont downlaod in the future -- cache 

for ifo in ifos:
     ifo.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
m_1_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_max)[0], Mc_q_to_m1_m2(M_c_max, q_min)[0], parameter_names=["m_1"])
m_2_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_min)[1], Mc_q_to_m1_m2(M_c_max, q_max)[1], parameter_names=["m_2"])
s1z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        m_1_prior,
        m_2_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)


## TODO: can we use some delta functions for some of the priors? 

sample_transforms = [
    ComponentMassesToChirpMassMassRatioTransform,
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=q_min, original_upper_bound=q_max),
    BoundToUnbound(name_mapping = [["s1_z"], ["s1_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["s2_z"], ["s2_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["d_L"], ["d_L_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2000.0),
    BoundToUnbound(name_mapping = [["t_c"], ["t_c_unbounded"]] , original_lower_bound=-0.05, original_upper_bound=0.05),
    BoundToUnbound(name_mapping = [["phase_c"], ["phase_c_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["iota"], ["iota_unbounded"]], original_lower_bound=0., original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping = [["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    ComponentMassesToChirpMassSymmetricMassRatioTransform,
]


### JIANAN's TEST BELOW



import jax.numpy as jnp
from jaxtyping import Array, Float
from jimgw.single_event.detector import Detector
import numpy as np
from astropy.time import Time
from tqdm.auto import tqdm
import jax.numpy as jnp
from jaxtyping import Array, Float
from jimgw.single_event.detector import Detector
from jimgw.single_event.likelihood import TransientLikelihoodFD
import matplotlib.pyplot as plt

def original_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Float[Array, " n_dim"]],
    detectors: list[Detector],
    freqs: Float[Array, " n_dim"],
    align_time: Float,
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    df = freqs[1] - freqs[0]
    for detector in detectors:
        h_dec = detector.fd_response(freqs, h_sky, params) * align_time
        match_filter_SNR = (
            4 * jnp.sum((jnp.conj(h_dec) * detector.data) / detector.psd * df).real
        )
        optimal_SNR = 4 * jnp.sum(jnp.conj(h_dec) * h_dec / detector.psd * df).real
        log_likelihood += match_filter_SNR - optimal_SNR / 2

    return log_likelihood



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



basic_lnl = TransientLikelihoodFD(
    detectors=ifos,
    prior=prior,
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

new_lnl = MultivariatePSDTransientLikelihoodFD(
    detectors=ifos,
    prior=prior,
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    diagonal_psd=True
)



params =  {'ra': 3.0501547234835615, 'dec': 0.25627591999554844, 'psi': 0.31829725279217863, 'iota': 0.6587444630143763, 'phase_c': 4.0457167805377505, 't_c': 0.03135253197548338, 'd_L': 1305.0286912380118, 's2_z': -0.5964997353135959, 's1_z': 0.8158665162274288, 'M_c': 25.24749708900822, 'eta': 0.24949611706329067}

lnls = []
new_lnls = []

Mcs = np.linspace(15, 35, 10)
for mc in tqdm(Mcs):
  new_theta = {**params}
  new_theta['M_c'] = mc
  freq = Mcs
  h_sky = RippleIMRPhenomD()(frequency=freq, params=new_theta)
  # lnls.append(original_likelihood(new_theta, h_sky, ifos, freq, 0))
  lnls.append(basic_lnl.evaluate(new_theta, ifos))
  new_lnls.append(new_lnl.evaluate(new_theta, ifos))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(Mcs, lnls)
axes[0].set_title('Original Likelihood')
axes[0].set_xlabel('M_c')
axes[0].set_ylabel('Likelihood')

axes[1].plot(Mcs, new_lnls)
axes[1].set_title('Multivariate Likelihood')
axes[1].set_xlabel('M_c')
axes[1].set_ylabel('Likelihood')

plt.tight_layout()
plt.show()
#plt.savefig("likelihood_comparison.pdf", format="pdf")
