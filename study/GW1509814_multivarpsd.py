# MultivarDetector


def multivar_psd_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Float[Array, " n_dim"]],
    detectors: list[Detector],
    freqs: Float[Array, " n_dim"],
    align_time: Float,
    **kwargs,
) -> Float:
    
    '''
    PSD matrix construction for all detectors
    '''
    psd_matrices = []
    for i in range(len(freqs)):
        
        # For each frequency, construct a diagonal PSD matrix
        diag_psd = [detector.psd[i] for detector in detectors]
        psd_matrix = jnp.diag(jnp.array(diag_psd))
        psd_matrices.append(psd_matrix)
    
    '''
    residual matrix for all detectors
    '''
    residuals = []
    # Iterate through each detector
    for detector in detectors:
        h_dec = detector.fd_response(freqs, h_sky, params) * align_time
        # Subtract h_dec from the detector's data
        residual = detector.data - h_dec
        residuals.append(residual)
    
    # Concatenate all detector residuals into a single vector
    residual_matrix = jnp.stack(residuals, axis=1)   # Shape: (n_freqs, n_detectors)
    
    '''
    Compute the log-likelihood
    '''
    log_likelihood = 0.0
    df = freqs[1] - freqs[0]
    for i in range(len(freqs)):

        psd_matrix = psd_matrices[i]
        inv_psd_matrix = jnp.linalg.inv(psd_matrix)
    
        residual_vector = residual_matrix[i, :]
        likelihood_contribution = (
            residual_vector.conj().T @ inv_psd_matrix @ residual_vector
        ).real
    
    
        log_likelihood += -0.5 * likelihood_contribution * df
   
    return log_likelihood   




