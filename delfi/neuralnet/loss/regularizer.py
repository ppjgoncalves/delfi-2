import numpy as np
import theano
import theano.tensor as tt

def svi_kl_init(mps, sps):
    """Regularization for SVI such that parameters stay close to initial values

    Parameters
    ----------
    mps : list of means
    sps : list of log stds

    Returns
    -------
    regularizer node
    """
    n_params = sum([mp.get_value().size for mp in mps])

    mps_init = [theano.shared(mp.get_value()) for mp in mps]
    sps_init = [theano.shared(sp.get_value()) for sp in sps]

    logdet_Sigma1 = sum([tt.sum(2*sp) for sp in sps])
    logdet_Sigma2 = sum([tt.sum(2*sp_init) for sp_init in sps_init])

    tr_invSigma2_Sigma1 = sum([tt.sum(tt.exp(2*(sp - sp_init))) for sp, sp_init in zip(sps, sps_init) ])

    quad_form = sum([tt.sum(((mp - mp_init)**2 / tt.exp(2*sp_init))) for mp, mp_init, sp_init in zip(mps, mps_init, sps_init)])

    L = 0.5 * ( logdet_Sigma2 - logdet_Sigma1 - n_params + tr_invSigma2_Sigma1 + quad_form )

    return L

def svi_kl_zero(mps, sps, wdecay):
    """Default regularization for SVI

    We assume that the prior is a spherical zero-centred gaussian whose precision corresponds to the weight decay parameter.

    Parameters
    ----------
    mps : list of means
    sps : list of log stds
    wdecay : precision parameter (lambda)

    Returns
    -------
    regularizer node
    """
    assert wdecay > 0.0

    n_params = sum([mp.get_value().size for mp in mps])

    L1 = 0.5 * wdecay * (sum([tt.sum(mp**2) for mp in mps]) + sum([tt.sum(tt.exp(sp*2)) for sp in sps]))
    L2 = sum([tt.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))

    L = L1 - L2 - Lc

    return L
