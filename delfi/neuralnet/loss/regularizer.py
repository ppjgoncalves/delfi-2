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

    logdet_Sigma1 = sum([tt.sum(2 * sp) for sp in sps])
    logdet_Sigma2 = sum([tt.sum(2 * sp_init) for sp_init in sps_init])

    tr_invSigma2_Sigma1 = sum([tt.sum(tt.exp(2 * (sp - sp_init)))
                               for sp, sp_init in zip(sps, sps_init)])

    quad_form = sum([tt.sum(((mp - mp_init)**2 / tt.exp(2 * sp_init)))
                     for mp, mp_init, sp_init in zip(mps, mps_init, sps_init)])

    L = 0.5 * (logdet_Sigma2 - logdet_Sigma1 -
               n_params + tr_invSigma2_Sigma1 + quad_form)

    return L


def svi_kl_zero_diag_gauss(mps_wp, sps_wp, mps_bp, sps_bp, a=1.0):
    """Regularization for SVI such that parameters stay close to zero

    The covariance matrix is a diagonal Gaussian. Entries on the diagonal are
    set according to 1/N_in where N_in is the number of incoming connections.
    For bias parameters, a*1/N_in.

    Parameters
    ----------
    mps_wp : list
        means of weight parameters
    sps_wp : list
        log stds of weight parameters
    mps_bp : list
        means of bias parameters
    sps_bp : list
        log stds of bias parameters
    a : float
        multiplies variance of bias parameters

    Returns
    -------
    regularizer node
    """
    n_params = sum([mp.get_value().size for mp in mps_wp]) + \
        sum([mp.get_value().size for mp in mps_bp])

    mps_init = [theano.shared(0. * mp.get_value()) for mp in mps_wp] + \
               [theano.shared(0. * mp.get_value()) for mp in mps_bp]
    sps_init = [theano.shared(0. * sp.get_value() +
                              np.log(np.sqrt(1.0 / sp.get_value().shape[0])))
                for sp in sps_wp] + \
               [theano.shared(0. * sp.get_value() +
                              a * np.log(np.sqrt(1.0 / sp.get_value().shape[0])))
                for sp in sps_bp]

    mps = mps_wp + mps_bp
    sps = sps_wp + sps_bp

    logdet_Sigma1 = sum([tt.sum(2 * sp) for sp in sps])
    logdet_Sigma2 = sum([tt.sum(2 * sp_init) for sp_init in sps_init])

    tr_invSigma2_Sigma1 = sum([tt.sum(tt.exp(2 * (sp - sp_init)))
                               for sp, sp_init in zip(sps, sps_init)])

    quad_form = sum([tt.sum(((mp - mp_init)**2 / tt.exp(2 * sp_init)))
                     for mp, mp_init, sp_init in zip(mps, mps_init, sps_init)])

    L = 0.5 * (logdet_Sigma2 - logdet_Sigma1 -
               n_params + tr_invSigma2_Sigma1 + quad_form)

    return L


def svi_kl_zero(mps, sps, wdecay):
    """Default regularization for SVI

    Prior is a spherical zero-centered Gauss whose precision corresponds to the
    weight decay parameter

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

    L1 = 0.5 * wdecay * (sum([tt.sum(mp**2) for mp in mps]) +
                         sum([tt.sum(tt.exp(sp * 2)) for sp in sps]))
    L2 = sum([tt.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))

    L = L1 - L2 - Lc

    return L
