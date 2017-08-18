import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.summarystats as ds
import numpy as np

from delfi.simulator.Gauss import Gauss


def test_trainer_updates():
    n_params = 2

    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # set up inference
    res = infer.Basic(g)

    # run with N samples
    out = res.run(100)
