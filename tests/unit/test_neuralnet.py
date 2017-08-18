import numpy as np
import theano

from delfi.neuralnet.NeuralNet import NeuralNet

dtype = theano.config.floatX


def test_lprobs():
    n_components = 2
    seed = 42
    svi = False

    nn = NeuralNet(n_components=n_components, n_hiddens=[10], n_inputs=1,
                   n_outputs=1, seed=seed, svi=svi)

    eval_lprobs = theano.function([nn.x, nn.y], nn.lprobs)

    res = eval_lprobs(np.array([[1.], [2.]], dtype=dtype),
                      np.array([[1.], [2.]], dtype=dtype))

    mog = nn.get_mog(np.array([[1.]]))
