import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl
import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import numpy as np
import theano
import theano.tensor as tt

from delfi.utils.odict import first, last, nth

dtype = theano.config.floatX


class NeuralNet(object):
    def __init__(self, n_inputs, n_outputs, n_components=1, n_hiddens=[50,50],
                 n_rnn=None, seed=None, svi=True):
        """Initialize a mixture density network with custom layers

        Parameters
        ----------
        n_inputs : int
            Dimensionality of input
        n_outputs : int
            Dimensionality of output
        n_components : int
            Number of components of the mixture density
        n_hiddens : list of ints
            Number of hidden units per layer
        n_rnn : None or int
            Number of RNN units
        seed : int or None
            If provided, random number generator will be seeded
        svi : bool
            Whether to use SVI version or not
        """
        self.n_components = n_components
        self.n_hiddens = n_hiddens
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_rnn = n_rnn
        self.svi = svi

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        lasagne.random.set_rng(self.rng)

        # placeholders
        # x : input placeholder, (batch, self.n_inputs)
        # y : output placeholder, (batch, self.n_outputs)
        self.x = tt.matrix('x', dtype=dtype)
        self.y = tt.matrix('y', dtype=dtype)

        # compose layers
        self.layer = collections.OrderedDict()

        # input layer
        self.layer['input'] = ll.InputLayer((None, self.n_inputs), input_var=self.x)
        # ... or substitute NaN for zero
        # ... or learn replacement values
        # ... or a recurrent neural net

        # hidden layers
        for l in range(len(n_hiddens)):
            self.layer['hidden_' + str(l+1)] = dl.FullyConnectedLayer(
                last(self.layer), n_units=n_hiddens[l],
                svi=svi, name='h' + str(l+1))
        last_hidden = last(self.layer)

        # mixture layers
        self.layer['mixture_weights'] = dl.MixtureWeightsLayer(last_hidden,
            n_units=n_components, actfun=lnl.softmax, svi=svi,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(last_hidden,
            n_components=n_components, n_dim=n_outputs, svi=svi,
            name='means')
        self.layer['mixture_precisions'] = dl.MixturePrecisionsLayer(
            last_hidden, n_components=n_components, n_dim=n_outputs, svi=svi,
            name='precisions')
        last_mog = [self.layer['mixture_weights'], self.layer['mixture_means'],
                    self.layer['mixture_precisions']]

        # mixture parameters
        # a : weights, matrix with shape (batch, n_components)
        # ms : means, list of len n_components with (batch, n_dim, n_dim)
        # Us : precision factors, n_components list with (batch, n_dim, n_dim)
        # ldetUs : log determinants of precisions, n_comp list with (batch, )
        self.a, self.ms, precision_out  = ll.get_output(last_mog,
                                                        deterministic=False)

        self.Us = precision_out['Us']
        self.ldetUs = precision_out['ldetUs']

        self.comps = {
            **{'a': self.a},
            **{'m'+str(i): self.ms[i] for i in range(self.n_components)},
            **{'U'+str(i): self.Us[i] for i in range(self.n_components)}}

        # log probability of y given the mixture distribution
        # lprobs_comps : log probs per component, list of len n_components with (batch, )
        # probs : log probs of mixture, (batch, )
        self.lprobs_comps = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle(
            [0, 'x', 1]) * U, axis=2)**2, axis=1) + ldetU \
            for m, U, ldetU in zip(self.ms, self.Us, self.ldetUs)]
        self.lprobs = tt.log(tt.sum(tt.exp(tt.stack(self.lprobs_comps, axis=1) \
            + tt.log(self.a)), axis=1)) - (0.5 * self.n_outputs * np.log(2*np.pi))

        # the quantities from above again, but with deterministic=True
        # --- in the svi case, this will disable injection of randomness;
        # the mean of weights is used instead
        self.da, self.dms, dprecision_out  = ll.get_output(last_mog,
                                                           deterministic=True)
        self.dUs = dprecision_out['Us']
        self.dldetUs = dprecision_out['ldetUs']
        self.dcomps = {
            **{'a': self.da},
            **{'m'+str(i): self.dms[i] for i in range(self.n_components)},
            **{'U'+str(i): self.dUs[i] for i in range(self.n_components)}}
        self.dlprobs_comps = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle(
            [0, 'x', 1]) * U, axis=2)**2, axis=1) + ldetU \
            for m, U, ldetU in zip(self.dms, self.dUs, self.dldetUs)]
        self.dlprobs = tt.log(tt.sum(tt.exp(tt.stack(self.dlprobs_comps, axis=1) \
            + tt.log(self.da)), axis=1)) - (0.5 * self.n_outputs * np.log(2*np.pi))

        # all parameters of network
        self.params = ll.get_all_params(last_mog)
        self.mps = ll.get_all_params(last_mog, mp=True)
        self.sps = ll.get_all_params(last_mog, sp=True)

        # theano functions
        self.compile_funs()

    def compile_funs(self):
        """Compiles theano functions"""
        self._f_eval_comps = theano.function(
            inputs=[self.x],
            outputs=self.dcomps)
        self._f_eval_lprobs = theano.function(
            inputs=[self.x, self.y],
            outputs=self.dlprobs)

    def eval_comps(self, x):
        """Evaluate the parameters of all mixture components at given inputs

        Parameters
        ----------
        x : np.array
            rows are input locations

        Returns
        -------
        mixing coefficients, means and scale matrices
        """
        return self._f_eval_comps(x.astype(dtype))

    def eval_lprobs(self, xy):
        """Evaluate log probabilities for given input-output pairs.

        Parameters
        ----------
        xy : np.array
            a pair (x, y) where x rows are inputs and y rows are outputs

        Returns
        -------
        log probabilities : log p(y|x)
        """
        x, y = xy
        return self._f_eval_lprobs(x.astype(dtype), y.astype(dtype))

    def get_mog(self, x, n_samples=None):
        """Return the conditional MoG at location x

        Parameters
        ----------
        x : np.array
            single input location
        n_samples : None or int
            ...
        """
        assert x.shape[0] == 1, 'x.shape[0] needs to be 1'

        comps = self.eval_comps(x)
        a = comps['a'][0]
        ms = [comps['m'+str(i)][0] for i in range(self.n_components)]
        Us = [comps['U'+str(i)][0] for i in range(self.n_components)]

        return dd.MoG(a=a, ms=ms, Us=Us, seed=self.gen_newseed())

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    @property
    def params_dict(self):
        """Getter for params as dict"""
        pdict = {}
        for p in self.params:
            pdict[str(p)] = p.get_value()
        return pdict

    @params_dict.setter
    def params_dict(self, pdict):
        """Setter for params as dict"""
        for p in self.params:
            if str(p) in pdict.keys():
                p.set_value(pdict[str(p)])
    @property
    def spec_dict(self):
        """Specs as dict"""
        return {'n_inputs' : self.n_inputs,
                'n_outputs' : self.n_outputs,
                'n_components' : self.n_components,
                'n_hiddens' : self.n_hiddens,
                'n_rnn' : self.n_rnn,
                'seed' : self.seed,
                'svi' : self.svi}

    def _consistent_init(self):
        # TODO: init according to prior on weights
        raise NotImplementedError
