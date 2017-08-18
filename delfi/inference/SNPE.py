import numpy as np
import theano
import theano.tensor as tt

from delfi.inference.BaseInference import InferenceBase
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.ops import distribution_pyop, kernel_pyop
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero

dtype = theano.config.floatX


class SNPE(InferenceBase):
    def __init__(self, generator, obs, convert_to_T=False, reg_lambda=100.,
                 seed=None, **kwargs):
        """Sequential neural posterior estimation (SNPE)

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_components : int
                    Number of components of the mixture density
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        super().__init__(generator, seed=seed, **kwargs)
        self.obs = obs
        self.reg_lambda = reg_lambda
        self.round = 0

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        # importance weights for loss
        if self.generator.proposal is not None:
            prior = distribution_pyop(self.generator.prior)
            proposal = distribution_pyop(self.generator.proposal)
            y_zt = tt.constant(self.params_std, dtype=dtype) * self.network.y \
                + tt.constant(self.params_mean, dtype=dtype)
            prior_eval = prior(y_zt)
            proposal_eval = proposal(y_zt)
        else:
            prior_eval = tt.constant(1.)
            proposal_eval = tt.constant(1.)
        iws = prior_eval / proposal_eval

        loss = -tt.mean(iws * self.network.lprobs)

        if self.svi:
            if self.round == 1:
                # keep weights close to zero-centered prior (first round)
                kl = svi_kl_zero(self.network.mps, self.network.sps,
                                 self.reg_lambda)
            else:
                # keep weights close to the init (i.e., those of previous
                # round)
                kl = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.iws'] = iws
        self.observables['loss.prior'] = prior_eval
        self.observables['loss.proposal'] = proposal_eval
        self.observables['loss.kl'] = kl

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=1000, minibatch=50,
            monitor=None, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int
            Number of data points drawn per round
        n_rounds : int
            Number of rounds
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets
        """
        logs = []
        trn_datasets = []

        for r in range(n_rounds):
            self.round += 1

            trn_data = self.gen(n_train)  # z-transformed params and stats

            t = Trainer(self.network, self.loss(N=n_train), trn_data,
                        seed=self.gen_newseed(),
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
            logs.append(t.train(epochs=epochs, minibatch=minibatch))
            trn_datasets.append(trn_data)

            # posterior becomes new proposal prior
            self.generator.proposal = self.predict(self.obs)  # see super

        return logs, trn_datasets
