import numpy as np
import theano
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero_diag_gauss
from delfi.utils.progress import no_tqdm, progressbar

dtype = theano.config.floatX


class SNPEexp(BaseInference):
    def __init__(self, generator, obs, prior_norm=True, pilot_samples=100,
                 recover_adam=True, retain_data=False, convert_to_T=False,
                 seed=None, verbose=True, **kwargs):
        """Sequential neural posterior estimation (SNPE)

        With experimental features

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        convert_to_T : bool or int
            Convert proposal distribution to Student's T? If a number if given,
            the number specifies the degrees of freedom
        recover_adam : bool
            If True, will recover state of adam when new round begins
        retain_data : bool
            If True, will do dataset retention
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
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
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)
        self.obs = obs
        self.round = 0
        self.convert_to_T = convert_to_T
        self.recover_adam = recover_adam
        self.retain_data = retain_data

        # placeholder for importance weights
        self.network.iws = tt.vector('iws', dtype=dtype)

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.iws * self.network.lprobs)

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.iws'] = self.network.iws

        if self.svi:
            if self.round == 1 or self.retain_data:
                # weights close to zero-centered prior in the first round
                kl, imvs = svi_kl_zero_diag_gauss(self.network.mps_wp,
                                                  self.network.sps_wp,
                                                  self.network.mps_bp,
                                                  self.network.sps_bp)
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            monitor=None, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
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
            Training datasets
        posteriors : list of distributions
            Posterior after each round
        """
        logs = []
        trn_datasets = []
        optim_state = []
        posteriors = []

        if not self.verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=n_rounds)
            desc = 'Round '
            pbar.set_description(desc)

        with pbar:
            for r in range(n_rounds):
                self.round += 1

                # if round > 1, set new proposal distribution before sampling
                if self.round > 1:
                    # posterior becomes new proposal prior
                    proposal = self.predict(self.obs)  # see super

                    # convert proposal to student's T?
                    if self.convert_to_T is not None:
                        if type(self.convert_to_T) == int:
                            dofs = self.convert_to_T
                        else:
                            dofs = 10
                        proposal = proposal.convert_to_T(dofs=dofs)

                    self.generator.proposal = proposal

                # number of training examples to generate for this round
                if type(n_train) == list:
                    try:
                        n_train_round = n_train[self.round-1]
                    except:
                        n_train_round = n_train[-1]
                else:
                    n_train_round = n_train

                # draw training data (z-transformed params and stats)
                verbose = '(round {}) '.format(self.round) if self.verbose else False
                trn_data = self.gen(n_train_round, verbose=False)

                # precompute importance weights
                iws = np.ones((n_train_round,))
                if self.generator.proposal is not None:
                    params = self.params_std * trn_data[0] + self.params_mean
                    p_prior = self.generator.prior.eval(params, log=False)
                    p_proposal = self.generator.proposal.eval(params, log=False)
                    iws *= p_prior / p_proposal

                trn_data = (trn_data[0], trn_data[1], iws)
                trn_datasets.append(trn_data)

                params_ = np.array([i for sub in trn_datasets for i in sub[0]])
                stats_ = np.array([i for sub in trn_datasets for i in sub[1]])
                iws_ = np.array([i for sub in trn_datasets for i in sub[2]])

                trn_data_round = (params_, stats_, iws_)

                trn_inputs = [self.network.params, self.network.stats,
                              self.network.iws]

                t = Trainer(self.network, self.loss(N=n_train_round),
                            trn_data=trn_data_round, trn_inputs=trn_inputs,
                            seed=self.gen_newseed(),
                            monitor=self.monitor_dict_from_names(monitor),
                            **kwargs)

                # recover adam state variables
                if self.recover_adam and len(optim_state) != 0:
                    for p, value in zip(t.updates.keys(), optim_state):
                        p.set_value(value)

                # train
                logs.append(t.train(epochs=epochs, minibatch=minibatch,
                                    verbose=verbose))

                # save state of optimizer
                optim_state = [p.get_value() for p in t.updates.keys()]

                # append posterior to list
                posteriors.append(self.predict(self.obs))

                pbar.update(1)

            return logs, trn_datasets, posteriors
