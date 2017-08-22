import delfi.neuralnet.DataStream as ds
import lasagne.updates as lu
import numpy as np
import pdb
import theano
import theano.tensor as tt

from delfi.utils.progress import progressbar

dtype = theano.config.floatX


class Trainer:
    def __init__(self, network, loss, trn_data, val_data=None,
                 step=lu.adam, max_norm=0.1, monitor=None, seed=None):
        """Construct and configure the trainer

        The trainer takes as inputs a neural network, a loss function and
        training data. During init the theano functions for training are
        compiled.

        Parameters
        ----------
        network : NeuralNet instance
            The neural network to train
        loss : theano variable
            Loss function to be computed for network training
        trn_data : array
            Training data in the form (params, stats)
        val_data : array
            Validation data in the form (params, stats)
        step : function
            Function to call for updates, will pass gradients and parameters
        max_norm : float
            Total norm constraint for gradients
        monitor : dict
            Dict containing theano variables (and names as keys) that should be
            recorded during training along with the loss function
        seed : int or None
            If provided, random number generator for batches will be seeded
        """
        self.network = network
        self.loss = loss

        # gradients
        grads = tt.grad(self.loss, self.network.params)
        if max_norm is not None:
            grads = lu.total_norm_constraint(grads, max_norm=max_norm)

        # updates
        self.updates = step(grads, self.network.params)

        # prepare train data
        trn_data_vars = [theano.shared(x.astype(dtype)) for x in trn_data]
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(
            n_trn_data_list) == 1, 'Number of train data is not consistent.'
        self.n_trn_data = list(n_trn_data_list)[0]

        # theano function for single batch update
        idx = tt.ivector('idx')
        trn_inputs = [self.network.y] + [self.network.x]  # trn_data is (params, stats)
        trn_inputs_data = [x[idx] for x in trn_data_vars]
        self.trn_outputs_names = ['loss']
        self.trn_outputs_nodes = [self.loss]
        if monitor is not None:
            monitor_names, monitor_nodes = zip(*monitor.items())
            self.trn_outputs_names += monitor_names
            self.trn_outputs_nodes += monitor_nodes
        self.make_update = theano.function(
            inputs=[idx],
            outputs=self.trn_outputs_nodes,
            givens=list(zip(trn_inputs, trn_inputs_data)),  # (x,y), (p[i],s[i])
            updates=self.updates
        )

        # if validation data is given, then set up validation too
        if val_data is not None:
            # prepare validation data
            n_val_data_list = set([x.shape[0] for x in val_data])
            assert len(
                n_val_data_list) == 1, 'Number of validation data is not consistent.'
            self.n_val_data = list(n_val_data_list)[0]
            val_data = [theano.shared(x.astype(dtype)) for x in val_data]

            # compile theano function for validation
            val_inputs = [
                model.input] if val_target is None else [
                model.input, val_target]
            self.validate = theano.function(
                inputs=[],
                outputs=val_loss,
                givens=list(zip(val_inputs, val_data))
            )
        else:
            self.validate = None

        # initialize variables
        self.loss = float('inf')
        self.loss_val = float('inf')
        self.loss_val_min = float('inf')
        self.loss_val_min_iter = None
        self.loss_val_min_params = None

        # pointers to model from self for better debugging
        self.idx = idx
        self.trn_inputs = trn_inputs
        self.trn_data = trn_data

        self.idx_stream = ds.IndexSubSampler(self.n_trn_data, seed=seed)

    def train(self,
              epochs=100,
              minibatch=50,
              monitor_every=None,
              pdb_iter=None,
              stop_on_nan=False,
              tol=None,
              tol_val=None,
              verbose=False):
        """Trains the model

        Parameters
        ----------
        epochs : int
            number of epochs (iterations per sample)
        minibatch : int
            minibatch size
        monitor_every : int
            monitoring frequency
        pdb_iter : int
            if set, will set a breakpoint after given number of iterations
        stop_on_nan : bool (default: False)
            if True, will stop if loss becomes NaN
        tol : float
            tolerance criterion for stopping based on training set
        tol_val : float
            tolerance criterion for stopping based on validation set
        verbose : bool
            if True, print progress during training

        Returns
        -------
        dict : containing loss values and possibly additional keys
        """
        maxiter = int(self.n_trn_data * epochs / minibatch)

        # initialize variables
        iter = 0

        progress_trn_iter = []
        progress_trn_val = []
        progress_trn_out = []
        progress_trn_obs = []

        progress_val_iter = []
        progress_val_val = []

        minibatch = self.n_trn_data if minibatch is None else minibatch
        monitor_every = float(
            'inf') if monitor_every is None else monitor_every

        trn_outputs = {}
        for key in self.trn_outputs_names:
            trn_outputs[key] = []

        # main training loop
        with progressbar(total=maxiter * minibatch) as pbar:
            pbar.set_description('Training ')

            for iter in range(maxiter):
                minibatch_idx = self.idx_stream.gen(minibatch)
                outputs = self.make_update(minibatch_idx)
                for name, value in zip(self.trn_outputs_names, outputs):
                    trn_outputs[name].append(value)

                trn_loss = trn_outputs['loss'][-1]
                diff = self.loss - trn_loss
                self.loss = trn_loss

                iter += 1

                if iter % monitor_every == 0:

                    if self.validate is not None:
                        progress_val_iter.append(iter)
                        val_loss = self.validate()
                        progress_val_val.append(val_loss)
                        diff_val = self.loss_val - val_loss
                        self.loss_val = val_loss

                        if self.loss_val < self.loss_val_min:
                            self.loss_val_min = self.loss_val
                            self.loss_val_min_iter = iter
                            self.loss_val_min_params = self.model.get_params()

                # check for convergence
                if tol is not None:
                    if abs(diff) < tol:
                        break

                # check for nan
                if stop_on_nan and np.isnan(trn_loss):
                    break

                # possible breakpoint
                if pdb_iter is not None and pdb_iter == iter:
                    mb = np.array(minibatch_idx)

                    def get_outputs(outputs, minibatch_idx=mb):
                        fun = theano.function(
                            inputs=[self.idx],
                            outputs=outputs,
                            givens=list(zip(self.trn_inputs,
                                            [x[self.idx] for x in
                                             self.trn_data])),
                            on_unused_input='ignore'
                        )
                        return fun(minibatch_idx.astype(np.int32))
                    go = get_outputs
                    pdb.set_trace()

                pbar.update(minibatch)

        # convert lists to arrays
        for name, value in trn_outputs.items():
            trn_outputs[name] = np.asarray(value)

        return trn_outputs
