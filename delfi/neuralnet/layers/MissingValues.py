import lasagne
import lasagne.layers as ll
import theano
import theano.tensor as tt

dtype = theano.config.floatX


class ImputeMissingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_inputs, R=lasagne.init.Normal(0.01), **kwargs):
        """Inputs that are NaN will be replaced by learned imputation value"""
        super(ImputeMissingLayer, self).__init__(incoming, **kwargs)
        self.R = self.add_param(R, (*n_inputs,), name='R')

    def get_output_for(self, input, **kwargs):
        input_nan = tt.cast(tt.isnan(input), dtype)
        input_not_nan = tt.cast(tt.invert(tt.isnan(input)), dtype)
        return input * input_nan + input_not_nan * self.R

    def get_output_shape_for(self, input_shape):
        return input_shape


class ReplaceMissingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super(ReplaceMissingLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        input_not_nan = tt.cast(tt.invert(tt.isnan(input)), dtype)
        return input * input_not_nan

    def get_output_shape_for(self, input_shape):
        return input_shape
