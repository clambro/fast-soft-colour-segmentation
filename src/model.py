import config
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D, InputLayer, Lambda, LeakyReLU, MaxPool2D,
                                     UpSampling2D)
from tensorflow.keras.models import Model


class FSCSModel:
    """"""

    def __init__(self, model_path=None):
        """"""
        pass

    def train(self, x_train, x_val):
        """"""
        pass

    def evaluate(self, x):
        """"""
        pass

    def optimize_for_one_image(self, img):
        """"""
        pass

    def save(self, path):
        """"""
        pass

    def _build_model(self):
        """"""
        input_layer = InputLayer((None, None, 3))

        pool = input_layer
        down_layers = []
        for i in range(config.NUM_DOWNSAMPLE_LAYERS):
            down_filters = config.BASE_FILTERS * 2 ** i
            layer, pool = self._do_down_conv(pool, down_filters)
            down_layers.append(layer)

        bottleneck_filters = config.BASE_FILTERS * 2 ** (config.NUM_DOWNSAMPLE_LAYERS + 1)
        layer = self._do_double_conv(pool, bottleneck_filters)

        for i in range(config.NUM_DOWNSAMPLE_LAYERS - 1):
            up_filters = config.BASE_FILTERS * 2 ** (config.NUM_DOWNSAMPLE_LAYERS - i)
            layer = self._do_up_conv(layer, down_layers.pop(-1), up_filters)

        output_layer = self._do_final_layer(layer, down_layers.pop(-1))

        model = Model(input_layer, output_layer)
        model.compile()
        return model

    def _do_down_conv(self, layer, num_filters):
        layer = self._do_double_conv(layer, num_filters)
        pool = MaxPool2D()(layer)
        return layer, pool

    def _do_up_conv(self, up_layer, skip_layer, num_filters):
        up_layer = UpSampling2D(up_layer)
        layer = Concatenate()([up_layer, skip_layer])
        return self._do_double_conv(layer, num_filters)

    def _do_final_layer(self, up_layer, skip_layer):
        up_layer = self._do_up_conv(up_layer, skip_layer.pop(-1), 4 * config.NUM_SEGMENTS)

        def normalize_final_layer(layer):
            alpha = tf.math.softmax(layer[..., :config.NUM_SEGMENTS])
            colours = tf.math.sigmoid(layer[..., config.NUM_SEGMENTS:])
            return tf.concat([alpha, colours])

        return Lambda(normalize_final_layer)(up_layer)

    @staticmethod
    def _do_double_conv(layer, num_filters):
        layer = Conv2D(num_filters, padding='same')(layer)
        layer = LeakyReLU(config.LEAKINESS)(layer)
        layer = Conv2D(num_filters, padding='same')(layer)
        layer = LeakyReLU(config.LEAKINESS)(layer)
        return BatchNormalization()(layer)
