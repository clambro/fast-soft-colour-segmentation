import config
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model


class FSCSModel:
    """"""

    def __init__(self, model_path=None):
        """"""
        if model_path is None:
            self.model = self._build_model()
        else:
            self.model = load_model(model_path)

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
        input_layer = layers.Input((None, None, 3))

        pool = input_layer
        down_layers = []
        for i in range(config.NUM_DOWNSAMPLE_LAYERS):
            down_filters = config.BASE_FILTERS * 2 ** i
            layer, pool = self._do_down_conv(pool, down_filters)
            down_layers.append(layer)

        bottleneck_filters = config.BASE_FILTERS * 2 ** config.NUM_DOWNSAMPLE_LAYERS
        layer = self._do_double_conv(pool, bottleneck_filters)

        for i in range(config.NUM_DOWNSAMPLE_LAYERS - 1):
            up_filters = config.BASE_FILTERS * 2 ** (config.NUM_DOWNSAMPLE_LAYERS - i - 1)
            layer = self._do_up_conv(layer, down_layers.pop(-1), up_filters)

        output_layer = self._do_final_layer(layer, down_layers.pop(-1))

        model = Model(input_layer, output_layer)
        model.compile()
        return model

    def _do_down_conv(self, layer, num_filters):
        layer = self._do_double_conv(layer, num_filters)
        pool = layers.MaxPool2D()(layer)
        return layer, pool

    def _do_up_conv(self, up_layer, skip_layer, num_filters):
        up_layer = layers.UpSampling2D()(up_layer)
        layer = layers.Concatenate()([up_layer, skip_layer])
        return self._do_double_conv(layer, num_filters)

    def _do_final_layer(self, up_layer, skip_layer):
        up_layer = self._do_up_conv(up_layer, skip_layer, 4 * config.NUM_SEGMENTS)

        def normalize_final_layer(layer):
            alpha = tf.math.softmax(layer[..., :config.NUM_SEGMENTS])
            colours = tf.math.sigmoid(layer[..., config.NUM_SEGMENTS:])
            return tf.concat([alpha, colours], axis=-1)

        return layers.Lambda(normalize_final_layer)(up_layer)

    @staticmethod
    def _do_double_conv(layer, num_filters):
        layer = layers.Conv2D(num_filters, 3, padding='same')(layer)
        layer = layers.LeakyReLU(config.LEAKINESS)(layer)
        layer = layers.Conv2D(num_filters, 3, padding='same')(layer)
        layer = layers.LeakyReLU(config.LEAKINESS)(layer)
        return layers.BatchNormalization()(layer)
