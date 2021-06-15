import config
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model


class FSCSModel:
    """The Fast Soft Colour Segmentation model.

    Attributes
    ----------
    model : Model
        The tensorflow model for segmentation.
    """

    def __init__(self, weight_path=None):
        """Initialize the FSCS model and optionally load saved weights.

        Parameters
        ----------
        weight_path : Optional[str]
            If given, the model will load its weights from this path. Otherwise they will be randomly generated.
        """
        self.model = self._build_model()
        if weight_path is not None:
            self.model.load_weights(weight_path)

    def train(self, x_train, x_val):
        """"""
        pass

    def evaluate(self, x):
        """"""
        pass

    def optimize_for_one_image(self, img):
        """"""
        pass

    def save_weights(self, path):
        """Save the model weights to file.

        Parameters
        ----------
        path : str
            The full path to which the weights will be saved.
        """
        self.model.save_weights(path)

    def _build_model(self):
        """Build the model

        Returns
        -------
        model : Model
            The tensorflow model for image segmentation.
        """
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
        return model

    def _do_down_conv(self, layer, num_filters):
        """Helper function for convolution and downsampling.

        Parameters
        ----------
        layer : Layer
            The layer to operate on.
        num_filters : int
            The number of filters for the convolutional layers.

        Returns
        -------
        layer : Layer
            The layer after convolution.
        pool : Layer
            The layer after convolution and pooling.
        """
        layer = self._do_double_conv(layer, num_filters)
        pool = layers.MaxPool2D()(layer)
        return layer, pool

    def _do_up_conv(self, up_layer, skip_layer, num_filters, is_final=False):
        """Helper function for convolution and upsampling.

        Parameters
        ----------
        up_layer : Layer
            The layer to upsample.
        skip_layer : Layer
            The layer to concatenate with the upsampled layer.
        num_filters : int
            The number of filters for the convolutional layers.
        is_final : bool
            Whether or not this is the final layer in the model (suppresses activation and normalization).

        Returns
        -------
        Layer
            The layer after upsampling, concatenation, and convolution.
        """
        up_layer = layers.UpSampling2D()(up_layer)
        layer = layers.Concatenate()([up_layer, skip_layer])
        return self._do_double_conv(layer, num_filters, is_final)

    def _do_final_layer(self, up_layer, skip_layer):
        """Helper function for the final layer which generates the outputs.

        Parameters
        ----------
        up_layer : Layer
            The layer to upsample.
        skip_layer : Layer
            The layer to concatenate with the upsampled layer.

        Returns
        -------
        Layer
            The final output layer with config.NUM_SEGMENTS alpha channels followed by the same number of RGB images
            concatenated along the channel axis.
        """
        up_layer = self._do_up_conv(up_layer, skip_layer, 4 * config.NUM_SEGMENTS, is_final=True)

        def normalize_final_layer(layer):
            alpha = tf.math.softmax(layer[..., :config.NUM_SEGMENTS])
            colours = tf.math.sigmoid(layer[..., config.NUM_SEGMENTS:])
            return tf.concat([alpha, colours], axis=-1)

        return layers.Lambda(normalize_final_layer)(up_layer)

    @staticmethod
    def _do_double_conv(layer, num_filters, is_final=False):
        """Helper function for double convolution.

        Parameters
        ----------
        layer : Layer
            The layer to convolve.
        num_filters : int
            The number of filters for the convolutional layers.
        is_final : bool
            Whether or not this is the final layer in the model (suppresses activation and normalization).

        Returns
        -------
        Layer
            The layer after convolution.
        """
        layer = layers.Conv2D(num_filters, 3, padding='same')(layer)
        layer = layers.LeakyReLU(config.LEAKINESS)(layer)
        layer = layers.Conv2D(num_filters, 3, padding='same')(layer)
        if is_final:
            return layer
        else:
            layer = layers.LeakyReLU(config.LEAKINESS)(layer)
            return layers.BatchNormalization()(layer)
