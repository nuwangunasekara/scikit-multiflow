import numpy as np

Sigmoid = 1
Tanh = 2
Relu = 3
LeakyRelu = 4


def derivative_of_relu(z):
    # g'(z) = 0 ; z < 0
    #       = 1 ; z >= 0
    return 1 if z >= 0 else 0


def derivative_of_leaky_relu(z):
    # g'(z) = 0.01 ; z < 0
    #       = 1 ; z >= 0
    return 1 if z >= 0 else 0.01


def map_class_to_label(y, class_to_label_map):
    return class_to_label_map[y]


dev_of_relu = np.vectorize(derivative_of_relu)
dev_of_leaky_relu = np.vectorize(derivative_of_leaky_relu)
vectorized_map_class_to_label = np.vectorize(map_class_to_label)


def auto_configure_layers(input_dimensions: int = None, power_skip_size=-1, return_basic: bool = False):
    if input_dimensions is None:
        return None
    layers_config = []
    log2_of_input_dimensions = np.log2(input_dimensions)
    if not return_basic and log2_of_input_dimensions > 1:
        layers_config.append({'neurons': 0, 'input_d': input_dimensions})
        if int(log2_of_input_dimensions//1 - power_skip_size) > power_skip_size:
            for i in range(int(log2_of_input_dimensions//1 - power_skip_size), 1, -power_skip_size):
                layers_config.append({'neurons': 2 ** i, 'g': Tanh})
        else:
            layers_config.append({'neurons': 4, 'g': Tanh})
        layers_config.append({'neurons': 1, 'g': Sigmoid})
    else:
        layers_config.append({'neurons': 0, 'input_d': input_dimensions})
        layers_config.append({'neurons': 4, 'g': Tanh})
        layers_config.append({'neurons': 1, 'g': Sigmoid})
    print("Network Structure= {}".format(layers_config))
    return layers_config
