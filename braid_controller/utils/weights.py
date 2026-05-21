import numpy as np


def sigmoid(x: float, coeff_sharpness: float = 10, coeff_center: float = 0.9) -> float:
    """
    Sigmoid function that transitions from 0 to 1 around coeff_center with
    sharpness coeff_sharpness.

    Args:
        x (float): Input value.
        coeff_sharpness (float): Sharpness of the sigmoid transition.
        coeff_center (float): Center of the sigmoid transition.

    Returns:
        float: Output value of the sigmoid function.
    """
    return 1 / (1 + np.exp(-coeff_sharpness * (x - coeff_center)))
