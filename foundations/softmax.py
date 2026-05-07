import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_ = max(z)
        exp_z = np.e ** (z - max_)
        normalization = np.sum(exp_z)
        soft_max_ = exp_z / normalization
        return np.round(soft_max_, 4)
