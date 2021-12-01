import mmcv
import numpy as np
arr = np.random.randn(10, 10)
levels = 20
qarr = mmcv.quantize(arr, -1, 1, levels)