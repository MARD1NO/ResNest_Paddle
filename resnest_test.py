import paddle
from paddle import fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from resnest import *
import numpy as np



if __name__ == "__main__":
    with fluid.dygraph.guard():
        x = np.random.randn(1, 3, 224, 224).astype('float32')
        x = to_variable(x)
        net = resnest50(10)

        out = net(x)
        print(out.shape)
