import paddle
from paddle import fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D
from paddle.fluid.layers import relu
from torch.nn.modules.utils import _pair


class SplAtConv2d(fluid.dygraph.Layer):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            # RFCONV？？
            pass
        else:
            self.conv = Conv2D(num_channels=in_channels,
                               num_filters=channels*radix,
                               filter_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups*radix)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        # relu激活
        # self.relu =

        self._adaptivePool = Pool2D(pool_type='avg', global_pooling=True)

        self.fc1 = Conv2D(channels, inter_channels, filter_size=1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2D(inter_channels, channels*radix, 1, groups=self.cardinality)
        if self.dropblock_prob > 0.0:
            # DROP BLOCK 代码
            pass
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            pass
            # DROP BLOCK 实现
            # x = self.dropblock()

        x = relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = fluid.layers.split(x, rchannel//self.radix, dim=1)
            gap = fluid.layers.sum(splited)
        else:
            gap = x
        gap = self._adaptivePool(gap)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        atten = fluid.layers.reshape(atten, shape=(batch, -1, 1, 1))

        if self.radix > 1:
            attens = fluid.layers.split(atten, rchannel//self.radix, dim=1)
            out = fluid.layers.sum(
                [fluid.layers.elementwise_mul(att, split)
                 for(att, split) in zip(attens, splited)]
            )
        else:
            out = fluid.layers.elementwise_mul(x, atten)

        return out


class rSoftMax(fluid.dygraph.Layer):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.shape[0]
        if self.radix > 1:
            x = fluid.layers.reshape(x, shape=(batch, self.cardinality, self.radix, -1))
            x = fluid.layers.transpose(x, [1, 2])
            x = fluid.layers.softmax(x)
            x = fluid.layers.reshape(x, shape=(batch, -1))
        else:
            x = fluid.layers.sigmoid(x)

        return x

