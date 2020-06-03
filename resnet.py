import math
import paddle
from paddle import fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Sequential
from paddle.fluid.layers import relu

from splat import SplAtConv2d


class GlobalAvgPool2D(fluid.dygraph.Layer):
    """
    全局平均池化
    """

    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()
        self.pool = Pool2D(pool_type='avg', global_pooling=True)

    def forward(self, x):
        return self.pool(x)
"""
block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
"""

class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = Conv2D(inplanes, group_width, filter_size=1)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            # 平均池化
            self.avd_layer = Pool2D(pool_size=3, pool_stride=stride, pool_padding=1,
                                    pool_type='avg')
            stride = 1

        if dropblock_prob > 0.0:
            # 尚未实现dropblock
            pass

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob
            )
        elif rectified_conv:
            # 尚未实现rectified_conv
            pass
        else:
            self.conv2 = Conv2D(
                num_channels=group_width,
                num_filters=group_width,
                filter_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = Conv2D(
            num_channels=group_width,
            num_filters=planes * 4,
            filter_size=1,
        )
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            # 暂时不用
            pass

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            # 暂未实现dropblock
            pass
        out = fluid.layers.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                # 暂未实现
                pass
            out = fluid.layers.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = fluid.layers.elementwise_add(out, residual)
        out = fluid.layers.relu(out)

        return out


class ResNet(fluid.dygraph.Layer):
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=BatchNorm):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            # 暂未实现
            # from rfconv import RFConv2d
            # conv_layer = RFConv2d
            pass
        else:
            conv_layer = Conv2D

        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}

        if deep_stem:
            self.conv1 = Sequential(
                Conv2D(num_channels=3, num_filters=stem_width, filter_size=3, stride=2, padding=1),
                norm_layer(stem_width, act='relu'),
                Conv2D(num_channels=stem_width, num_filters=stem_width, filter_size=3, stride=1, padding=1),
                norm_layer(stem_width, act='relu'),
                Conv2D(num_channels=stem_width, num_filters=stem_width * 2, filter_size=3, stride=1, padding=1)
            )
        else:
            self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=7, stride=2, padding=3)

        self.bn1 = norm_layer(self.inplanes)

        self.maxpool = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.drop = True if final_drop > 0.0 else False
        self.avgpool = GlobalAvgPool2D()
        self.fc = Linear(512 * block.expansion, num_classes)

        # 下面是权重初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, norm_layer):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layer = []
            if self.avg_down:
                if dilation == 1:
                    down_layer.append(Pool2D(pool_size=stride, pool_stride=stride,
                                             ceil_mode=True))
                else:
                    down_layer.append(Pool2D(pool_size=1, pool_stride=1,
                                             ceil_mode=True))
                down_layer.append(Conv2D(num_channels=self.inplanes,
                                         num_filters=planes * block.expansion,
                                         filter_size=1, stride=1))
            else:
                down_layer.append(Conv2D(self.inplanes, planes * block.expansion,
                                         filter_size=1, stride=stride))
            down_layer.append(norm_layer(planes * block.expansion))
            print("down layer has ", down_layer)
            downsample = Sequential(*down_layer)

        layers = []
        if dilation == 1 or dilation == 2:
            print(1)
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = self.maxpool(x)

        print("maxpool shape", x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = fluid.layers.flatten(x, 1)
        if self.drop:
            x = fluid.layers.dropout(x, dropout_prob=self.final_drop)
        x = self.fc(x)

        return x
