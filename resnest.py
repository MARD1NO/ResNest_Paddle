from resnet import ResNet, Bottleneck

def resnest50(num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, num_classes=num_classes)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model
