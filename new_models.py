base = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [1024, 1024, 1024]}


# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=4, dilation=4, bias=False)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

    return TUN_bone(*extra_layer(vgg(base['tun'], 3), vgg(base['tun_ex'], 512)))

    self.net_bone.base.load_state_dict(torch.load(self.config.vgg))
