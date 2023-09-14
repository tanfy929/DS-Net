
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from basic_module import BasicBlock, Bottleneck, BN_MOMENTUM


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class build_resnet(nn.Module):
    def __init__(self, block, layers):
        super(build_resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, 64, layers[0])
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)

    def _make_layers(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        stage_1_feature = x
        x = self.layer2(x)
        stage_2_feature = x
        x = self.layer3(x)
        stage_3_feature = x
        x = self.layer4(x)
        stage_4_feature = x

        return x

    def init_weights(self, resnet_model_name):
        url = model_urls[resnet_model_name]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)


def get_resnet_34(pretrain=True):
    model = build_resnet(BasicBlock, [3, 4, 6, 3])
    if pretrain:
        model.init_weights(resnet_model_name='resnet34')
    return model


_resnet_backbone = {
    'resnet34': get_resnet_34,
}


def get_resnet_backbone(model_name):
    support_resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    assert model_name in support_resnet_models, "We just support the following models: {}".format(support_resnet_models)

    model = _resnet_backbone[model_name]

    return model

if __name__ == '__main__':
    str1 = 'resnet18'
    model = get_resnet_backbone(str1)

    print(model(pretrain=True))