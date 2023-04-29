import torch
import torch.nn as nn
import torchsummary

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


# model urls for pretrained models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, planes, stride=1, groups=1, dilation=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module): #输出channel和输入channel都是64
    """Basic Block for ResNet 18 and 34

    Note
    ----
    BasicBlock is the same as a bottleneck block with expansion=1
    input channel = output channel = 64

    Parameters
    ----------
    in_planes : int
        input channel
    planes : int
        output channel
    stride : int, optional
        stride of the first convolutional layer, by default 1
    downsample : nn.Module, optional
        downsample module, 
        if not None, the input will be downsampled to the same size as the output, by default None
    groups : int, optional
        number of groups for the 3x3 convolution, by default 1
    base_width : int, optional
        base width for the bottleneck block, by default 64
    dilation : int, optional
        dilation for the 3x3 convolution, by default 1
    norm_layer : nn.Module, optional
        normalization layer, by default None


    Raises
    ------
    ValueError
        groups!=1 or base_width!=64
    NotImplementedError
        dilation > 1 not supported in BasicBlock
    """    
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups!=1 or base_width!=64:
            raise ValueError('BasicBlock only supports groups=1 and basic_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward

        Note
        ----
        x.shape = (batch_size, in_planes, H, W)
        x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> downsample -> relu
        x.shape : (batch_size, in_planes, H, W) -> (batch_size, out_planes, H/stride, W/stride)
         -> (batch_size, out_planes, H/stride, W/stride) -> (batch_size, out_planes, H/stride, W/stride)

        Parameters
        ----------
        x : torch.Tensor[batch_size, in_planes, H, W]
            input tensor

        Returns
        -------
        out : torch.Tensor[batch_size, out_planes, H/stride, W/stride]
            output tensor
        """        
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        # residual connection
        out = x + identity
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    """Bottleneck Block for ResNet 50, 101, 152

    Note
    ----
    input channel = 256, output channel = 64, expansion = 4
    the output need to be expanded to 256 channels

    Parameters
    ----------
    in_planes : int
        input channel
    planes : int
        output channel
    stride : int, optional
        stride of the first convolutional layer, by default 1
    downsample : nn.Module, optional
        downsample module,
        if not None, the input will be downsampled to the same size as the output, by default None
    groups : int, optional
        number of groups for the 3x3 convolution, by default 1
    base_width : int, optional
        base width for the bottleneck block, by default 64
    dilation : int, optional
        dilation for the 3x3 convolution, by default 1
    norm_layer : nn.Module, optional
        normalization layer, by default None
    """    
    expansion = 4
    __constants__ = ['downsample']
    def __init__(self, in_planes, planes, stride=1, downsample=None,
                    groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward

        Note
        ----
        x.shape = (batch_size, in_planes, H, W)
        x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3 -> downsample -> relu
        x.shape : (batch_size, in_planes, H, W) -> (batch_size, out_planes, H/stride, W/stride) 
        -> (batch_size, out_planes, H/stride, W/stride) -> (batch_size, out_planes*expansion, H/stride, W/stride)

        Parameters
        ----------
        x : torch.Tensor[batch_size, in_planes, H, W]
            input tensor

        Returns
        -------
        out : torch.Tensor[batch_size, out_planes, H/stride, W/stride]
            output tensor
        """            
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = identity + x

        return out

class ResNet(nn.Module):
    """ResNet 18, 34, 50, 101, 152

    Parameters
    ----------
    block : nn.Module
        block module
    layers : list
        number of layers for each block
    num_classes : int, optional
        number of classes, by default 1000
    zero_init_residual : bool, optional
        if True, the residual will be initialized to zero,
        instead of kaiming normal, by default False
    groups : int, optional
        number of groups for the 3x3 convolution, by default 1
    width_per_group : int, optional
        number of channels for each group, by default 64
    replace_stride_with_dilation : list, optional
        if not None, the stride will be replaced with dilation, by default None
    norm_layer : nn.Module, optional
        normalization layer, by default None
    """    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # every element in the tuple indicates if we should use dilation replace stride
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 224x224 -> 112x112;  Cin=3 -> Cout=64, 7x7 kernel, stride=2, padding=3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 112x112 -> 56x56; 3x3 maxpoll, stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 56x56 -> 56x56; Cin=64 -> 64->Cout=64*block.expansion=256, stride=1
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 56x56 ->; Cin=256 -> 128 -> Cout=128*block.expansion=512, stride=2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        # 28x28 -> 14x14; Cin=512 -> 256 -> Cout=256*block.expansion=1024, stride=2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # 14x14 -> 7x7; Cin=1024 -> 512 -> Cout=512*block.expansion=2048, stride=2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # 7x7 -> 1x1; Cin=2048 -> Cout=512*block.expansion=2048
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 1x1 -> 1x1; Cin=2048 -> Cout=num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """make layer

        Parameters
        ----------
        block : nn.Module
            block module
        planes : int
            output channel of the first conv layer
        blocks : int
            number of blocks
        stride : int, optional
            stride, by default 1
        dilate : bool, optional
            if True, the stride will be replaced with dilation, by default False

        Returns
        -------
        nn.Sequential
            a sequential module
        """        
        norm_layer = self._norm_layer
        downsample = None
        prevision_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        layers = []
        # HxW -> H/stridexW/stride; inplanes -> planes -> planes*block.expansion, stride=2
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, prevision_dilation, norm_layer))
        
        # planes*block.expansion -> planes -> planes*block.expansion, stride=1
        # planes*block.expansion -> planes -> planes*block.expansion, stride=1
        # ...
        # input channel = planes*block.expansion, output channel = planes*block.expansion
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _foward_impl(self, x):
        """forward

        Notes
        -----
        x -> conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> flatten -> fc
        x.shape -> (batch_size, 3, 224, 224) -> (batch_size, 64, 112, 112) 
        -> (batch_size, 64, 112, 112) -> (batch_size, 64, 112, 112) 
        -> (batch_size, 64, 56, 56) -> (batch_size, 256, 56, 56) 
        -> (batch_size, 512, 28, 28) -> (batch_size, 1024, 14, 14) 
        -> (batch_size, 2048, 7, 7) -> (batch_size, 2048, 1, 1) 
        -> (batch_size, 2048) -> (batch_size, num_classes)

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        x : torch.Tensor
            output tensor
        """        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._foward_impl(x)

def _resnet(arch, block, layers, pretiraned, progress, **kwargs):
    """_resnet

    Parameters
    ----------
    arch : str
        type of resnet, e.g. resnet18, resnet34, resnet50, resnet101, resnet152
    block : nn.Module
        block module
    layers : list
        number of blocks in each layer
    pretiraned : bool
        if True, load the pretrained model
    progress : bool
        if True, display the progress bar

    Returns
    -------
    model : nn.Module
        resnet model
    """    
    model = ResNet(block, layers, **kwargs)
    if pretiraned:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        try:
            model.load_state_dict(state_dict)
        except:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            model.load_state_dict(state_dict, strict=False)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """resnet18

    Notes
    -----
    1. conv1: 3x3, stride=2, padding=1, output channel=64
    2. maxpool: 3x3, stride=2, padding=1
    3. layer1: 2x BasicBlock, output channel=64
    4. layer2: 2x BasicBlock, output channel=128
    5. layer3: 2x BasicBlock, output channel=256
    6. layer4: 2x BasicBlock, output channel=512
    7. avgpool: 7x7, stride=1
    8. fc: output channel=num_classes

    Parameters
    ----------
    pretrained : bool, optional
        if True, load the pretrained model, by default False

    progress : bool, optional
        if True, display the progress bar, by default True

    Returns
    -------
    model : nn.Module
        resnet18 model
    """    
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', BottleNeck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleNeck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', BottleNeck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

if __name__ == "__main__":
    model = resnet50(pretrained=True, num_classes=44*2)
    print(model)
    print(torchsummary.summary(model, (3, 224, 224)))
