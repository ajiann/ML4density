import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from math import cos, pi

import os
import os.path as osp
import json

from tqdm import tqdm

from utils.losses import build_loss
from utils.configs import Parser
from utils.metrics import build_metric, build_roc_prc_metric

from collections import OrderedDict

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class raw_dataset(Dataset):
    def __init__(self, features, label):
        self.len = len(features)
        self.features = torch.from_numpy(features).float()
        self.label = torch.from_numpy(label).float()

    def __getitem__(self, index):
        return self.features[index], self.label[index]
        
    def __len__(self):
        return self.len
    
def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.BatchNorm2d(dim_out),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, in_dim=1, out_dim=32):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)  # 这里先用这个
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)  # 这里先用这个
        self.c3 = nn.Sequential(
                nn.Conv2d(64, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Tanh()
                )

    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.pool1(h1)
        h3 = self.c2(h2)
        h4 = self.pool2(h3)
        h5 = self.c3(h4)
        return h5, h4

class Decoder(nn.Module):
    def __init__(self, out_dim=1, in_dim=32):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 32)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(64+16, 4)
        self.conv3 =  nn.Sequential(
                nn.Conv2d(4, out_dim, 3, 1, 1),
                nn.Sigmoid()
                )

    def init_weights(self):
        generation_init_weights(self)
        
    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        output = self.conv3(d4)  # shortpath from 2->7
        return output

class RouteNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            self.encoder.init_weights()
            self.decoder.init_weights()
        else:
            raise TypeError("'pretrained' must be a str or None. ")


X = np.zeros((497,1,512,512),dtype=np.float32)
Y = np.zeros((497,1,256,256),dtype=np.float32)
for i in range(1,498):
    x_temp = np.load('./inputnpy/adaptec1_input_%d.npy'%i)
    y_temp = np.load('./outputnpy/adaptec1_output_%d.npy'%i)
    X[i-1][0] = x_temp
    Y[i-1][0] = y_temp
ymax = np.max(Y)
Y /= ymax

X_train, X_, y_train, y_ = train_test_split(X, Y, test_size = 0.3, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size = 0.5, random_state=1234)
train_set = raw_dataset(features=X_train, label=y_train)
val_set = raw_dataset(features=X_val, label=y_val)
test_set = raw_dataset(features=X_test, label=y_test)

train_iter = DataLoader(dataset = train_set, batch_size = 16, shuffle = True, drop_last = False)
val_iter = DataLoader(dataset = val_set, batch_size = 16, shuffle = False, drop_last = False)
test_iter = DataLoader(dataset = test_set, batch_size = 8, shuffle = False, drop_last = False)

def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    # arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True

    # print('===> Loading datasets')
    # Initialize dataset
    # dataset = build_dataset(arg_dict)

    model = RouteNet()
    model.init_weights('/home/ajian/placement/dataset/work_dir/NNfordensity/model_iters_20000.pth')
    model.eval()
    model = model.cuda()

    # Build metrics
    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    count =0
    with tqdm(total=len(test_iter)) as bar:
        for data in test_iter:
            # if arg_dict['cpu']:
            #     input, target = feature, label
            # else:
            #     input, target = feature.cuda(), label.cuda()

            feature, label = data
            input, target = feature.cuda(), label.cuda()
            prediction = model(input)
            for metric, metric_func in metrics.items():
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

            # if arg_dict['plot_roc']:
            #     save_path = osp.join(arg_dict['save_path'], 'test_result')
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     file_name = osp.splitext(osp.basename(label_path[0]))[0]
            #     save_path = osp.join(save_path, f'{file_name}.npy')
            #     output_final = prediction.float().detach().cpu().numpy()
            #     np.save(save_path, output_final)
            #     count +=1

            bar.update(1)
    print(len(test_iter))
    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(test_iter))) 

    # eval roc&prc
    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))


if __name__ == "__main__":
    test()