import torch
import torch.nn as nn

################
## Functions ###
################

# custom weight initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def define_netD():
    netD = _netD()
    netD.apply(weights_init)
    return netD

def define_netG(in_ch=2):
    netG = _netG(in_ch)
    netG.apply(weights_init)
    return netG

################
## Classes ###
################

class _netG(nn.Module):
    def __init__(self, in_ch):
        super(_netG, self).__init__()
        self.in_ch = in_ch
        
        # Convolutional 1
        self.conv1 = nn.Sequential(
            # input shape [batch_size x 2 (noise + input mel-cepstrum) x 40 (mgc dim) x T]
            nn.Conv2d(in_ch, 128, 5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        # Convolutional 2
        # input shape [batch_size x 128 + input mel-cepstrum x 40 x T]
        self.conv2 = nn.Sequential(
            nn.Conv2d(129, 256, 5, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

        # Convolutioanl 3
        # input shape [batch_size x 256 + input mel-cepstrum x 40 x T]
        self.conv3 = nn.Sequential(
            nn.Conv2d(257, 128, 5, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        # Convolutional 4
        # input shape [batch_size x 128 + input mel-cepstrum x 40 x T]
        self.conv4 = nn.Sequential(
            nn.Conv2d(129, 1, 5, padding=2, bias=True),
            #nn.Tanh()
        )
        # final output shape [batch_size x 1 x 40 x T]

    def forward(self, noise_input, cond_input):
        x = torch.cat((noise_input, cond_input), 1)

        x = self.conv1(x)
        x = torch.cat((x, cond_input), 1)

        x = self.conv2(x)
        x = torch.cat((x, cond_input), 1)

        x = self.conv3(x)
        x = torch.cat((x, cond_input), 1)

        x = self.conv4(x)
        return x


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # Convolutional block
        self.conv1 = nn.Sequential(
            # input shape batch_size x 1 (number of channels) x 40 (mgc dim) x 40 (time)
            nn.Conv2d(1, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # shape [batch_size x 64 x 18 x 18]
            nn.Conv2d(64, 128, 5, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # shape [batch_size x 128 x 7 x 7]
            nn.Conv2d(128, 256, 5, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # shape [batch_size x 256 x 3 x 3]
            nn.Conv2d(256, 128, 3, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # after flatten [batch_size x 128 * 1 * 1]
        # Dense block
        self.fc1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # final output shape [batch_size x 1]

    def forward(self, mgc_input):
        x = self.conv1(mgc_input)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x