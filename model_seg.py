import torch
import torch.nn as nn
import torch.nn.functional as F


class Seg(nn.Module):
    def __init__(self, args, part_num=13):
        super(Seg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3, 512, kernel_size=1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, kernel_size=1)
        self.convs3 = nn.Conv1d(256, self.part_num, kernel_size=1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size, _, num_point = x.size()  # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)  # B, 1024, N
        x_max = F.adaptive_max_pool1d(x, 1)     # B,D???
        x_avg = F.adaptive_avg_pool1d(x, 1)     # B,D
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_point)  # B,D,N
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_point)  # B,D,N
        x_global_feature = torch.cat([x_max_feature, x_avg_feature], 1)  # B, 2*1024, N
        x = torch.cat((x, x_global_feature), 1)  # B, 1024 * 3, N
        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.trans_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def get_loss(logics, labels, class_weights):
    '''
    logics: B*one_hot*N
    labels: B*N
    '''
    logics = logics.permute(0, 2, 1)    # B,N,C
    n_class = logics.size(2)
    logics = logics.reshape(-1, n_class)   # B*N,C
    labels = labels.contiguous().view(-1)   # B*N,
    loss = F.cross_entropy(logics, labels, class_weights[0])
    return loss
