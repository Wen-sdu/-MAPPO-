import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):


    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]#也是等同于 if else 结构，使用 ReLU函数 or Tanh函数（激活函数）
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]#Xavier初始化方法 or 正交初始化方法（初始化方法）
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])#根据激活函数，选择不同的gain

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(#第一层
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList([nn.Sequential(init_(#后续的隐藏层
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x):#前向传播
        x = self.fc1(x)
        for i in range(self._layer_N):

            x = self.fc2[i](x)
        return x

#用法均在Actor和Critic的py文件中，根据obs的维度，选择CNN或者MLP
class MLPBase(nn.Module):#外部调用接口，两个功能：输入归一化，调用MLP。
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]#输入维数

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)#归一化

        self.mlp = MLPLayer(obs_dim, self.hidden_size,#前面的MLPLayer的使用
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x