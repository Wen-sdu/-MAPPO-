import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space





class R_Actor(nn.Module):
    """在深度学习框架（PyTorch）的语境下，这种继承不仅仅是语法上的，还有强制性功能要求
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    #R_Actor策略网络的初始化函数
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size#隐藏层神经元数量

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)#获取观测空间的形状
        self.base = MLPBase(args, obs_shape)#当前RMAPPO只使用向量观测的MLP特征提取网络

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:#根据配置创建RNN层，观测输入-CNN/MLP提取特征-RNN带有历史记忆的输出
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)#创建动作输出层

        self.to(device)#所有的参数都移动到指定设备上
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.根据输入计算出输出
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.输入到网络的观测值
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.RNN的隐藏状态
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent能够产生的所有的actions
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.动作的产生是“随机采样动作”or“实行概率最大的动作”
          训练阶段，deterministic=false，表示，使用随机采样动作，保证策略具有探索能力。部署时，该参数为True，ui=action_i。
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:


            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:#使用rnn时的时候
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)#根据要求选择不同的“动作选择方案（随机型or最大概率）”，使用对数概率的原因是，便于新旧策略对比。

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):#评价动作的执行情况
        """
        前面的forward（），是根据当下的观测生成action；目前的evaluate_actions作用是，
        Compute log probability and entropy of given actions.计算所给动作的对数概率和熵。
        :param obs: (torch.Tensor) observation inputs into network.当前的观测到的内容。
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.已经执行过的动作
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.智能体是否有效

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.动作对数概率
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.动作分布熵
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:#这是mappo所特有的，判断智能体是否还存在。
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:#普通的PPO/MAPPO分支
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):#继承torch.nn.Module，因此也是一个PyTorch网络
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).集中输入、本地obs
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.是集中式的观测空间
    :param device: (torch.device) specifies the device to run on (cpu/gpu).设备device
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart#popart标准化方法
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]#等价于if else 的用法

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:#根据是否使用popart的条件，选择使用popart或者linear方法。
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))



        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):#根据给定的输入计算动作价值
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.价值函数预测（值函数）
        :return rnn_states: (torch.Tensor) updated RNN hidden states.更新 RNN 隐藏状态
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states#values用于计算优势函数以及PPO value loss 价值估计和新的隐藏状态
