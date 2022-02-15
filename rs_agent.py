import copy
import numpy as np


class RSAgent:
    """
        Q学習 エージェント
    """

    def __init__(
            self,
            aleph_g=1,
            gamma_g=.99,
            alpha_tau = 0.1,
            gamma_tau = .99,
            alpha=.2,
            gamma=.99,
            scaling={},
            actions=None,
            observation=None):
        self.aleph_g = aleph_g
        self.gamma_g = gamma_g
        self.alpha_tau = alpha_tau
        self.gamma_tau = gamma_tau
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.action = None
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.reward_history = []
        self.q_table = self.init_table(0.0)
        self.tau_table = self.init_table(0)
        self.tau_current_table = self.init_table(0)
        self.tau_post_table = self.init_table(0)
        self.rs_table = self.init_table(0.0)
        self.aleph_table = self.init_table(0.0)
        self.eg = 0.0
        self.ng = 0.0
        if not scaling:
            scaling[self.state] = 1
        self.scaling = scaling
        self.choice_act(self.state)

    def init_table(self, repeats):
        table = {}
        table[self.state] = self.init_action_table(repeats)
        return table

    def init_action_table(self, repeats):
        return np.repeat(repeats, len(self.actions))

    def act(self):
        return self.action

    def choice_act(self, state):
        max_rs = max(self.rs_table[state])
        action = np.random.choice([i for i, rs in enumerate(self.rs_table[state]) if rs == max_rs])
        # print(str(max_rs) + ": " + str(self.rs_table[state]))
        # print(state)

        self.previous_action = copy.deepcopy(self.action)
        self.action = action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)

        # デフォルトスケーリングパラメーター
        if next_state not in self.scaling:
            self.scaling[next_state] = 1

        if next_state not in self.q_table:
            self.q_table[next_state] = self.init_action_table(0.0)
            self.tau_table[next_state] = self.init_action_table(0)
            self.tau_current_table[next_state] = self.init_action_table(0)
            self.tau_post_table[next_state] = self.init_action_table(0)
            self.rs_table[next_state] = self.init_action_table(0.0)
            self.aleph_table[next_state] = self.init_action_table(0.0)


        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.choice_act(self.state)
            self.reward_history.append(reward)
            self.learn(reward)
        else:
            self.choice_act(self.ini_state)

    def learn(self, reward):

        next_max_q = max(self.q_table[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_table[self.previous_state][self.previous_action] += self.alpha * (reward + self.gamma * next_max_q - self.q_table[self.previous_state][self.previous_action])
        q = self.q_table[self.previous_state][self.previous_action]

        tau = self.tau_current_table[self.previous_state][self.previous_action] + self.tau_post_table[self.previous_state][self.previous_action]
        self.tau_table[self.previous_state][self.previous_action] = tau
        self.tau_current_table[self.previous_state][self.previous_action] += 1
        self.tau_post_table[self.previous_state][self.previous_action] += self.alpha_tau * (self.gamma_tau * self.tau_table[self.state][self.action] - self.tau_post_table[self.previous_state][self.previous_action])
        # self.tau_post_table[self.previous_state][self.previous_action] = (1 - self.alpha_tau) * self.tau_post_table[self.previous_state][self.previous_action] + self.alpha_tau * self.gamma_tau * self.tau_table[self.state][self.action]

        delta_g = min(self.eg - self.aleph_g, 0)
        max_q = max(self.q_table[self.previous_state])
        aleph = max_q - (self.scaling[self.previous_state] * delta_g)

        self.rs_table[self.previous_state][self.previous_action] = tau * (q - aleph)

    def learn_eg(self, e_tmp):
        self.eg = (e_tmp + self.gamma_g * self.ng * self.eg) / (1 + self.gamma_g * self.ng)
        self.ng = 1 + self.gamma_g * self.ng
