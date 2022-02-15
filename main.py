import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent
from rs_agent import RSAgent
from grid_world import GridWorld
import time

# 定数

NB_EPISODE = 1000    # エピソード数
NB_SIMULATION = 1000
EPSILON = 0.1    # 探索率
ALPHA = .1      # 学習率
GAMMA = .99     # 割引率
ACTIONS = np.arange(4)  # 行動の集合
BUFFER = 5

if __name__ == '__main__':
    grid_env = GridWorld()  # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    # agent = QLearningAgent(
    #     alpha=ALPHA,
    #     gamma=GAMMA,
    #     epsilon=EPSILON,  # 探索率
    #     actions=ACTIONS,   # 行動の集合
    #     observation=ini_state)  # Q学習エージェント
    aleph_g = 395
    scaling = {}
    # scaling[str(ini_state)] = 1
    simulations = []
    is_end_episode = False  # エージェントがゴールしてるかどうか？

    while(len(simulations) < NB_SIMULATION):
        print("\r"+str(len(simulations) + 1), end="")
        rewards = []
        aleph_g = 95
        agent = RSAgent(
            aleph_g=aleph_g,
            gamma_g=.97,
            alpha_tau = .05,
            gamma_tau = .97,
            alpha=ALPHA,
            gamma=GAMMA,
            scaling=scaling,
            actions=ACTIONS,   # 行動の集合
            observation=ini_state)  # Q学習エージェント        # 実験
        is_rs = type(agent) is RSAgent
        for episode in range(NB_EPISODE):
            episode_reward = []  # 1エピソードの累積報酬
            start = time.time()
            while(is_end_episode == False):    # ゴールするまで続ける
                action = agent.act()  # 行動選択
                state, reward, is_end_episode = grid_env.step(action)
                agent.observe(state, reward)   # 状態と報酬の観測
                episode_reward.append(reward)
                if time.time() - start > 30:
                    break
            if time.time() - start > 30:
                break
            reward_average = np.sum(episode_reward)
            if is_rs:
                agent.learn_eg(reward_average)
                # 段階別
                if aleph_g < 300 and len(rewards) >= BUFFER and all([r >= aleph_g for r in rewards[-BUFFER:]]):
                    aleph_g += 100
                    agent.aleph_g = aleph_g
                    print(str(episode) + "change: " + str(aleph_g))
            rewards.append(reward_average)  # このエピソードの平均報酬を与える
            state = grid_env.reset()  # 初期化
            agent.observe(state)    # エージェントを初期位置に
            is_end_episode = False
        if time.time() - start > 30:
            print("continue")
            continue
        simulations.append(rewards)

    show_data = np.average(simulations, axis=0)

    # np.save('grid_smal_uzumaki_normal_2.npy', show_data)
    # np.save('grid_smal_uzumaki_dankai_2.npy', show_data)
    # np.save('grid_large_normal_2.npy', show_data)
    # np.save('grid_large_dankai_1.npy', show_data)

    # 結果のプロット
    plt.plot(np.arange(NB_EPISODE), show_data)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    plt.show()
