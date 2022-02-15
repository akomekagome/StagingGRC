import copy


class GridWorld:

    def __init__(self):

        self.filed_type = {
            "N": 0,  # 通常
            "G": 1,  # ゴール
            "W": 2,  # 壁
            "T": 3,  # トラップ
            "G2": 4,
            "G3": 5,
            "G4": 6,
            "G5": 7,
            "G6": 8
        }
        self.actions = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }
        self.map = [[1, 2, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 4],
                    [0, 2, 2, 2, 0, 2],
                    [0, 2, 6, 0, 0, 2],
                    [0, 2, 2, 2, 5, 2]]
        # self.map = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.start_pos = 0, 4   # エージェントのスタート地点(x, y)
        # self.start_pos = 0, 16   # エージェントのスタート地点(x, y)
        self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点

    def step(self, action):
        """
            行動の実行
            状態, 報酬、ゴールしたかを返却
        """
        to_x, to_y = copy.deepcopy(self.agent_pos)

        # 移動可能かどうかの確認。移動不可能であれば、ポジションはそのままにマイナス報酬
        if self._is_possible_action(to_x, to_y, action) == False:
            return self.agent_pos, -1, False

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        # print("\r"+str((to_x, to_y)), end="")
        is_goal = self._is_end_episode(to_x, to_y)  # エピソードの終了の確認
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = to_x, to_y
        return self.agent_pos, reward, is_goal

    def _is_end_episode(self, x, y):
        """
            x, yがエピソードの終了かの確認。
        """
        if self.map[y][x] == self.filed_type["G"]:      # ゴール
            return True
        elif self.map[y][x] == self.filed_type["T"]:    # トラップ
            return True
        elif self.map[y][x] == self.filed_type["G2"]:    # トラップ
            return True
        elif self.map[y][x] == self.filed_type["G3"]:    # トラップ
            return True
        elif self.map[y][x] == self.filed_type["G4"]:    # トラップ
            return True
        elif self.map[y][x] == self.filed_type["G5"]:    # トラップ
            return True
        elif self.map[y][x] == self.filed_type["G6"]:    # トラップ
            return True
        else:
            return False

    def _is_wall(self, x, y):
        """
            x, yが壁かどうかの確認
        """
        if self.map[y][x] == self.filed_type["W"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False
        elif self._is_wall(to_x, to_y):
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == self.filed_type["N"]:
            return 0
        elif self.map[y][x] == self.filed_type["G"]:
            return 100
        elif self.map[y][x] == self.filed_type["G2"]:
            return 200
        elif self.map[y][x] == self.filed_type["G3"]:
            return 300
        elif self.map[y][x] == self.filed_type["G4"]:
            return 400
        elif self.map[y][x] == self.filed_type["G5"]:
            return 500
        elif self.map[y][x] == self.filed_type["G6"]:
            return 600
        elif self.map[y][x] == self.filed_type["T"]:
            return -100

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos
