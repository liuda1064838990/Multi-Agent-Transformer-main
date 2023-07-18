# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:35:31 2020

@author: freedom
"""

import numpy as np
#from python_enviroment.core import Agent
#import dgl
import torch
# from dgl.nn import GATConv


# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:15:21 2020

智能体属性

@author: freedom
"""

#import numpy as np

class Agent(object):
    def __init__(self):
        self.name = ''
        self.size = None
        self.pos = None
        self.speed = None
        self.max_speed = 60
        self.max_acc=60
        self.accel_range = 10
        self.index=1
        self.u = None
        self.action=None
        self.safe_distance_limit = 5
        self.sefe_distance_warning = 2
        self.safe_distance = []
        self.adversary = False
        self.done = True
        self.action = None
        self.survive=1
        self.nearest_obstacle_pos=[15,15,15]
        self.target_pos=None
        self.normal_position = None
        self.target_normal_position = None
        self.nearest_obstacle_normal_postion = None


class python_env(object):
    def __init__(self, num_agents, num_adversaries):
        """

        :rtype:
        """
        self.world_dim = 3
        self.world_with = np.array([25, 10, 25])
        self.num_agents = num_agents
        self.num_adversaries = num_adversaries
        self.num_goods = self.num_agents - self.num_adversaries

        self.agents = [Agent() for i in range(self.num_agents)]
        self.done = 0
        # self.history_HP = [self.agents[i].HP for i in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.index = i
            agent.name = 'agent %d' % i
            agent.size = np.sqrt(np.sum(np.square([1.2, 0.8])))
            agent.adversary = True if i > self.num_agents - self.num_adversaries - 1 else False
            agent.accel_range = 4 if agent.adversary == True else 6
            agent.max_speed = 60 if agent.adversary == True else 60
            agent.max_acc = 60 if agent.adversary == True else 120
            agent.fire_range = 60 if agent.adversary == True else 65
            agent.safe_distance_limit = 10
            agent.sefe_distance_warning = 10
            agent.safe_distance = agent.safe_distance_limit * np.ones(self.world_dim)
            agent.normal_position = np.float32([1, 1, 1])
            agent.target_normal_position = np.float32([1, 1, 1])
            agent.nearest_obstacle_normal_postion = np.float32([1, 1, 1])

    def good_agents(self):
        return [agent for agent in self.agents if not agent.adversary]

    def adversaries(self):
        return [agent for agent in self.agents if agent.adversary]

    def reward(self, agent):

        main_reward = self.adversary_reward(agent) if agent.adversary else self.agent_reward(agent)
        return main_reward

    def calculate_target_distance(self, agent):

        tar_relative_pos = np.array(agent.pos - agent.target_pos)
        dis_tar = np.sqrt(np.sum(np.square(tar_relative_pos)))

        if dis_tar > 40:
            dis_tar_rew = -2
        elif 10 < dis_tar <= 40:
            dis_tar_rew = -1
        elif 0.2 < dis_tar <= 10:
            dis_tar_rew = -np.tanh(7.5 * (dis_tar) - 3)
        elif dis_tar < 0.2:
            dis_tar_rew = 1
        # print("dis",dis_tar_rew)
        return dis_tar_rew

    def calculate_catcher_distance(self, agent):
        catcher_pos = np.array([abs(agent.pos - catcher.pos) for catcher in self.adversaries()])
        dis = np.sqrt(np.sum(np.square(catcher_pos), axis=-1))
        catcher_dis = min(dis)
        k = 1e-3

        penatration_distance = np.logaddexp(0, (agent.sefe_distance_warning - catcher_dis) / k) * k

        #
        catcher_dis_rew = 0.02 * penatration_distance
        # print(catcher_dis_rew)
        return catcher_dis_rew

    def calculate_done(self):
        done_index = []
        survival = np.array([agent.survive for agent in self.good_agents()])
        for index, value in enumerate(survival):
            if value == 2:
                done_index.append(index)
        if len(done_index) >= 3:
            # print("done",survival)
            return 1
        else:
            return 0

    def agent_reward(self, agent):
        if agent.adversary:
            return self.adversary_reward(agent)
        else:
            rew = 0

            dis_infor = []
            #123

            # gain_hp = delta_hp[5:]
            # print('delta_hp',delta_hp)

            # rew = -np.sum(loss_hp) * 0
            good_agents = self.good_agents()
            tar_relative_pos = np.array(agent.pos - agent.target_pos)
            dis_tar = np.sqrt(np.sum(np.square(tar_relative_pos)))

            obstacle_relative_pos = np.sqrt(np.sum(np.square(np.array(agent.pos - agent.nearest_obstacle_pos))))

            k = 1e-3
            k_1 = 0.04

            penatration_distance = np.logaddexp(0, (agent.sefe_distance_warning - obstacle_relative_pos) / k) * k
            if dis_tar > 3:
                rew -= 0.02 * penatration_distance
                rew -= self.calculate_catcher_distance(agent)

            rew -= k_1 * dis_tar
            winflag = 0
            # for i, good_agent in enumerate(good_agents):
            #     if good_agent.survive==2:
            #         winflag+=1
            # print("survive_state", [good_agent.survive for good_agent in good_agents])

            if self.done :
                rew +=10
                print("rew_done!", rew)
                print("survive_state", [good_agent.survive for good_agent in good_agents])
            # self.calculate_catcher_distance(agent)
            for other in good_agents:
                if other is agent: continue
                if other.survive == 0:
                    rew -= 0.1

            # rew+=self.calculate_target_distance(agent)

            return rew

    def adversary_reward(self, agent):

        rew = 0

        dis_infor = []

        # gain_hp = delta_hp[5:]
        # print('delta_hp',delta_hp)

        # rew = -np.sum(loss_hp) * 0
        good_agents = self.good_agents()

        tar_relative_pos = np.array(agent.pos - agent.target_pos)
        dis_tar = np.sqrt(np.sum(np.square(tar_relative_pos)))

        obstacle_relative_pos = np.sqrt(np.sum(np.square(np.array(agent.pos - agent.nearest_obstacle_pos))))

        k = 1e-3
        k_1 = 0.02
        k_2 = 0.02
        penatration_distance = np.logaddexp(0, (agent.sefe_distance_warning - obstacle_relative_pos) / k) * k
        rew -= 0.02 * penatration_distance
        if dis_tar < 5:

            rew -= k_2 * dis_tar
        else:
            rew -= k_1 * dis_tar

        # print("agent_id",agent.index,"obstacle_reward:",0.1 * penatration_distance,"obstacle_relative_pos",obstacle_relative_pos,"tar_reward:",k_1*dis_tar)
        # if good_agents[(agent.index-5)//2].survive==0:
        #     rew+=50
        # if good_agents[(agent.index-5)//2].survive==2:
        #     rew-=10
        # print("bad_agents:", "distance_reward", -k_1 * dis_tar, "obs_reward", -0.1 * penatration_distance,)

        return rew

    # def finde_other_obstacle(self,agent):
    #     other_obstacle=copy.deepcopy(agent.pos)
    #     delta_abs_pos=self.world_with-abs(agent.pos)
    #     flag=0
    #     boundary_value=1
    #     # if delta_abs_pos[0]<1 and agent.pos[0]<0:
    #     #     other_obstacle[0]=-self.world_with[0]
    #     # if delta_abs_pos[0] < 1 and agent.pos[0] >0:
    #     #     other_obstacle[0] = -self.world_with[0]
    #     #
    #     # if delta_abs_pos[2]<1 and agent.pos[2]<0:
    #     #     other_obstacle[2]=-self.world_with[2]
    #     # if delta_abs_pos[2] < 1 and agent.pos[2] >0:
    #     #     other_obstacle[2] = -self.world_with[2]
    #     #
    #     # if delta_abs_pos[1]<-11 :
    #     #     other_obstacle[1]
    #     for i in range(len(self.world_with)):
    #         if i==1:
    #             if delta_abs_pos [i]<-1.5:#y10-[2,12]=[-2,8]
    #                 other_obstacle[i]=12
    #                 flag=1
    #
    #             elif delta_abs_pos[i]>7.5:
    #                 other_obstacle[i]=2
    #                 flag = 1
    #
    #         else:
    #             if delta_abs_pos[i]<boundary_value and agent.pos[i]<0:
    #                 other_obstacle[i]=-self.world_with[i]
    #                 flag = 1
    #             elif delta_abs_pos[i]<boundary_value and agent.pos[i]>0:
    #                 other_obstacle[i] = self.world_with[i]
    #                 flag = 1
    #
    #     if flag==1:
    #         #print("flag",flag)
    #         agent.nearest_obstacle_pos=other_obstacle
    #         #print("1")

    def update_normal_pos(self):

        for i, agent in enumerate(self.agents):  # y[8,18]
            for k in range(len(agent.pos)):
                if k == 1:
                    agent.normal_position[k] = (agent.pos[k] - 13) / 5
                    agent.nearest_obstacle_normal_postion[k] = (agent.nearest_obstacle_pos[k] - 13) / 5
                    agent.target_normal_position[k] = (agent.target_pos[k] - 13) / 5
                else:
                    agent.normal_position[k] = agent.pos[k] / self.world_with[k]
                    agent.nearest_obstacle_normal_postion[k] = agent.nearest_obstacle_pos[k] / self.world_with[k]
                    agent.target_normal_position[k] = agent.target_pos[k] / self.world_with[k]

    def observation(self, agent, normalizaton=True):
        other_pos = []
        other_vel = []
        other_tar = []
        other_survive = []
        if agent.survive == 0:
            return [[0, 0, 0] for i in range(19)]
        if normalizaton:
            for other in self.agents:
                if other is agent: continue
                if other.survive == 0:
                    other_pos.append(np.array([0, 0, 0]))
                else:
                    other_pos.append(other.normal_position - agent.normal_position)
                if not agent.adversary:  # if this obs for good_obs    velocity
                    if other.survive == 0:
                        other_vel.append(np.array([0, 0, 0]))
                    else:
                        other_vel.append(other.speed / other.max_speed)
                    if not other.adversary:
                        other_survive.append(other.survive)
                        other_tar.append(other.normal_position - agent.target_normal_position)
                elif agent.adversary:  # if this obs for adv_obs      velocity

                    if other.survive == 0:
                        other_vel.append(np.array([0, 0, 0]))
                    else:
                        other_vel.append(other.speed / other.max_speed)
                    if other.adversary:
                        other_survive.append(other.survive)
                        other_tar.append(other.normal_position - other.target_normal_position)  # 其他围捕无人机到突围无人机的距离
            other_survive.append(agent.survive)
            other_survive = np.array([other_survive]) / 2
            other_pos = np.reshape(other_pos, (1, -1))
            other_vel = np.reshape(other_vel, (1, -1))
            other_tar = np.reshape(other_tar, (1, -1))
            agent_pos = agent.normal_position
            agent_speed = agent.speed / agent.max_speed
            target_pos = np.array(agent.target_normal_position - agent.normal_position)
            obstacle_pos = np.array(agent.nearest_obstacle_normal_postion - agent.normal_position)
            # a=np.array(agent.nearest_obstacle_pos/self.world_with-agent.pos/self.world_with)

        else:
            for other in self.agents:
                if other is agent: continue
                other_pos.append(other.pos - agent.pos)
                if not agent.adversary:
                    if not other.adversary:
                        other_vel.append(other.speed)
            other_pos = np.reshape(other_pos, (1, -1))
            other_vel = np.reshape(other_vel, (1, -1))
            other_tar = np.reshape(other_tar, (1, -1))
            agent_pos = agent.pos
            agent_speed = agent.speed
            agent_safe_distance = agent.safe_distance
            target_pos = np.array(agent.target_pos - agent.pos)
            obstacle_pos = np.array(agent.nearest_obstacle_pos)
        # a=np.concatenate(([agent_pos], [agent_speed], other_pos, other_tar,other_vel,[target_pos],[obstacle_pos]),axis=-1)
        # return np.concatenate(([agent_pos], [agent_speed], other_pos, other_vel,[target_pos],[obstacle_pos]))
        return np.concatenate(
            ([agent_pos], [agent_speed], other_pos, other_tar, other_vel, [target_pos], [obstacle_pos], other_survive),
            axis=-1)

    def observation_LSTM_series(self, agent, normalizaton=True):
        good_agents = self.good_agents()
        adv_agents = self.adversaries()
        adv_pos = []
        state_n = []
        if normalizaton:
            agent_pos = agent.pos / self.world_with
            agent_speed = agent.speed / agent.max_speed
            agent_safe_distance = agent.safe_distance / agent.safe_distance_limit
            for adv in adv_agents:
                adv_pos.append(adv.pos / self.world_with - agent.pos / self.world_with)
            state_n.append(np.concatenate([agent_pos] + [agent_speed] + [agent_safe_distance] + adv_pos))
            for other in good_agents:
                adv_pos = []
                if other is agent: continue
                other_pos = other.pos / self.world_with - agent.pos / self.world_with
                other_vel = other.speed / other.max_speed - agent.speed / agent.max_speed
                other_safe_distance = other.safe_distance / other.safe_distance_limit
                for adv in adv_agents:
                    adv_pos.append(adv.pos / self.world_with - other.pos / self.world_with)
                state_n.append(np.concatenate([other_pos] + [other_vel] + [other_safe_distance] + adv_pos))
        else:
            agent_pos = agent.pos
            agent_speed = agent.speed
            agent_safe_distance = agent.safe_distance
            for adv in adv_agents:
                adv_pos.append(adv.pos - agent.pos)
            state_n.append(np.concatenate([agent_pos] + [agent_speed] + [agent_safe_distance] + adv_pos))
            for other in good_agents:
                adv_pos = []
                if other is agent: continue
                other_pos = other.pos - agent.pos
                other_vel = other.speed - agent.speed
                other_safe_distance = other.safe_distance
                for adv in adv_agents:
                    adv_pos.append(adv.pos - other.pos)
                state_n.append(np.concatenate([other_pos] + [other_vel] + [other_safe_distance] + adv_pos))
        return np.stack(state_n)

    def set_action(self):
        for i, agent in enumerate(self.agents):
            if not agent.adversary:
                agent.u = agent.u * agent.max_acc
                # if np.sqrt(np.sum(np.square(agent.u))) >= agent.max_speed:
                #     agent.u = agent.max_speed * np.ones_like(agent.u)
            else:
                agent.u = agent.u * agent.max_acc
                # if np.sqrt(np.sum(np.square(agent.u))) >= agent.max_speed:
                #     agent.u = agent.max_speed * np.ones_like(agent.u)

        return [self.agents[i].u for i in range(self.num_agents)]

    def clip_action(self):
        for i, agent in enumerate(self.agents):
            if np.sqrt(np.sum(np.square(agent.u))) > agent.accel_range:
                agent.u = agent.u * (agent.accel_range / (np.sqrt(np.sum(np.square(agent.u)))))
            next_speed = agent.speed + agent.u
            if np.sqrt(np.sum(np.square(next_speed))) > agent.max_speed:
                next_speed = next_speed * (agent.max_speed / np.sqrt(np.sum(np.square(next_speed))))
                agent.u = next_speed - agent.speed

    def Attack_target(self, agent, obs, die_id):
        attack_id = -1
        dis_infor = []
        if agent.adversary:
            if agent.fire == False or agent.HP == 0:
                agent.attack_id = attack_id
                return attack_id
            else:
                tar_pos_information = obs[0][12:22]

                tar_pos_information = np.reshape(tar_pos_information, (-1, 2))
                for i in range(self.num_agents - self.num_adversaries):
                    dis_infor.append(np.sqrt(np.sum(np.square(tar_pos_information[i]))))
                for i in range(len(die_id)):
                    if die_id[i] < 5:
                        dis_infor[die_id[i]] = np.inf
                dis_min = min(dis_infor)
                if dis_min * self.world_with > agent.fire_range:  # not in attack range
                    attack_id = -1
                    # print('out of range')

                else:
                    attack_id = np.argmin(dis_infor)
                # print('dieid', die_id, 'tar_pos,', dis_infor,'attack_id',attack_id)
                # print('dis_min', dis_min * self.world_with,'dis_infor',dis_infor,'tarpos',tar_pos_information)
        else:
            if agent.fire == False or agent.HP == 0:
                agent.attack_id = attack_id
                return attack_id
            else:
                tar_pos_information = obs[0][12:22]
                tar_pos_information = np.reshape(tar_pos_information, (-1, 2))
                for i in range(self.num_adversaries):
                    dis_infor.append(np.sqrt(np.sum(np.square(tar_pos_information[i]))))
                for i in range(len(die_id)):
                    if die_id[i] > 4:
                        dis_infor[die_id[i] - 5] = np.inf
                dis_min = min(dis_infor)
                if dis_min * self.world_with > agent.fire_range:  # not in attack range
                    attack_id = -1

                else:
                    attack_id = np.argmin(dis_infor)
                # print('dis_in', dis_min * self.world_with,'dis_infor',dis_infor)
                # print('dieid', die_id, 'tar_pos,', dis_infor, 'attack_id', attack_id)
        agent.attack_id = attack_id
        return attack_id

    def build_graph(self, ):
        # g=dgl.DGLGraph()
        num_ag = self.num_agents - self.num_adversaries
        # g.add_nodes(num_ag)
        # observation=env.observation_callback[2:4]
        position_in_vector = [self.agents[i].pos for i in range(0, self.num_agents - self.num_adversaries)]

        adj_matrix = self.get_connectivity(position_in_vector)
        edge_list = []
        for i in range(0, num_ag):
            for j in range(0, num_ag):
                if adj_matrix[i][j] > 0:
                    edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        src = list(src)
        dst = list(dst)
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)))
        # g.add_edges(src,dst)
        # g.add_edges(dst,src)
        g = dgl.to_bidirected(g)

        g.set_e_initializer(dgl.init.zero_initializer)
        g.set_n_initializer(dgl.init.zero_initializer)
        # g.ndata['feat'] = torch.eye(num_ag)
        g = g.to('cuda')

        return g

    def get_connectivity(self, position_in_vectior, degree=2):

        neigh = NearestNeighbors(n_neighbors=degree)
        neigh.fit(position_in_vectior)
        a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

        return a_net