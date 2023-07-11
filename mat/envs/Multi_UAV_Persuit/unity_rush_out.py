
import numpy as np

from unityagents import UnityEnvironment
from mat.envs.Multi_UAV_Persuit.python_env import python_env
def unityenv_2_pyenv(u3d_env_info, good_agents_brain_name, py_env):
    good_agents_obs = u3d_env_info[good_agents_brain_name].vector_observations
    enemy_obs = good_agents_obs[0][52:142]
    ouragent_obs = good_agents_obs[0][7:52]
    a = np.isinf(ouragent_obs)
    b = np.isinf(good_agents_obs)
    enemy_obs[np.isinf(enemy_obs)] = 0
    ouragent_obs[np.isinf(ouragent_obs)] = 0
    good_agents_number = py_env.num_agents - py_env.num_adversaries
    for i, agent in enumerate(py_env.agents):
        if not agent.adversary:
            agent.pos = np.float32(ouragent_obs[9 * i:9 * i + 3])
            agent.speed = np.float32(ouragent_obs[3 + 9 * i:3 + 9 * i + 3])
            # print("speed",agent.index,agent.speed)
            agent.nearest_obstacle_pos = np.float32(ouragent_obs[6 + 9 * i:9 + 9 * i])
            agent.nearest_obstacle_pos[np.isinf(agent.nearest_obstacle_pos)] = agent.safe_distance_limit
            agent.target_pos = np.float32(good_agents_obs[0][-3:])
            agent.survive = np.float(good_agents_obs[0][2 + i])
            # py_env.finde_other_obstacle(agent)

            # safe_distance = agent.pos - np.float32(good_agents_obs[i][4:6])
            # safe_distance[np.isinf(safe_distance)] = agent.safe_distance_limit
            # agent.safe_distance = safe_distance
            # agent.target_pos=np.float32(good_agents_obs[i][6:8])
        else:
            j = i - good_agents_number
            agent.pos = np.float32(enemy_obs[9 * j:9 * j + 3])
            agent.speed = np.float32(enemy_obs[3 + 9 * j:3 + 9 * j + 3])
            agent.nearest_obstacle_pos = np.float32(enemy_obs[6 + 9 * j:9 + 9 * j])
            agent.nearest_obstacle_pos[np.isinf(agent.nearest_obstacle_pos)] = agent.safe_distance_limit
            agent.target_pos = np.float32(ouragent_obs[(j // 2) * 9:(j // 2) * 9 + 3])

            agent.survive = True
            # py_env.finde_other_obstacle(agent)
    py_env.update_normal_pos()

class unity_wrapper_env():
    def __init__(self,number_parallel_env=8,num_agents=15,num_adversaries=10,train_mode=False):
        #self.u3d_env=UnityEnvironment(file_name=r"../../../mat/envs/unity_environment/rushOut.exe", worker_id=7)
        #self.py_env=python_env(num_agents=num_agents, num_adversaries=num_adversaries)
        self.u3d_env =[UnityEnvironment(file_name=r"../../../mat/envs/parrel_unity_env/"+str(i)+"/rushOut.exe", worker_id=i) for i in range(number_parallel_env)]
        self.py_env = [python_env(num_agents=num_agents, num_adversaries=num_adversaries) for i in range(number_parallel_env)]
        self.n_threading = number_parallel_env


        self.num_agents=self.py_env[0].num_agents
        self.number_good_agents=self.py_env[0].num_agents-self.py_env[0].num_adversaries
        self.train_mode=train_mode

        good_agents_brain_name = self.u3d_env[0].brain_names[0]
        u3d_env_info = [self.u3d_env[i].reset(train_mode=train_mode) for i in range(number_parallel_env)]
        for i in range(number_parallel_env):
            unityenv_2_pyenv(u3d_env_info[i], good_agents_brain_name, self.py_env[i])

        self.share_observation_space=[len(self.py_env[0].observation(self.py_env[0].agents[i])[0]) for i in range(self.py_env[0].num_agents-self.py_env[0].num_adversaries)]
        self.observation_space=[len(self.py_env[0].observation(self.py_env[0].agents[i])[0]) for i in range(self.py_env[0].num_agents-self.py_env[0].num_adversaries)]
        self.action_space=[3 for i in range(self.py_env[0].num_agents-self.py_env[0].num_adversaries)]

    # def step(self,actions):#return ：obs, share_obs, rewards, dones, infos, _
    #     all_die=False
    #
    #     for i,agent in enumerate(self.py_env.good_agents()):
    #         agent.action=actions[0][i]
    #         agent.u=agent.action
    #     for i,agent in enumerate(self.py_env.adversaries()):
    #         agent.action=2*np.random.random(3)-1
    #         agent.u=agent.action
    #
    #     good_agents_actions_normal = np.concatenate([self.py_env.agents[i].action for i in range(self.py_env.num_agents - self.py_env.num_adversaries)], axis=0)
    #     adv_agents_actions_normal = np.concatenate([self.py_env.agents[i].action for i in range(self.py_env.num_agents - self.py_env.num_adversaries, self.py_env.num_agents)],axis=0)
    #     self.py_env.set_action()
    #     good_agents_actions = np.concatenate([self.py_env.agents[i].u for i in range(self.py_env.num_agents - self.py_env.num_adversaries)], axis=0)
    #     good_agents_actions = np.reshape(good_agents_actions, (1, -1))
    #
    #     adv_agents_actions = np.concatenate(
    #         [self.py_env.agents[i].u for i in range(self.py_env.num_agents - self.py_env.num_adversaries, self.py_env.num_agents)], axis=0)
    #     adv_agents_actions = np.reshape(adv_agents_actions, (1, -1))
    #
    #     AI_agent_actions = np.hstack((good_agents_actions, adv_agents_actions))
    #
    #     good_agents_brain_name = self.u3d_env.brain_names[0]
    #     #print("good_actions",good_agents_actions,"good_actions_normal",good_agents_actions_normal,"adv_action",adv_agents_actions)
    #     u3d_env_info = self.u3d_env.step({good_agents_brain_name: AI_agent_actions})
    #
    #     unityenv_2_pyenv(u3d_env_info, good_agents_brain_name, self.py_env)
    #
    #     new_obs_n = []
    #
    #     reward_n = []
    #
    #     done_unity = u3d_env_info[good_agents_brain_name].vector_observations[0][-1]
    #     done = self.py_env.calculate_done()
    #     self.py_env.done = done
    #
    #     for i, agent in enumerate(self.py_env.agents):
    #
    #         new_obs = self.py_env.observation(agent, normalizaton=True)
    #         new_obs = np.reshape(new_obs, [1, -1])
    #         new_obs_n.append(new_obs)
    #         reward_n.append(self.py_env.reward(agent))
    #
    #     new_obs_n=np.reshape(new_obs_n[:self.number_good_agents],(-1,self.number_good_agents,self.observation_space[0]))
    #
    #     share_obs=new_obs_n
    #
    #     info=None
    #     if done:
    #         done_n=np.array([True for _ in range(self.number_good_agents)])
    #         new_obs_n,share_obs,info=self.reset()
    #     else:
    #         done_n=np.array([False for _ in range(self.number_good_agents)])
    #     done_n=np.reshape(done_n,(self.n_threading,self.number_good_agents))
    #     reward_n=np.reshape(reward_n,(self.n_threading,self.num_agents,1))
    #
    #     if done==-1:
    #         all_die=True
    #
    #     return new_obs_n,share_obs,reward_n[:,:self.number_good_agents,:],done_n,info,None

    def step(self,actions):#return ：obs, share_obs, rewards, dones, infos, _
        new_obs_n = []

        reward_n = []
        done_n_list=[]
        for env_id in range (self.n_threading):
            all_die=False

            for i,agent in enumerate(self.py_env[env_id].good_agents()):
                agent.action=actions[env_id][i]
                agent.u=agent.action
            for i,agent in enumerate(self.py_env[env_id].adversaries()):
                agent.action=2*np.random.random(3)-1
                agent.u=agent.action

            # good_agents_actions_normal = np.concatenate([self.py_env.agents[i].action for i in range(self.py_env.num_agents - self.py_env.num_adversaries)], axis=0)
            # adv_agents_actions_normal = np.concatenate([self.py_env.agents[i].action for i in range(self.py_env.num_agents - self.py_env.num_adversaries, self.py_env.num_agents)],axis=0)
            self.py_env[env_id].set_action()
            good_agents_actions = np.concatenate([self.py_env[env_id].agents[i].u for i in range(self.py_env[env_id].num_agents - self.py_env[env_id].num_adversaries)], axis=0)
            good_agents_actions = np.reshape(good_agents_actions, (1, -1))

            adv_agents_actions = np.concatenate(
                [self.py_env[env_id].agents[i].u for i in range(self.py_env[env_id].num_agents - self.py_env[env_id].num_adversaries, self.py_env[env_id].num_agents)], axis=0)
            adv_agents_actions = np.reshape(adv_agents_actions, (1, -1))

            AI_agent_actions = np.hstack((good_agents_actions, adv_agents_actions))

            good_agents_brain_name = self.u3d_env[env_id].brain_names[0]
            #print("good_actions",good_agents_actions,"good_actions_normal",good_agents_actions_normal,"adv_action",adv_agents_actions)
            u3d_env_info = self.u3d_env[env_id].step({good_agents_brain_name: AI_agent_actions})

            unityenv_2_pyenv(u3d_env_info, good_agents_brain_name, self.py_env[env_id])



            done_unity = u3d_env_info[good_agents_brain_name].vector_observations[0][-1]
            done = self.py_env[env_id].calculate_done()
            self.py_env[env_id].done = done

            for i, agent in enumerate(self.py_env[env_id].good_agents()):

                new_obs = self.py_env[env_id].observation(agent, normalizaton=True)
                new_obs = np.reshape(new_obs, [1, -1])
                new_obs_n.append(new_obs)
                reward_n.append(self.py_env[env_id].reward(agent))



            info=None
            if done:
                done_n=np.array([True for _ in range(self.number_good_agents)])
                new_obs_n_,share_obs_,info=self.reset(env_id=env_id)
                # for i in range(self.number_good_agents):
                #     new_obs_n[-self.number_good_agents+i]=new_obs_n_[0][i].reshape(1,-1)
                new_obs_n[-self.number_good_agents:] = np.reshape(new_obs_n_,(5,1,-1))
            else:
                done_n=np.array([False for _ in range(self.number_good_agents)])
            done_n_list.append(done_n)

        new_obs_n = np.reshape(np.array(new_obs_n),(-1, self.number_good_agents, self.observation_space[0]))

        share_obs = new_obs_n
        done_n_list = np.reshape(done_n_list, (self.n_threading, self.number_good_agents))
        reward_n = np.reshape(reward_n, (self.n_threading, self.number_good_agents, 1))

        return new_obs_n,share_obs,reward_n,done_n_list,info,None

    def reset(self,env_id=None):
        if env_id is None:
            obs_n = []
            for env_index in range(self.n_threading):
                good_agents_brain_name = self.u3d_env[env_index].brain_names[0]
                # Unity Environment to python Environment
                u3d_env_info = self.u3d_env[env_index].reset(train_mode=self.train_mode)
                unityenv_2_pyenv(u3d_env_info, good_agents_brain_name, self.py_env[env_index])
                # self.py_env.history_HP = [self.py_env.agents[i].HP for i in range(self.py_env.num_agents)]
                # self.py_env.cumulated_deLta_hp = np.array([0.0 for _ in range(self.py_env.num_agents)])

                for i, agent in enumerate(self.py_env[env_index].good_agents()):

                    obs = self.py_env[env_index].observation(agent, normalizaton=True)
                    obs = np.reshape(obs, [1, -1])
                    obs_n.append(obs)

                #obs_n=np.reshape(obs_n,(-1,self.num_agents,self.observation_space[0]))
            good_obs_n=np.reshape(np.array(obs_n),(-1,self.number_good_agents,self.observation_space[0]))

            share_obs=good_obs_n

            info=None

            return good_obs_n,share_obs,info
        else:
            obs_n = []
            env_index=env_id
            good_agents_brain_name = self.u3d_env[env_index].brain_names[0]
            # Unity Environment to python Environment
            u3d_env_info = self.u3d_env[env_index].reset(train_mode=self.train_mode)
            unityenv_2_pyenv(u3d_env_info, good_agents_brain_name, self.py_env[env_index])
            # self.py_env.history_HP = [self.py_env.agents[i].HP for i in range(self.py_env.num_agents)]
            # self.py_env.cumulated_deLta_hp = np.array([0.0 for _ in range(self.py_env.num_agents)])

            for i, agent in enumerate(self.py_env[env_index].agents):
                obs = self.py_env[env_index].observation(agent, normalizaton=True)
                obs = np.reshape(obs, [1, -1])
                obs_n.append(obs)

                # obs_n=np.reshape(obs_n,(-1,self.num_agents,self.observation_space[0]))
            good_obs_n = np.reshape(obs_n[:self.number_good_agents], (-1, self.number_good_agents, self.observation_space[0]))

            share_obs = good_obs_n

            info = None

            return good_obs_n, share_obs, info


    def close(self):
        for i in range(self.n_threading):
            self.u3d_env[i].close()

