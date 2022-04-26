import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import datetime
import torch
import math


class Decentralized(object):

    def __init__(self, horizon, trial, num_player, num_arm, player_ranking, arm_ranking, player_mean):
        self.path = './ResultsData/decen/'

        self.horizon = horizon
        self.trials = trial

        self.p_lambda = 0.1
        self.epsilon = 10**(-10)

        # phased ETC algorithm
        self.varEpsilon = 0.2

        self.num_players = num_player
        self.num_arms = num_arm
        self.players_ranking = player_ranking
        self.arms_rankings = arm_ranking  
        self.players_mean = player_mean

        # UCB-D4
        self.beta = 1/(2*self.num_arms)
        self.gamma = 2
        
        self.pessimal_matching = self.get_pessimal_matching(self.players_ranking,self.arms_rankings).tolist()
           
        print("player_ranking", self.players_ranking)
        print("arm_ranking", self.arms_rankings)
        print("players_mean", self.players_mean)
        print("pessimal matching",self.pessimal_matching)

        At=np.zeros(self.num_players)
        for a,p in enumerate(self.pessimal_matching):
                At[p]=a
        print("Pessimal Matching Player:",At)

    def get_pessimal_matching(self,players_rankings,arms_rankings):
        # propose_order records the order arms should follow while proposing
        init_propose_order = np.zeros(self.num_arms, int)
        propose_order = init_propose_order
        # matched record whether a specific player is matched or not
        matched = np.zeros(self.num_arms, bool)
        # matching records the choice of a player for a specific arm
        matching = [[] for _ in range(self.num_players)]

        # Terminates if all matched
        while np.sum(matched) != self.num_arms:

            # arms propose at the same time
            for a_idx in range(self.num_arms):
                if not matched[a_idx]:
                    # p_proposal is the index of an arm
                    # propose_order is the vector, p_o[i] is the order of player i's next proposal
                    a_proposal = arms_rankings[a_idx][propose_order[a_idx]]
                    matching[a_proposal].append(a_idx)

            # arms choose its player
            for p_idx in range(self.num_players):
                p_choices = matching[p_idx]

                if len(p_choices) != 0:    
                    # each arm chooses the its most preferable one
                    p_choice = next((x for x in players_rankings[p_idx] if x in matching[p_idx]), None)
                    # update arm's choice where there should only be one left
                    matching[p_idx] = [p_choice]
                    # update player's state of matched
                    for a_idx in p_choices:
                        matched[a_idx] = (a_idx == p_choice)
                        propose_order[a_idx] += (1 - (a_idx == p_choice))
        return np.squeeze(matching)

    def isUnstable(self, arm_matching):
        # arm_matching: [0,1,-1]
        # arm 0 matches player 0; arm 1 matches player 1; arm 2 matches nothing

        # if unstable return 1, otherwise return 0
        arm_matching = arm_matching.tolist()

        if -1 in arm_matching:
            return 1

        player_matching = np.ones(self.num_players)*(-1)
        for p_idx in range(self.num_players):
            if p_idx in arm_matching:
                player_matching[p_idx] = arm_matching.index(p_idx)

        if -1 in player_matching:
            return 1
        
        # find blocking pair
        for p_idx in range(self.num_players):
            for possible_arm_rank in range(self.players_ranking[p_idx].index(player_matching[p_idx])):
                arm = self.players_ranking[p_idx][possible_arm_rank]
                for possible_player_rank in range(self.arms_rankings[arm].index(arm_matching[arm])):
                    if self.arms_rankings[arm][possible_player_rank] == p_idx:
                        return 1
        return 0
    
    def Gale_Shapley(self, player_ranking):
            # propose_order records the order players should follow while proposing
        init_propose_order = np.zeros(self.num_players, int)
        propose_order = init_propose_order
        # matched record whether a specific player is matched or not
        matched = np.zeros(self.num_players, bool)
        # matching records the choice of a player for a specific arm
        matching = [[] for _ in range(self.num_arms)]

        # Terminates if all matched
        while np.sum(matched) != self.num_players:

            # players propose at the same time
            for p_idx in range(self.num_players):
                if not matched[p_idx]:
                    # p_proposal is the index of an arm
                    # propose_order is the vector, p_o[i] is the order of player i's next proposal
                    p_proposal = player_ranking[p_idx][propose_order[p_idx]]
                    matching[p_proposal].append(p_idx)

            # arms choose its player
            for a_idx in range(self.num_arms):
                a_choices = matching[a_idx]

                if len(a_choices) != 0:    
                    # each arm chooses the its most preferable one
                    a_choice = next((x for x in self.arms_rankings[a_idx] if x in matching[a_idx]), None)
                    # update arm's choice where there should only be one left
                    matching[a_idx] = [a_choice]
                    # update player's state of matched
                    for p_idx in a_choices:
                        matched[p_idx] = (p_idx == a_choice)
                        propose_order[p_idx] += (1 - (p_idx == a_choice))
    
        return np.squeeze(matching)






# Alg and Baselines

    def run_phasedETC(self,Beta):

        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        for trial in tqdm(range(self.trials), ascii=True, desc="Running the decentralized phasedETC"):
            unstable_one_trial = np.ones(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])

            players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
            players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]


            # Index_estimation 
            indexs = np.ones(self.num_players)*self.num_players-1
            arms = np.zeros(self.num_players)

            At = np.ones(self.num_players)*(-1)
            last_pulled = np.ones(self.num_arms)*(-1)

            for round in range(self.num_players):
                for p_idx in range(self.num_players):
                    At[p_idx] = arms[p_idx]

                At = At.astype(int)
                last_pulled = np.ones(self.num_arms)*(-1)
                for a_idx in range(self.num_arms):
                    if a_idx in At:
                        for p_rank in range(self.num_players):
                            if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                break
                last_pulled = last_pulled.astype(int)
               
                for p_idx in range(self.num_players):
                    if last_pulled[At[p_idx]]==p_idx:
                        regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                        rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        if At[p_idx]==0:
                            indexs[p_idx]=round
                            arms[p_idx] = 1
                    else:
                        regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                        rewards_one_trial[p_idx][round] = 0
                
            # print(indexs)
            current_player_ranking = [ np.zeros(self.num_arms) for j in range(self.num_players)]
            current_match = np.zeros(self.num_arms)

            for round in range(self.num_players,self.horizon):
                
                i = math.floor(math.log(round,2))
                # exploration
                if round-2**i+1 <= self.num_arms*math.floor(i**self.varEpsilon):
                    
                    for p_idx in range(self.num_players):
                        At[p_idx] = (round+2+indexs[p_idx]-2**i)%self.num_arms
                    # print("Explore-------round ",round, At)
                    last_pulled = np.ones(self.num_arms)*(-1)
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    # Here: whether stable matching according to last_pulled.
                    last_pulled = last_pulled.astype(int)
                    unstable_one_trial[round] = self.isUnstable(last_pulled)
                    
                    At = At.astype(int)
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]]==p_idx:
                            # update
                            # reward = np.random.binomial(1, self.players_mean[p_idx][At[p_idx]])

                            reward=np.random.normal(loc=self.players_mean[p_idx][At[p_idx]], scale=1.0, size=None)


                            players_count[p_idx][At[p_idx]]+=1
                            players_es_mean[p_idx][At[p_idx]]+= (reward-players_es_mean[p_idx][At[p_idx]]) / players_count[p_idx][At[p_idx]]
                            
                            # record
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        else:
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0
                # commit
                else:
                    for j in range(self.num_players):
                        current_player_ranking[j] = np.argsort(-players_es_mean[j])
                
                    current_match = self.Gale_Shapley(current_player_ranking)
                    
                    # At=np.zeros(self.num_players)

                    # for a_idx,p_idx in enumerate(current_match):
                    #     At[p_idx] = a_idx

                    # print("-----round ",round, At)
                    
                    # Here: whether stable matching according to last_pulled.
                    unstable_one_trial[round] = self.isUnstable(current_match)

                    for a_idx, p_idx in enumerate(current_match):
                        # print("Commit: (",p_idx, a_idx,")" )
                        regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][a_idx])
                        rewards_one_trial[p_idx][round] = self.players_mean[p_idx][a_idx]
                    
            for i in range(self.num_players):
                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            unstable_trials[trial] = unstable_one_trial
           
        np.savez('./ResultsData/Decen_PhasedETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Regret.npz', regret=regrets_trials)
        np.savez('./ResultsData/Decen_PhasedETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Reward.npz', reward=rewards_trials)
        np.savez('./ResultsData/Decen_PhasedETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Unstable.npz', unstable=unstable_trials)
        print(unstable_trials)
        cumulative_unstable = np.cumsum(np.array(unstable_trials), axis=1)
        for i in range(self.trials):
            print(cumulative_unstable[i][-1])
        


    def run_UCBD3(self, Beta):
        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        for trial in tqdm(range(self.trials), ascii=True, desc="Running the decentralized UCB-D3"):

            unstable_one_trial = np.ones(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])

            players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
            players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]
            players_ucb = [np.ones(self.num_arms) * np.inf for j in range(self.num_players)]

            # Index_estimation 
            indexs = np.ones(self.num_players)*self.num_players-1
            arms = np.zeros(self.num_players)

            At = np.ones(self.num_players)*(-1)
            last_pulled = np.ones(self.num_arms)*(-1)

            for round in range(self.num_players):
               
                for p_idx in range(self.num_players):
                    At[p_idx] = arms[p_idx]

                At = At.astype(int)
                # print("--initial-----round", round, At)
                last_pulled = np.ones(self.num_arms)*(-1)
                for a_idx in range(self.num_arms):
                    if a_idx in At:
                        for p_rank in range(self.num_players):
                            if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                break
                last_pulled = last_pulled.astype(int)
                unstable_one_trial[round] = self.isUnstable(last_pulled)
               
                for p_idx in range(self.num_players):
                    if last_pulled[At[p_idx]]==p_idx:
                        regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                        rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        if At[p_idx]==0:
                            indexs[p_idx]=round
                            arms[p_idx] = 1
                    else:
                        regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                        rewards_one_trial[p_idx][round] = 0
            print("index", indexs)

            flag = True
            phase_index = 0
            global_deletion = [[] for p_idx in range(self.num_players)]
            local_deletion = [[] for p_idx in range(self.num_players)]
            active_set =  [range(self.num_arms) for p_idx in range(self.num_players)]
            # collision_counter = [np.zeros(self.num_arms) for j in range(self.num_players)]
            while flag == True:
                phase_index+=1
                collision_counter = [np.zeros(self.num_arms) for j in range(self.num_players)]
                for p_idx in range(self.num_players):
                    active_set[p_idx] = list(set(range(self.num_arms)).difference(set(global_deletion[p_idx])))
                
                # print("active set", active_set)
       
                # range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                # print("start",self.num_players+2**(phase_index-1)-1+self.num_players*self.num_arms*(phase_index-1))
                # for round in range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                    
                # print("start", self.num_players+2**(phase_index-1)+self.num_players*self.num_arms*(phase_index-1))
                # print("end",self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index-1) )

                play_times_in_phase =  [np.zeros(self.num_arms) for j in range(self.num_players)]
                for round in range(self.num_players+2**(phase_index-1)+self.num_players*self.num_arms*(phase_index-1), self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index-1)):
                    if round >= self.horizon:
                        flag = False
                        break

                    for p_idx in range(self.num_players):
                        # local_deletion[p_idx]= [a_idx for a_idx in range(self.num_arms) if collision_counter[p_idx][a_idx] >= math.ceil(self.beta*(2**phase_index)) ]
                        # active_set[p_idx] = list(set(active_set[p_idx]).difference(set(local_deletion[p_idx])))\\


                        # find the argmax arm in active set
                    
                        for a_idx in range(self.num_arms):
                            players_ucb[p_idx][a_idx] = players_es_mean[p_idx][a_idx]+np.sqrt(2*self.gamma * np.log(round+1) / (players_count[p_idx][a_idx]+self.epsilon))
                    
                        ucb_max = -inf
                        At[p_idx] = 0
                        for a_idx in active_set[p_idx]:
                            if players_ucb[p_idx][a_idx] >= ucb_max:
                                ucb_max =  players_ucb[p_idx][a_idx]
                                At[p_idx] = a_idx

                    # print(active_set)

                    At = At.astype(int)
                    # print("----Phase", phase_index, "-----round", round, At)
                    # print("-----round", round, At)
                    last_pulled = np.ones(self.num_arms)*(-1)
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    last_pulled = last_pulled.astype(int)
                    
                    unstable_one_trial[round] = self.isUnstable(last_pulled)
               
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]]==p_idx:
                            # reward = np.random.binomial(1, self.players_mean[p_idx][At[p_idx]])
                            reward=np.random.normal(loc=self.players_mean[p_idx][At[p_idx]], scale=1.0, size=None)
                            players_count[p_idx][At[p_idx]]+=1
                            players_es_mean[p_idx][At[p_idx]]+= (reward-players_es_mean[p_idx][At[p_idx]]) / players_count[p_idx][At[p_idx]]

                            play_times_in_phase[p_idx][At[p_idx]] +=1
                        
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                            
                        else:
                            collision_counter[p_idx][At[p_idx]]+=1
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0


                # max_played = [np.argsort(-players_count[p_idx])[0] for p_idx in range(self.num_players)]
                max_played = [np.argsort(-play_times_in_phase[p_idx])[0] for p_idx in range(self.num_players)]



                # print("max_played", max_played)
                global_deletion = [[] for p_idx in range(self.num_players)]
                
                for round in range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                    if round>=self.horizon:
                        flag = False
                        break
                    t = round - self.num_players-2**phase_index-self.num_players*self.num_arms*(phase_index-1)
                    explore = [True for p_idx in range(self.num_players)]
                    for p_idx in range(self.num_players):
                        if t <= (indexs[p_idx]+1)*self.num_players-1 and t>= (indexs[p_idx])*self.num_players:
                            At[p_idx] = t%self.num_arms
                            explore[p_idx] = True
                        else:
                            At[p_idx] = max_played[p_idx]
                            explore[p_idx] = False
                    
                    # print("----Phase", phase_index, "-----round", round, At)
                    last_pulled = np.ones(self.num_arms)*(-1)
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    last_pulled = last_pulled.astype(int)
                    unstable_one_trial[round] = self.isUnstable(last_pulled)
                    
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]] == p_idx:
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        else:
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0
                            if explore[p_idx]==True and self.arms_rankings[At[p_idx]][p_idx]> self.arms_rankings[At[p_idx]][last_pulled[At[p_idx]]]:
                                global_deletion[p_idx].append(At[p_idx])
        
            for i in range(self.num_players):
                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            unstable_trials[trial] = unstable_one_trial
    

        np.savez('./ResultsData/Decen_UCBD3_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Regret.npz', regret=regrets_trials)
        np.savez('./ResultsData/Decen_UCBD3_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Reward.npz', reward=rewards_trials)
        np.savez('./ResultsData/Decen_UCBD3_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Unstable.npz', unstable=unstable_trials)
        cumulative_unstable = np.cumsum(np.array(unstable_trials), axis=1)
        for i in range(self.trials):
            print(cumulative_unstable[i][-1])
        print(unstable_trials)
     



    def run_UCBD4(self, Beta):
        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        for trial in tqdm(range(self.trials), ascii=True, desc="Running the decentralized UCB-D4"):

            unstable_one_trial = np.ones(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])

            players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
            players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]
            players_ucb = [np.ones(self.num_arms) * np.inf for j in range(self.num_players)]


            # Index_estimation 
            indexs = np.ones(self.num_players)*self.num_players-1
            arms = np.zeros(self.num_players)

            At = np.ones(self.num_players)*(-1)
            last_pulled = np.ones(self.num_arms)*(-1)

            for round in range(self.num_players):
               
                for p_idx in range(self.num_players):
                    At[p_idx] = arms[p_idx]

                At = At.astype(int)
                # print("--initial-----round", round, At)
                last_pulled = np.ones(self.num_arms)*(-1)
                for a_idx in range(self.num_arms):
                    if a_idx in At:
                        for p_rank in range(self.num_players):
                            if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                break
                last_pulled = last_pulled.astype(int)
                unstable_one_trial[round] = self.isUnstable(last_pulled)
               
                for p_idx in range(self.num_players):
                    if last_pulled[At[p_idx]]==p_idx:
                        regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                        rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        if At[p_idx]==0:
                            indexs[p_idx]=round
                            arms[p_idx] = 1
                    else:
                        regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                        rewards_one_trial[p_idx][round] = 0
            # print(indexs)

            flag = True
            phase_index = 0
            global_deletion = [[] for p_idx in range(self.num_players)]
            local_deletion = [[] for p_idx in range(self.num_players)]
            active_set =  [range(self.num_arms) for p_idx in range(self.num_players)]
            # collision_counter = [np.zeros(self.num_arms) for j in range(self.num_players)]
            while flag == True:
                phase_index+=1
                collision_counter = [np.zeros(self.num_arms) for j in range(self.num_players)]
                for p_idx in range(self.num_players):
                    active_set[p_idx] = list(set(range(self.num_arms)).difference(set(global_deletion[p_idx])))
       
                # range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                # print("start",self.num_players+2**(phase_index-1)-1+self.num_players*self.num_arms*(phase_index-1))
                # for round in range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                    
                # print("start", self.num_players+2**(phase_index-1)+self.num_players*self.num_arms*(phase_index-1))
                # print("end",self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index-1) )
                play_times_in_phase =  [np.zeros(self.num_arms) for j in range(self.num_players)]
                for round in range(self.num_players+2**(phase_index-1)+self.num_players*self.num_arms*(phase_index-1), self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index-1)):
                    if round >= self.horizon:
                        flag = False
                        break

                    for p_idx in range(self.num_players):
                        local_deletion[p_idx]= [a_idx for a_idx in range(self.num_arms) if collision_counter[p_idx][a_idx] >= math.ceil(self.beta*(2**phase_index)) ]
                        active_set[p_idx] = list(set(active_set[p_idx]).difference(set(local_deletion[p_idx])))
                        # find the argmax arm in active set
                    
                        for a_idx in range(self.num_arms):
                            players_ucb[p_idx][a_idx] = players_es_mean[p_idx][a_idx]+np.sqrt(2*self.gamma * np.log(round+1) / (players_count[p_idx][a_idx]+self.epsilon))
                    
                        ucb_max = -inf
                        At[p_idx] = 0
                        for a_idx in active_set[p_idx]:
                            if players_ucb[p_idx][a_idx] >= ucb_max:
                                ucb_max =  players_ucb[p_idx][a_idx]
                                At[p_idx] = a_idx

                  

                    At = At.astype(int)
                 
                    last_pulled = np.ones(self.num_arms)*(-1)
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    last_pulled = last_pulled.astype(int)
                    
                    unstable_one_trial[round] = self.isUnstable(last_pulled)
               
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]]==p_idx:
                            # reward = np.random.binomial(1, self.players_mean[p_idx][At[p_idx]])
                            reward=np.random.normal(loc=self.players_mean[p_idx][At[p_idx]], scale=1.0, size=None)

                            players_count[p_idx][At[p_idx]]+=1
                            players_es_mean[p_idx][At[p_idx]]+= (reward-players_es_mean[p_idx][At[p_idx]]) / players_count[p_idx][At[p_idx]]
                            play_times_in_phase[p_idx][At[p_idx]]+=1
                        
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                            
                        else:
                            collision_counter[p_idx][At[p_idx]]+=1
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0

                # max_played = [np.argsort(-players_count[p_idx])[0] for p_idx in range(self.num_players)]
                 # max_played = [np.argsort(-players_count[p_idx])[0] for p_idx in range(self.num_players)]
                max_played = [np.argsort(-play_times_in_phase[p_idx])[0] for p_idx in range(self.num_players)]
                # print("max_played", max_played)
                global_deletion = [[] for p_idx in range(self.num_players)]
                
                for round in range(self.num_players+2**phase_index +self.num_players*self.num_arms*(phase_index-1) , self.num_players+2**phase_index+self.num_players*self.num_arms*(phase_index)):
                    if round>=self.horizon:
                        flag = False
                        break
                    t = round - self.num_players-2**phase_index-self.num_players*self.num_arms*(phase_index-1)
                    explore = [True for p_idx in range(self.num_players)]
                    for p_idx in range(self.num_players):
                        if t <= (indexs[p_idx]+1)*self.num_players-1 and t>= (indexs[p_idx])*self.num_players:
                            At[p_idx] = t%self.num_arms
                            explore[p_idx] = True
                        else:
                            At[p_idx] = max_played[p_idx]
                            explore[p_idx] = False
                    
                    # print("----Phase", phase_index, "-----round", round, At)
                    last_pulled = np.ones(self.num_arms)*(-1)
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    last_pulled = last_pulled.astype(int)
                    unstable_one_trial[round] = self.isUnstable(last_pulled)
                    
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]] == p_idx:
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]
                        else:
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0
                            if explore[p_idx]==True:
                                global_deletion[p_idx].append(At[p_idx])
        
            for i in range(self.num_players):
          
                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            unstable_trials[trial] = unstable_one_trial
       

        np.savez('./ResultsData/Decen_UCBD4_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Regret.npz', regret=regrets_trials)
        np.savez('./ResultsData/Decen_UCBD4_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Reward.npz', reward=rewards_trials)
        np.savez('./ResultsData/Decen_UCBD4_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Unstable.npz', unstable=unstable_trials)
        cumulative_unstable = np.cumsum(np.array(unstable_trials), axis=1)
        for i in range(self.trials):
            print(cumulative_unstable[i][-1])
        print(unstable_trials)





    def run_CA(self, alg, Beta, TSsca):
        # run CA-UCB with alg='UCB'
        # run CA-TS with alg='TS'
        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        for trial in tqdm(range(self.trials), ascii=True, desc="Running the decentralized CA-"+alg):
            
            unstable_one_trial = np.zeros(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])
            
            if alg=='TS':
                # self.players_a = [np.ones(self.num_arms) for j in range(self.num_players)]
                # self.players_b = [np.ones(self.num_arms) for j in range(self.num_players)]

                self.players_theta = [np.zeros(self.num_arms) for j in range(self.num_players)]
                self.players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
                self.players_count = [np.ones(self.num_arms) for j in range(self.num_players)]
                for p_idx in range(self.num_players):
                    for a_idx in range(self.num_arms):
                        reward  = np.random.normal(loc=self.players_mean[p_idx][a_idx], scale=1.0, size=None)
                        self.players_es_mean[p_idx][a_idx] = reward


            elif alg=='UCB':
                self.players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
                self.players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]
                self.players_ucb = [np.ones(self.num_arms) * np.inf for j in range(self.num_players)]
           
            last_pull_player = np.zeros(self.num_players)

            last_pulled = np.zeros(self.num_arms)
            for a_idx in range(self.num_arms):
                last_pulled[a_idx] = self.arms_rankings[a_idx][-1]
            
            for round in range(self.horizon):

                if alg=='TS':
                    for j in range(self.num_players):
                        for a in range(self.num_arms):
                            # self.players_theta[j][a] = np.random.beta(self.players_a[j][a],self.players_b[j][a])
                            self.players_theta[j][a] = np.random.normal(loc=self.players_es_mean[j][a], scale=TSsca*math.sqrt(1/self.players_count[j][a]), size=None)
                elif alg=='UCB':
                    for j in range(self.num_players):
                        for a in range(self.num_arms):
                            self.players_ucb[j][a] = self.players_es_mean[j][a] + 2*np.sqrt(np.log(round+1) / ((self.players_count[j][a] + self.epsilon)))
                            

                At = np.ones(self.num_players)*(-1)
                for p_idx in range(self.num_players):
                    if np.random.binomial(1, self.p_lambda)==0:
                        plausible_arms = []
                        for a_idx in range(self.num_arms):
                            if self.arms_rankings[a_idx].index(last_pulled[a_idx])>= self.arms_rankings[a_idx].index(p_idx):
                                plausible_arms.append(a_idx)
                       
                        if alg=='TS':
                            max_theta = -inf
                            # print("Player ",p_idx, "Plausible set ",plausible_arms)
                            for a_idx in plausible_arms:
                                if max_theta <= self.players_theta[p_idx][a_idx]:
                               
                                    At[p_idx] = a_idx
                                    max_theta = self.players_theta[p_idx][a_idx]
                        elif alg=='UCB':
                            max_ucb = -inf
                            for a_idx in plausible_arms:
                            
                                if max_ucb <= self.players_ucb[p_idx][a_idx]:
                                
                                    At[p_idx] = a_idx
                                    max_ucb = self.players_ucb[p_idx][a_idx]
                    else:
                        At[p_idx] = last_pull_player[p_idx]
                last_pull_player = At

                
                # print("-----round", round, " At ", At)

                last_pulled = np.ones(self.num_arms)*(-1)
                for a_idx in range(self.num_arms):
                    rank = self.num_players
                    flag = False
                    
                    for p_idx in range(self.num_players):
                        if At[p_idx] == a_idx and self.arms_rankings[a_idx].index(p_idx)<rank:
                            flag = True
                            matched_p = p_idx
                            rank = self.arms_rankings[a_idx].index(p_idx)
                    
                    # update, a_idx, last_pulled[a_idx]
                    # p_idx = last_pulled[a_idx].astype(int)
                    if flag==True:
                        last_pulled[a_idx] = matched_p
                     
                        # reward = np.random.binomial(1, self.players_mean[matched_p][a_idx])
                        reward=np.random.normal(loc=self.players_mean[matched_p][a_idx], scale=1.0, size=None)
                        # if alg=='TS':
                        #     if reward==1:
                        #         self.players_a[matched_p][a_idx]+=1
                        #     else:
                        #         self.players_b[matched_p][a_idx]+=1
                        # elif alg=='UCB':
                        self.players_count[matched_p][a_idx]+=1
                        self.players_es_mean[matched_p][a_idx]+= (reward-self.players_es_mean[matched_p][a_idx]) / self.players_count[matched_p][a_idx]
                            

                        regrets_one_trial[matched_p][round]=max(0,self.players_mean[matched_p][self.pessimal_matching[matched_p]] - self.players_mean[matched_p][a_idx])
                        rewards_one_trial[matched_p][round] = self.players_mean[matched_p][a_idx]
                        
                # Here: whether stable matching according to last_pulled.
                # print("last pulled", last_pulled)
                last_pulled = last_pulled.astype(int)
                unstable_one_trial[round] = self.isUnstable(last_pulled)
                

                for p in range(self.num_players):
                    if p in last_pulled:
                        continue
                    else: 
                        regrets_one_trial[p][round]=self.players_mean[p][self.pessimal_matching[p]] 
                        rewards_one_trial[p][round]=0
                        

                if round==self.horizon-1:
                    print('--------'+alg+': A_t = ',At)    
                    
                # Save Data


                for a_idx in range(self.num_arms):
                    if last_pulled[a_idx]==-1:
                        last_pulled[a_idx]= self.arms_rankings[a_idx][-1]

            for i in range(self.num_players):
            
                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            unstable_trials[trial] = unstable_one_trial

        if alg=='TS':
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_'+str(TSsca)+'_Regret.npz', regret=regrets_trials)
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_'+str(TSsca)+'_Reward.npz', reward=rewards_trials)
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_'+str(TSsca)+'_Unstable.npz', unstable=unstable_trials)
        else: 
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Regret.npz', regret=regrets_trials)
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Reward.npz', reward=rewards_trials)
            np.savez('./ResultsData/Decen_'+alg+'_Beta_'+str(Beta)+'N_'+str(self.num_players)+'_Unstable.npz', unstable=unstable_trials)




    def run_ETC(self, h, Beta):
       
        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        for trial in tqdm(range(self.trials), ascii=True, desc="Running the Decentralized-ETC"):
            
            unstable_one_trial = np.zeros(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])
            
            etc_player_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
            etc_players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]

            
            for round in range(self.horizon):
                # print("----------------TS time step: ",round, "-------------")
                if round < h * self.num_arms:
                    
                    step = round % self.num_arms
                    if step ==0:
                        arm_order = [ np.random.permutation(self.num_arms).tolist() for j in range(self.num_players)]

                    At = np.ones(self.num_players)*(-1)
                    last_pulled = np.ones(self.num_arms)*(-1)

                    for p_idx in range(self.num_players):
                        At[p_idx] = arm_order[p_idx][step]

                    At = At.astype(int)
                    
                    for a_idx in range(self.num_arms):
                        if a_idx in At:
                            for p_rank in range(self.num_players):
                                if At[self.arms_rankings[a_idx][p_rank]]==a_idx:
                                    last_pulled[a_idx] = self.arms_rankings[a_idx][p_rank]
                                    break
                    last_pulled = last_pulled.astype(int)
               
                    for p_idx in range(self.num_players):
                        if last_pulled[At[p_idx]]==p_idx:
                            regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][At[p_idx]])
                            rewards_one_trial[p_idx][round] = self.players_mean[p_idx][At[p_idx]]

                            # reward = np.random.binomial(1, self.players_mean[p_idx][At[p_idx]])
                            reward=np.random.normal(loc=self.players_mean[p_idx][At[p_idx]], scale=1.0, size=None)
                            etc_players_count[p_idx][At[p_idx]]+=1
                            etc_player_es_mean[p_idx][At[p_idx]]+= (reward-etc_player_es_mean[p_idx][At[p_idx]]) / etc_players_count[p_idx][At[p_idx]]
                            
                            
                        else:
                            regrets_one_trial[p_idx][round] = self.players_mean[p_idx][self.pessimal_matching[p_idx]]
                            rewards_one_trial[p_idx][round] = 0
                
    
                    unstable_one_trial[round] = 1
                      
                
                # Commit
                else:
                    
                    if round == h * self.num_arms:
                        current_player_ranking = [ np.zeros(self.num_arms) for j in range(self.num_players)]
                        for j in range(self.num_players):
                            current_player_ranking[j] = np.argsort(-etc_player_es_mean[j])
                        current_match = self.Gale_Shapley(current_player_ranking)
                    
                    # self.two_side_market.proceed()
                    for a_idx, p_idx in enumerate(current_match):
                        # reward = np.random.binomial(1, self.players_mean[p_idx][a_idx])
    
                        # etc_players_count[p_idx][a_idx]+=1
                        # etc_player_es_mean[p_idx][a_idx]+= (reward-etc_player_es_mean[p_idx][a_idx]) / etc_players_count[p_idx][a_idx]
                        
                    
                        regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][a_idx])
                        rewards_one_trial[p_idx][round] = self.players_mean[p_idx][a_idx]
                        
                
                        
                    # Here: whether stable matching according to last_pulled.
                    unstable_one_trial[round] = self.isUnstable(current_match)
                

                if round==self.horizon-1:
                    print('--------ETC:', 'Arm matching = ',current_match)    
                    

            for i in range(self.num_players):
            
                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            unstable_trials[trial] = unstable_one_trial


        np.savez('./ResultsData/Decen_ETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'h_'+str(h)+'_Regret.npz', regret=regrets_trials)
        np.savez('./ResultsData/Decen_ETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'h_'+str(h)+'_Reward.npz', reward=rewards_trials)
        np.savez('./ResultsData/Decen_ETC_Beta_'+str(Beta)+'N_'+str(self.num_players)+'h_'+str(h)+'_Unstable.npz', unstable=unstable_trials)

        cumulative_unstable = np.cumsum(np.array(unstable_trials), axis=1)
        for i in range(self.trials):
            print(cumulative_unstable[i][-1])
        print(unstable_trials)


        cumulative_regrets=np.cumsum(np.array(regrets_trials), axis=2)
        print(cumulative_regrets)
        
