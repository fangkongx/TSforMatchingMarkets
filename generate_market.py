import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import datetime
import torch


# market ID.


# two parameters: beta; N is the market size
# beta=-2: N=5 [0.1,0.9] for global preferences; N=3 for counterexample in appendix
# beta=-1: random preference [0.1,0.9], vary size N=5,10,15,20
# beta = 0, 10,50,,100: size N=5
# beta = 0.1/0.2/0.3/0.4:  vary Delta=0.05/0.1/0.15/0.2, N=5
# beta = -3: Fixed Delta=0.05, vary size N=5,10,20,40



Beta = [-3,-2,-1,0,10,100,1000,0.1,0.2,0.3,0.4]
Size = [5,10,15,20]


# generate market


# !!!!! global preference

beta=-2
if beta == -2:
        num_players=5
        num_arms =5
        delta = 0.5
        start = delta*num_arms-delta


        players_ranking = []
        for p_idx in range(num_players):
            players_ranking.append([0,1,2,3,4])
        
        arms_ranking = []
        for a_idx in range(num_arms):
            arms_ranking.append([0,1,2,3,4])
        
        print(players_ranking)
        print(arms_ranking)


        players_mean_value = [np.zeros(num_arms) for j in range(num_players)]
        for j in range(num_players):
            for i in range(num_arms):
                    players_mean_value[j][i] = start-delta*i
        print(players_mean_value)

        players_mean = [np.zeros([num_arms]) for j in range(num_players)]

        # change the index of players_mean
        for p_idx in range(num_players):
            for arm in range(num_arms):
                players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

        
        print(players_mean)
        np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)
    


# !!!!!Generate counterexample in appendix
num_players=3
num_arms = 3
beta = -2
players_ranking = [ [2,1,0],[0,2,1],[1,2,0] ]
arms_ranking = [[0,1,2],[0,1,2],[1,0,2] ]
        
players_mean_value = [[0.9,0.2,0.1], [0.9,0.5,0.1],[0.9,0.5,0.1]]
players_mean = [np.zeros([num_arms]) for j in range(num_players)]

        # change the index of players_mean
for p_idx in range(num_players):
            for arm in range(num_arms):
                players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)
    

print("beta = ",beta, "num_player = ", num_players )
market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
        # print("Size", len(market['player_rank']))
print("Player_rank", market['player_rank'])
print("Arm_rank", market['arm_rank'])
print("Player_mean", market['player_mean'])






# vary the Delta = 0.2,0.15,0.1,0.05, N=5

start = [0.1, 0.2, 0.3, 0.4]
end =   [0.9, 0.8, 0.7, 0.6]

num_players=5
num_arms = 5


players_ranking = [ np.random.permutation(num_arms).tolist() for j in range(num_players)]
arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]


for s in range(len(start)):
    players_mean_value = [np.linspace(end[s],start[s], num_arms) for j in range(num_players)]
    players_mean = [np.zeros([num_arms]) for j in range(num_players)]

    # change the index of players_mean
    for p_idx in range(num_players):
        for arm in range(num_arms):
            players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

    np.savez('./Markets/beta_'+str(start[s])+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)


    

# num_players=5
# for s in range(len(start)):
#     beta = start[s]
#     print("beta = ",beta, "num_player = ", num_players )
#     market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
#     # print("Size", len(market['player_rank']))
#     print("Player_rank", market['player_rank'])
#     print("Arm_rank", market['arm_rank'])
#     print("Player_mean", market['player_mean'])




# !!!!!!vary beta=0,10,50,100, N=5
num_players = 5
num_arms = 5
Beta = [0,10,50, 100]
arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]

for beta in Beta:
        players_mean = [np.zeros([num_arms]) for j in range(num_players)]
    
        # using beta to calculate the mean and then calculate the ranking
        x = np.random.uniform(low=0.0, high=1.0, size=num_arms)
        varepsilon = np.random.logistic(0, 1, size = (num_players,num_arms))
    
        barmu = np.zeros([num_players,num_arms])
        for i in range(num_players):
            for j in range(num_arms):
                barmu[i,j] = beta*x[j]+varepsilon[i][j]

        for i in range(num_players):
            for j in range(num_arms):
                for arm in range(num_arms):
                    if barmu[i][arm] <= barmu[i][j]:
                        players_mean[i][j]+=1
             
        players_mean = np.array(players_mean)/num_arms
 
        players_mean = torch.from_numpy(players_mean)
        _,ranking = players_mean.topk(num_arms, 1)
        players_ranking = ranking.numpy().tolist()
        np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)






# !!!!!vary Market size N
# Generate Beta=-3, Delta=0.2, Vary size N
N = [5, 10, 20, 40]
delta = 0.2
beta = -3
for num in N:
    num_players = num
    num_arms = num
    players_ranking = [ np.random.permutation(num_arms).tolist() for j in range(num_players)]
    arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]
        
    start = delta*num_arms
    players_mean_value = [np.zeros(num_arms) for j in range(num_players)]
    for j in range(num_players):
            for i in range(num_arms):
                    players_mean_value[j][i] = start-delta*i
    # print(players_mean_value)

    players_mean = [np.zeros([num_arms]) for j in range(num_players)]

    # change the index of players_mean
    for p_idx in range(num_players):
        for arm in range(num_arms):
            players_mean[p_idx][arm] = round(players_mean_value[p_idx][players_ranking[p_idx].index(arm)],2)
    
    print(players_mean)
      
    np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)



