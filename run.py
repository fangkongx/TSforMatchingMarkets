from tokenize import Decnumber
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import datetime
import torch

from algs import Decentralized


# two parameters: beta; N is the market size
# beta=-2: N=5 [0.1,0.9] for global preferences; N=3 for counterexample in appendix
# beta=-1: random preference [0.1,0.9], vary size N=5,10,15,20
# beta = 0, 10,50,,100: size N=5
# beta = 0.1/0.2/0.3/0.4:  vary Delta=0.05/0.1/0.15/0.2, N=5
# beta = -3: Fixed Delta=0.05, vary size N=5,10,20,40




horizon = 100000
trials = 50

beta = -2
N = 5


market = np.load('./Markets/beta_'+str(beta)+'N_'+str(N)+'.npz')
num_players = N
num_arms = N
player_ranking = market['player_rank'].tolist()
arm_ranking = market['arm_rank'].tolist()
player_mean = market['player_mean'].tolist()



test = Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_CA(alg='TS', Beta = beta)

test = Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_CA(alg='UCB', Beta = beta)

test =  Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_UCBD3(Beta=beta)

test =  Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_UCBD4(Beta=beta)

test = Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_ETC(h=500, Beta=beta)

test = Decentralized(horizon = horizon, trial=trials,  num_player=num_players, num_arm=num_arms, player_ranking=player_ranking, arm_ranking=arm_ranking, player_mean=player_mean)
test.run_phasedETC(Beta=beta)
