import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import datetime
import torch
import math

def fmt(x, y):
    if x >= 1000000:
        if x % 1000000 == 0:
            return str(int(x / 1000000)) + 'm'
        else:
            return str(int(x / 100000) / 10) + 'm'
    elif x >= 1000:
        if x%1000==0:
                return str(int(x /1000)) + 'k'
        else:
                return str(round(x /1000,1)) + 'k'
    else:
        return str(int(x))





# regret_list: dimention: alg, parameter, trials, horizon
def plot_dif_N(alg_list, regret_list, regret_list2, horizon, trials, type_value, type_value_list, ylabel, ylabel2, title, path, y_max, y_max2):
        num_Parameters = len(type_value_list)

      

        matplotlib.rcParams.update({'figure.autolayout': True})
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=16)  # fontsize of the figure title

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5,5))

        # print(type(ax1))
        # print(ax1.shape)
        

        errorevery = 20000

        fmt_map = ['-','--','-.',':']
        colorMap = ['red','seagreen', 'chocolate' , 'dodgerblue', 'darkorchid', 'orange', 'mediumpurple', 'deepskyblue', 'seagreen','salmon', 'darksalmon', 'lightsalmon','chocolate','grey','darkgrey', 'palevioletred','thistle','lightpink']
        
        

        if type_value=='beta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\beta$ = '+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='N':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+type_value+'='+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='delta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\Delta$ = '+str(type_value_list[i]), errorevery = errorevery)
        ax1.locator_params('x',nbins=6)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        ax1.set_xlabel(r"Round $t$")
        ax1.set_ylabel(ylabel)
        if y_max!='none':
                ax1.set_ylim(0, y_max)
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))

        


        if type_value=='beta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\beta$ = '+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='N':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+type_value+'='+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='delta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\Delta$ = '+str(type_value_list[i]), errorevery = errorevery)
        

        ax2.locator_params('x',nbins=6)
        # 
        ax2.set_xlabel(r"Round $t$")
        ax2.set_ylabel(ylabel2)
        if y_max2!='none':
                ax2.set_ylim(0, y_max2)
        ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))

      
        # # fig.suptitle(title,fontsize=16)
        # a=fig.suptitle(title)
        # set(a,'FontSize',16)

        fig.suptitle(title, fontsize=20)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        plt.savefig(path)
        plt.close(fig)



def plot_dif_N_separateLegend(start, alg_list, regret_list, regret_list2, horizon, trials, type_value, type_value_list, ylabel, ylabel2, title, path, y_max, y_max2):
        num_Parameters = len(type_value_list)

      

        matplotlib.rcParams.update({'figure.autolayout': True})
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=16)  # fontsize of the figure title

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5,5))

        # print(type(ax1))
        # print(ax1.shape)
        

        errorevery = 20000

        fmt_map = ['-','--','-.',':']
        colorMap = ['red','seagreen', 'chocolate' , 'dodgerblue', 'darkorchid', 'orange', 'mediumpurple', 'deepskyblue', 'seagreen','salmon', 'darksalmon', 'lightsalmon','chocolate','grey','darkgrey', 'palevioletred','thistle','lightpink']
        
        
        alg_list_copy = alg_list
        alg_list  = alg_list_copy[0:2]
        if type_value=='beta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\beta$ = '+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='N':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+type_value+'='+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='delta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list[alg][i], axis=0)
                                ax1.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+r'$\Delta$ = '+str(type_value_list[i]), errorevery = errorevery)
        ax1.locator_params('x',nbins=6)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        ax1.set_xlabel(r"Round $t$")
        ax1.set_ylabel(ylabel)
        if y_max!='none':
                ax1.set_ylim(0, y_max)
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax1.legend()

        

        alg_list  = alg_list_copy[2:4]
        if type_value=='beta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg+start], label=alg_list[alg]+', '+r'$\beta$ = '+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='N':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg+start], label=alg_list[alg]+', '+type_value+'='+str(type_value_list[i]), errorevery = errorevery)
        elif type_value=='delta':
                for alg in range(len(alg_list)):
                        for i in range(num_Parameters):
                                regret_mean = np.mean(regret_list2[alg][i], axis=0)
                                ax2.errorbar(range(horizon), regret_mean,  fmt=fmt_map[i],yerr=np.std(regret_list2[alg][i],axis=0)/np.sqrt(trials), color=colorMap[alg+start], label=alg_list[alg]+', '+r'$\Delta$ = '+str(type_value_list[i]), errorevery = errorevery)
        

        ax2.locator_params('x',nbins=6)
        # 
        ax2.set_xlabel(r"Round $t$")
        ax2.set_ylabel(ylabel2)
        if y_max2!='none':
                ax2.set_ylim(0, y_max2)
        ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax2.set_yticks([])
        ax2.legend()

      
        # # fig.suptitle(title,fontsize=16)
        # a=fig.suptitle(title)
        # set(a,'FontSize',16)

        fig.suptitle(title, fontsize=20)

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        plt.savefig(path)
        plt.close(fig)




# !!!!!!!!!!!! different Delta
# different Delta:  cumulative unstable
def run_plot_varyDelta_stable(h,filesetting, pathsetting, alg_filename_list, alg_list, varyParam_list,varyParam_file_list):
        horizon = 100000
        trials = 50
        
        cumulative_unstable = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])
        averaged_unstable = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                for para in range(len(varyParam_list)):
                        print("load delta = ", varyParam_file_list[para])
                        if alg_filename_list[alg]=='ETC':
                                unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_'+str(varyParam_file_list[para])+'N_5h_'+str(h[alg])+'_Unstable.npz')
                        else:
                                unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_'+str(varyParam_file_list[para])+'N_5_Unstable.npz')
                

                        cumulative_unstable[alg][para] = np.cumsum(np.array(unstable_trials['unstable']), axis=1)
                 
                        # for trial in range(trials):
                        #         print("load trial",trial)  
                        #         cumulative_unstable[alg][para][trial]=np.cumsum(np.array(unstable_trials['unstable'][trial][0:horizon]), axis=0)
                 
                              
        return cumulative_unstable
        plot_dif_N(alg_list=alg_list, regret_list=cumulative_unstable, horizon=horizon, trials=trials, type_value='delta', type_value_list=varyParam_list, ylabel='Cumulative Market Unstability', title='(b) different preference gap, random preferences, N=5, K=5', path='./Results/'+pathsetting+'_varyDelta_cumStable.pdf',y_max=12000 )
        # plot_dif_N(alg_list=alg_list, regret_list=averaged_unstable, horizon=horizon, trials=trials, type_value='delta', type_value_list=varyParam_list, ylabel='Averaged Market Unstability', title=pathsetting+', different preference gap', path='./Results/'+pathsetting+'_varyDelta_aveStable.pdf',y_max=y_max)




# different Delta:  cumulative regret
def run_plot_varyDelta_regret(h,filesetting, pathsetting, alg_filename_list, alg_list, varyParam_list,varyParam_file_list):
        horizon = 100000
        trials = 50
        num_players = 5
        
        cumulative_regrets = np.zeros([len(alg_list), len(varyParam_list), num_players, trials, horizon])

        cumulative_max_regrets = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])
        # averaged_max_regrets = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                for para in range(len(varyParam_list)):
                        print("load delta = ", varyParam_file_list[para])
                        
                        if alg_filename_list[alg]=='ETC':
                                regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_'+str(varyParam_file_list[para])+'N_5h_'+str(h[alg])+'_Regret.npz')
                        else:
                                regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_'+str(varyParam_file_list[para])+'N_5_Regret.npz')

                        cumulative_regrets[alg][para] = np.cumsum(np.array(regrets_trials['regret']), axis=2)
                        
                        # for p_idx in range(num_players):
                        #         print("load player",p_idx)
                        #         for trial in range(trials):
                        #                 print("load trial",trial)
                        #                 cumulative_regrets[alg][para][p_idx][trial]=np.cumsum(np.array(regrets_trials['regret'][p_idx][trial][0:horizon]), axis=0)
                        
                        max_p_idx = 0
                        max_cum_regret = 0
                        for p_idx in range(num_players):
                                regret_mean = np.mean(cumulative_regrets[alg][para][p_idx], axis=0)[horizon-1]
                                if regret_mean > max_cum_regret:
                                        max_p_idx = p_idx
                                        max_cum_regret = regret_mean
        
                        cumulative_max_regrets[alg][para] = cumulative_regrets[alg][para][max_p_idx]
                        # for trial in range(trials):
                        #         cumulative_max_regrets[alg][para][trial]=cumulative_regrets[alg][para][max_p_idx][trial]
                        #         # averaged_max_regrets[alg][para][trial] = cumulative_max_regrets[alg][para][trial]/range(1,horizon+1)

        return cumulative_max_regrets
        plot_dif_N(alg_list=alg_list, regret_list=cumulative_max_regrets, horizon=horizon, trials=trials, type_value='delta', type_value_list=varyParam_list, ylabel="Maximum Cumulative Regret among Players", title='(a) different preference gap, random preferences, N=5, K=5', path='./Results/'+pathsetting+'_varyDelta_cumRegret.pdf',y_max='none' )
        # plot_dif_N(alg_list=alg_list, regret_list=averaged_max_regrets, horizon=horizon, trials=trials, type_value='delta', type_value_list=varyParam_list, ylabel="Maximum Cumulative Regret among Players", title=pathsetting+', different preference gap', path='./Results/'+pathsetting+'_varyDelta_aveRegret.pdf',y_max=y_max)




horizon = 100000
trials = 50
filesetting='Decen'
pathsetting='decentralized'

alg_filename_list = ['TS', 'UCB', 'PhasedETC','ETC']
h = [0,0,0,200]
alg_list = ['CA-TS', 'CA-UCB', 'P-ETC','D-ETC']


varyParam_file_list=[0.1,0.2,0.3,0.4]
varyParam_list=[0.2,0.15,0.1,0.05]

unstable = run_plot_varyDelta_stable(h=h,filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list,varyParam_file_list=varyParam_file_list)
regret = run_plot_varyDelta_regret(h=h, filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list,varyParam_file_list=varyParam_file_list)

plot_dif_N(alg_list=alg_list, regret_list=regret, regret_list2=unstable, horizon=horizon, trials=trials, type_value='delta', type_value_list=varyParam_list, ylabel="Maximum Cumulative Stable Regret", ylabel2="Cumulative Market Unstability", title=r'Different preference gaps, random preferences, $N=5$ players, $K=5$ arms', path='./Results/'+pathsetting+'_varyDelta.pdf',y_max=1500,y_max2=5000)
        






# !!!!!!!!!
# the setting with different N
# different N:  cumulative unstable
def run_plot_varyN_stable(TSsca, h,filesetting, pathsetting, alg_filename_list, alg_list, varyParam_list):
        horizon = 100000
        trials = 50
        
        cumulative_unstable = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                for para in range(len(varyParam_list)):
                        print("load N = ", varyParam_list[para])
                        if True:
                                if alg_filename_list[alg]=='ETC':
                                        unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'h_'+str(h[alg])+'_Unstable.npz')
                                elif alg_filename_list[alg]=='TS' and (varyParam_list[para]==10 or varyParam_list[para]==20):
                                        unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'_'+str(TSsca[para])+'_Unstable.npz')
                                else:
                                        unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'_Unstable.npz')
                
                        # for trial in range(trials):
                        #         print("load trial",trial)  
                        #         cumulative_unstable[alg][para][trial]=np.cumsum(np.array(unstable_trials['unstable'][trial][0:horizon]), axis=0)

                        cumulative_unstable[alg][para] = np.cumsum(np.array(unstable_trials['unstable']), axis=1)

        # print(cumulative_unstable)
        # for trial in range(trials):
        #         print(cumulative_unstable[0][0][trial])
        print(cumulative_unstable[0][0][35])


        return cumulative_unstable
        # plot_dif_N(alg_list=alg_list, regret_list=cumulative_unstable, horizon=horizon, trials=trials, type_value='N', type_value_list=varyParam_list, ylabel='Cumulative Market Unstability', title='(b) different market size, random preferences', path='./Results/'+pathsetting+'_varyN_cumStable.pdf',y_max='none')
        # +r'$\Delta$ = 0.05'
        # plot_dif_N(alg_list=alg_list, regret_list=averaged_unstable, horizon=horizon, trials=trials, type_value='N', type_value_list=varyParam_list, ylabel='Averaged Market Unstability', title='different market size, random preferences', path='./Results/'+pathsetting+'_varyN_aveStable.pdf',y_max='none')




# different N: cumulative regret
def run_plot_varyN_regret(TSsca, h, filesetting, pathsetting, alg_filename_list, alg_list, varyParam_list):
        horizon = 100000
        trials = 50
    
        cumulative_max_regrets = np.zeros([len(alg_list),len(varyParam_list), trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                for para in range(len(varyParam_list)):
                        print("load N = ", varyParam_list[para])
                        num_players = varyParam_list[para]
                        if True:
                                if alg_filename_list[alg]=='ETC':
                                        regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'h_'+str(h[alg])+'_Regret.npz')
                                elif alg_filename_list[alg]=='TS' and (varyParam_list[para]==10 or varyParam_list[para]==20):
                                        regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'_'+str(TSsca[para])+'_Regret.npz') 
                                else:
                                        regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-3N_'+str(varyParam_list[para])+'_Regret.npz')
                        
                        
                        cumulative_regrets = np.zeros([ num_players, trials, horizon])
                        cumulative_regrets=np.cumsum(np.array(regrets_trials['regret']), axis=2)
                        # for p_idx in range(num_players):
                        #         print("load player",p_idx)
                        #         for trial in range(trials):
                        #                 print("load trial",trial)
                        #                 cumulative_regrets[p_idx][trial]=np.cumsum(np.array(regrets_trials['regret'][p_idx][trial][0:horizon]), axis=0)
                        
                        max_p_idx = 0
                        max_cum_regret = 0
                        for p_idx in range(num_players):
                                regret_mean = np.mean(cumulative_regrets[p_idx], axis=0)[horizon-1]
                                if regret_mean > max_cum_regret:
                                        max_p_idx = p_idx
                                        max_cum_regret = regret_mean

                        
                        cumulative_max_regrets[alg][para] = cumulative_regrets[max_p_idx]
        
        return cumulative_max_regrets
        
        
# horizon = 100000
# trials = 50
# filesetting='Decen'
# pathsetting='decentralized'

# alg_filename_list = ['TS', 'UCB', 'PhasedETC','ETC']
# h = [0,0,0,200]
# alg_list = ['CA-TS', 'CA-UCB', 'P-ETC','D-ETC']


# varyParam_list=[5,10,20,40]
# unstable = run_plot_varyN_stable(h,filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)
# regret = run_plot_varyN_regret(h=h, filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)

# plot_dif_N(alg_list=alg_list, regret_list=regret, regret_list2=unstable, horizon=horizon, trials=trials, type_value='N', type_value_list=varyParam_list, ylabel="Maximum Cumulative Stable Regret", ylabel2="Cumulative Market Unstability", title='different market sizes, random preferences', path='./Results/'+pathsetting+'_varyN.pdf',y_max=80000,y_max2=30000)
        



horizon = 100000
trials = 50
filesetting='Decen'
pathsetting='decentralized'

alg_filename_list = ['TS', 'UCB']
h = [0,0,0,200]

TSsca = [0,0.75,0.85,0]
alg_list = ['CA-TS', 'CA-UCB']

varyParam_list=[5,10,20,40]
# unstable = run_plot_varyN_stable(TSsca = TSsca, h=h,filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)
regret = run_plot_varyN_regret(TSsca = TSsca, h=h, filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)




horizon = 100000
trials = 50
filesetting='Decen'
pathsetting='decentralized'

alg_filename_list = ['PhasedETC','ETC']
h = [0,200]
alg_list = ['P-ETC','D-ETC']


varyParam_list=[5,10,20,40]
unstableETC = run_plot_varyN_stable(TSsca = TSsca, h=h, filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)
regretETC = run_plot_varyN_regret(TSsca = TSsca, h=h, filesetting=filesetting,pathsetting=pathsetting,alg_filename_list=alg_filename_list,alg_list=alg_list,varyParam_list=varyParam_list)


full_alg_list = ['CA-TS', 'CA-UCB', 'P-ETC','D-ETC']

plot_dif_N_separateLegend(start = 2, alg_list=full_alg_list, regret_list=regret, regret_list2=regretETC, horizon=horizon, trials=trials, type_value='N', type_value_list=varyParam_list, ylabel="Maximum Cumulative Stable Regret", ylabel2="", title='Different market sizes, random preferences', path='./Results/'+pathsetting+'_varyN_regret.pdf',y_max=45000,y_max2=45000)
      





























# regret_list: dimention: alg, player, trials, horizon
def plot_regret_allPlayers_for_difAlg(alg_list,player_start, player_end, regret_list, horizon, trials, y_label, title, path, y_max):
        matplotlib.rcParams.update({'figure.autolayout': True})
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=16)  # fontsize of the figure title

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig, ax = plt.subplots()
        errorevery = 10000

        fmt_map = ['-','--',':','-.']

        players=[r'Player $p_1$',r'Player $p_2$',r'Player $p_3$',r'Player $p_4$',r'Player $p_5$']
        colorMap = ['red','seagreen', 'chocolate' , 'dodgerblue', 'darkorchid', 'orange', 'mediumpurple', 'deepskyblue', 'seagreen','salmon', 'darksalmon', 'lightsalmon','chocolate','grey','darkgrey', 'palevioletred','thistle','lightpink']
        
        
        # colorMap = [(1,0,0), (0.8,0,0.2 ), (0.6, 0, 0.4), (0.4,0, 0.6), (0.2, 0, 0.8 ), (0,0,1)]
        
        for alg in range(len(alg_list)):
            for p_idx in range(player_start,player_end):
                regret_mean = np.mean(regret_list[alg][p_idx], axis=0)
                plt.errorbar(range(horizon), regret_mean, fmt=fmt_map[0], yerr=np.std(regret_list[alg][p_idx])/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg]+', '+players[p_idx], errorevery = errorevery)

        
        plt.locator_params('x',nbins=6)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel(y_label)
        if y_max!='none':
                plt.ylim(0, y_max)

        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.yticks([])


        
        plt.title(title,fontsize = 20)
       
        plt.savefig(path)
        plt.close(fig)


# regret_list: dimention: alg, trials, horizon
def plot_max_regret(alg_list, regret_list, horizon, trials, ylabel, title, path,y_max):
        
        matplotlib.rcParams.update({'figure.autolayout': True})
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('figure', titlesize=16)  # fontsize of the figure title

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig, ax = plt.subplots()
        errorevery = 10000

        fmt_map = ['-','--','-.',':']
        colorMap = ['red','seagreen', 'chocolate' , 'dodgerblue', 'darkorchid', 'orange', 'mediumpurple', 'deepskyblue', 'seagreen','salmon', 'darksalmon', 'lightsalmon','chocolate','grey','darkgrey', 'palevioletred','thistle','lightpink']
        
        

        for alg in range(len(alg_list)):
                regret_mean = np.mean(regret_list[alg], axis=0)
                plt.errorbar(range(horizon), regret_mean,  fmt=fmt_map[0],yerr=np.std(regret_list[alg],axis=0)/np.sqrt(trials), color=colorMap[alg], label=alg_list[alg], errorevery = errorevery)
       
        plt.locator_params('x',nbins=6)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)

        if y_max!='none':
                plt.ylim(0, y_max)

        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        
        
        # plt.title(title,)
        plt.title(title,fontsize = 20)
        plt.savefig(path)

        plt.close(fig)





# Run the above functions to plot the result in different settings

# Plot all players regret for beta=-2, N=5
def run_plot_global_all(h,filesetting, pathsetting, alg_filename_list, alg_list):
        horizon = 100000
        trials = 50

        num_players = 3

        cumulative_regrets = np.zeros([len(alg_list), num_players, trials, horizon])
        # averaged_regrets = np.zeros([len(alg_list),num_players, trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                if alg_filename_list[alg]=='ETC':
                        regrets_trials  = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_5h_'+str(h[alg])+'_Regret.npz')
                else:
                        # regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_5_Regret.npz')
                        regrets_trials = np.load('./cenResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_3_Regret.npz')
                
                cumulative_regrets[alg] = np.cumsum(np.array(regrets_trials['regret']), axis=2)
               

        # plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=0,player_end=1,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(a) Player $p_1$', path='./Results/'+pathsetting+'_global_player0_cumRegret.pdf',y_max=600)
        # plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=1,player_end=2,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(b) Player $p_2$', path='./Results/'+pathsetting+'_global_player1_cumRegret.pdf',y_max=1200)
        # plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=2,player_end=3,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(c) Player $p_3$', path='./Results/'+pathsetting+'_global_player2_cumRegret.pdf',y_max=1200)
        # plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=3,player_end=4,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(d) Player $p_4$', path='./Results/'+pathsetting+'_global_player3_cumRegret.pdf',y_max=800)
        # plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=4,player_end=5,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(e) Player $p_5$', path='./Results/'+pathsetting+'_global_player4_cumRegret.pdf',y_max='none')

        plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=0,player_end=1,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(a) Player $p_1$', path='./Results/'+pathsetting+'_global_player0_cumRegret.pdf',y_max='none')
        plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=1,player_end=2,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(b) Player $p_2$', path='./Results/'+pathsetting+'_global_player1_cumRegret.pdf',y_max='none')
        plot_regret_allPlayers_for_difAlg(alg_list=alg_list, player_start=2,player_end=3,regret_list=cumulative_regrets, horizon=horizon,trials=trials, y_label='Cumulative Stable Regret', title=r'(c) Player $p_3$', path='./Results/'+pathsetting+'_global_player2_cumRegret.pdf',y_max='none')
        

# alg_filename_list = ['TS', 'UCB', 'PhasedETC','ETC','UCBD4']
# h = [0,0,0,200,0]
# alg_list = ['CA-TS', 'CA-UCB', 'P-ETC','D-ETC','UCB-D4']

# run_plot_global_all(h,filesetting='Decen', pathsetting='decentralized', alg_filename_list=alg_filename_list, alg_list=alg_list)


# # !!!!!!!!!!!!!!! counterexample
# h = [0,0,0,200,0]
# alg_filename_list = ['TS', 'UCB']
# alg_list = ['Centralized-TS', 'Centralized-UCB']
# run_plot_global_all(h, filesetting='Cen', pathsetting='centralized', alg_filename_list=alg_filename_list, alg_list=alg_list)




# !!!!!!!!!!!!!!
# Plot Unstable for beta=-2, N=3
def run_plot_unstable(h,filesetting, pathsetting, alg_filename_list, alg_list,y_max ):
        horizon = 100000
        trials = 50
        
        cumulative_unstable = np.zeros([len(alg_list), trials, horizon])
        # averaged_unstable = np.zeros([len(alg_list), trials, horizon])

        for alg in range(len(alg_filename_list)):
                print("load alg",alg_list[alg])
                unstable_trials = np.load('./cenResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_3_Unstable.npz')

                # if alg_filename_list[alg]=='ETC':
                #         unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_5h_'+str(h[alg])+'_Unstable.npz')
                # else:
                #         unstable_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_5_Unstable.npz')
                
                cumulative_unstable[alg] = np.cumsum(np.array(unstable_trials['unstable']), axis=1)
               

        # plot_max_regret(alg_list=alg_list,regret_list=cumulative_unstable, horizon=horizon,trials=trials, ylabel='Cumulative Market Unstability', title='Example mentioned in Section 7', path='./Results/'+pathsetting+'_global_cumStable.pdf',y_max=y_max)
        plot_max_regret(alg_list=alg_list,regret_list=cumulative_unstable, horizon=horizon,trials=trials, ylabel='Cumulative Market Unstability', title='(d) Cumulative market unstability', path='./Results/'+pathsetting+'_global_cumStable.pdf',y_max=y_max)
        

        # plot_max_regret(alg_list=alg_list,regret_list=averaged_unstable, horizon=horizon,trials=trials, ylabel='Averaged Market Unstability', title=', counterexample,  (3,3), Beta(1,1)', path='./Results/'+pathsetting+'_counter_aveStable_1.pdf',y_max='none')
        
        # !!! draw the counterexample
        # plot_max_regret(alg_list=alg_list,regret_list=cumulative_unstable, horizon=horizon,trials=trials, ylabel='Cumulative Market Unstability', title=', counterexample, (3,3), Beta(1,1)', path='./Results/'+pathsetting+'_counter_cumStable_1.pdf',y_max=y_max)
        # plot_max_regret(alg_list=alg_list,regret_list=averaged_unstable, horizon=horizon,trials=trials, ylabel='Averaged Market Unstability', title=', counterexample,  (3,3), Beta(1,1)', path='./Results/'+pathsetting+'_counter_aveStable_1.pdf',y_max='none')
       
        # plot_max_regret(alg_list=alg_list,regret_list=cumulative_unstable, horizon=horizon,trials=trials, ylabel='Cumulative Regret', title=pathsetting+', global preferences, 5 players, 5 arms', path='./Results/'+pathsetting+'_global_cumRegret.pdf',y_max=y_max)
        # plot_max_regret(alg_list=alg_list,regret_list=averaged_unstable, horizon=horizon,trials=trials, ylabel='Averaged Regret', title=pathsetting+', global preferences, 5 players, 5 arms', path='./Results/'+pathsetting+'_global_aveRegret.pdf',y_max='none')



# # !!!!!!!!!!!!!!! counter example
# h = [0,0,0,200,0]
# alg_filename_list = ['TS', 'UCB']
# alg_list = ['Centralized-TS', 'Centralized-UCB']
# run_plot_unstable(h,filesetting='Cen', pathsetting='centralized', alg_filename_list=alg_filename_list, alg_list=alg_list, y_max='none')


# # run decentralized

# alg_filename_list = ['TS', 'UCB', 'PhasedETC','ETC','UCBD4']
# h = [0,0,0,200,0]
# alg_list = ['CA-TS', 'CA-UCB', 'P-ETC','D-ETC','UCB-D4']
# run_plot_unstable(h=h,filesetting='Decen', pathsetting='decentralized', alg_filename_list=alg_filename_list, alg_list=alg_list, y_max=5000)





# # Plot Regret global preference, beta=-2,N=5
# def run_plot_global_max(filesetting, pathsetting, alg_filename_list, alg_list,y_max):
#         horizon = 100000
#         trials = 50

#         num_players = 3
#         num_arms = 3

#         # Plot the maximium regret among players

#         cumulative_regrets = np.zeros([len(alg_list), num_players, trials, horizon])

#         cumulative_max_regrets = np.zeros([len(alg_list),  trials, horizon])
#         averaged_max_regrets = np.zeros([len(alg_list), trials, horizon])

#         for alg in range(len(alg_filename_list)):
#                 print("load alg",alg_list[alg])
                
#                 regrets_trials = np.load('./ResultsData/'+filesetting+'_'+alg_filename_list[alg]+'_Beta_-2N_3_Regret.npz')
#                 for p_idx in range(num_players):
#                         print("load player",p_idx)
#                         for trial in range(trials):
#                                 print("load trial",trial)
#                                 cumulative_regrets[alg][p_idx][trial]=np.cumsum(np.array(regrets_trials['regret'][p_idx][trial][0:horizon]), axis=0)
                        
#                 max_p_idx = 0
#                 max_cum_regret = 0
#                 for p_idx in range(num_players):
#                         regret_mean = np.mean(cumulative_regrets[alg][p_idx], axis=0)[horizon-1]
#                         if regret_mean > max_cum_regret:
#                                 max_p_idx = p_idx
#                                 max_cum_regret = regret_mean
        
#                 for trial in range(trials):
#                         cumulative_max_regrets[alg][trial]=cumulative_regrets[alg][max_p_idx][trial]
#                         averaged_max_regrets[alg][trial] = cumulative_max_regrets[alg][trial]/range(1,horizon+1)

#         plot_max_regret(alg_list=alg_list, regret_list=cumulative_max_regrets, horizon=horizon, trials=trials, ylabel="Maximum Cumulative Regret among Players", title=pathsetting+', counterexample, (3,3), Beta(1,1)', path='./Results/'+pathsetting+'_counter_cumRegret_1.pdf', y_max=y_max)
#         plot_max_regret(alg_list=alg_list, regret_list=averaged_max_regrets, horizon=horizon, trials=trials, ylabel="Maximum Averaged Regret among Players",  title = pathsetting+', counterexample, (3,3), Beta(1,1)',path='./Results/'+pathsetting+'_counter_aveRegret_1.pdf', y_max='none')


# alg_filename_list = ['TS', 'UCB', 'PhasedETC','UCBD4']
# alg_list = ['CA-TS', 'CA-UCB', 'PhasedETC','UCB-D4']
# run_plot_global_max(filesetting='Decen', pathsetting='decentralized', alg_filename_list=alg_filename_list, alg_list=alg_list, y_max=1500)

# # !!!!!!!!!!!!!!! counter example
# alg_filename_list = ['TS', 'UCB']
# alg_list = ['Centralized-TS', 'Centralized-UCB']
# run_plot_global_max(filesetting='Cen', pathsetting='centralized', alg_filename_list=alg_filename_list, alg_list=alg_list, y_max='none')




 


