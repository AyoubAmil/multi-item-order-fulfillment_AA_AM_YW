# magician_problem.py

from order_fulfillment_network import OrderFulfillment
import numpy as np
from collections import defaultdict

class MagicianProblem:
    def __init__(self, x, gamma, order_fulfillment: OrderFulfillment):
        self.x = x # Probability list of breaking a wand
        self.gamma = gamma # Lower bound on the probability of opening a box
        self.T = order_fulfillment.T  # For convinience, using the same T from the order_fulfillment object
        self.theta = [] # List of thresholds
        self.prob_rand = [] # List of probabilities of opening a box
        self.open_list = [] # List of lists of 0s and 1s, where 1 means that the box is opened
        self.all_cdfs = [] # List of all CDFs
    
    def compute_CDF_Wtplus1(self, t, F_Wt, s_t):
        """
        Compute the CDF of W_{t+1} given the CDF of W_t (F_Wt) and the probability of opening a box at time t (as a function of broken wands l).
        """
        F_Wtplus1 = defaultdict(int)
        if t == 0:
            F_Wtplus1.update({0: 1, 1: 1})
        else:
            for l in range(t + 1):
                # Compute the CDF of W_{t+1} for l broken wands
                F_Wtplus1[l] = F_Wt[l] - s_t.get(l, 0) * self.x[t - 1] * (F_Wt[l] - F_Wt.get(l - 1, 0))
            F_Wtplus1[t + 1] = 1
        return F_Wtplus1
    
    def compute_all_CDFs(self):
      F_Wt = dict()
      s_t = defaultdict(int)
      for t in range(self.T):
            F_Wtplus1 = self.compute_CDF_Wtplus1(t, F_Wt, s_t)
            #update s_t
            theta_tplus1 = min(key for key, value in F_Wtplus1.items() if value >= self.gamma)
            s_t = defaultdict(int)
            for l in range(t + 2):
              if l < theta_tplus1:
                  s_t[l] = 1
              elif l == theta_tplus1:
                  # Compute the probability of opening a box at time t
                  s_t[l] = (self.gamma - F_Wtplus1[theta_tplus1 - 1]) / (F_Wtplus1[theta_tplus1] - F_Wtplus1[theta_tplus1 - 1])
            F_Wt = F_Wtplus1
            self.all_cdfs.append(F_Wt)

    def solve_withcdfs(self, all_cdfs):
        s_t = defaultdict(int)
        for t in range(self.T):
            F_Wtplus1 = all_cdfs[t]
            theta_tplus1 = min(key for key, value in F_Wtplus1.items() if value >= self.gamma)
            self.theta.append(theta_tplus1)
            s_t = defaultdict(int)
            open = []
            for l in range(t + 2):
                if l < theta_tplus1:
                    s_t[l] = 1
                    open.append(1)
                elif l == theta_tplus1:
                    # Compute the probability of opening a box at time t
                    s_t[l] = (self.gamma - F_Wtplus1[theta_tplus1 - 1]) / (F_Wtplus1[theta_tplus1] - F_Wtplus1[theta_tplus1 - 1])
                    open.append(np.random.binomial(1, s_t[l])) # Open with probability s_t[l] (added seed for reproducibility)
                    self.prob_rand.append(s_t[l])
                else:
                    open.append(0)
            self.open_list.append(open)
        return self.theta, self.open_list, self.prob_rand

    def solve(self):
        F_Wt = dict()
        s_t = defaultdict(int)
        for t in range(self.T):
            F_Wtplus1 = self.compute_CDF_Wtplus1(t, F_Wt, s_t)
            theta_tplus1 = min(key for key, value in F_Wtplus1.items() if value >= self.gamma)
            self.theta.append(theta_tplus1)
            s_t = defaultdict(int)
            open = []
            for l in range(t + 2):
                if l < theta_tplus1:
                    s_t[l] = 1
                    open.append(1)
                elif l == theta_tplus1:
                    # Compute the probability of opening a box at time t
                    s_t[l] = (self.gamma - F_Wtplus1[theta_tplus1 - 1]) / (F_Wtplus1[theta_tplus1] - F_Wtplus1[theta_tplus1 - 1])
                    open.append(np.random.binomial(1, s_t[l])) # Open with probability s_t[l] (added seed for reproducibility)
                    self.prob_rand.append(s_t[l])
                else:
                    open.append(0)
            self.open_list.append(open)
            F_Wt = F_Wtplus1
        return self.theta, self.open_list, self.prob_rand 
    
    
    
    