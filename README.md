# BCFL-NLM 

This repository contains the testbed dataset for the manuscript
 
    "Market-entry facility location and design considering competitor reactions under nested logit model" by Yun Hui Lin, Dongdong He, Qingyun Tian, and Yitong Yu.


Use the code **CFLDP_Data_Generator.py** to generate your own instances

## Description

This paper investigates the facility location and design problem for a “new-comer” company entering a market with an existing competitor. The entrant strategically opens facilities to maximize its revenue, anticipating that the competitor may redesign its services to retain customers. Customers choose between the two companies’ facilities based on a nested logit model (NLM). In our model, facilities are grouped into two categories to reflect similarities within each company’s facilities, resulting in a two-nest NLM. We aim to determine the optimal locations for the entrant’s facilities and their design levels, considering the competitor’s potential responses and customer preferences as dictated by the NLM. To address this problem, we develop a nonlinear 0-1 bilevel program and an exact solution algorithm featuring an iterative process with two bounding problems and specialized branch-and-cut subroutines. Extensive computational experiments using a common testbed from existing literature validate our algorithm’s efficiency. Additionally, we conduct sensitivity analysis to examine the impact of NLM parameters on strategic facility location decisions and compare our NLM-based model with a multinomial logit model (MNL)-based model. Our findings show that using NLM over MNL can prevent potential revenue losses. Specifically, if the actual decision context is better represented by NLM but MNL is incorrectly applied, the entrant could face significant revenue decreases, up to 29.48%. This highlights the importance of adopting the NLM to minimize revenue loss risk, underscoring the practical value of our proposed model.


    The ogrinal CFLDP dataset is available at http://old.math.nsc.ru/AP/benchmarks/Design/design_en.html.

    It is re-processed by solving a optimization problem to get K reasonable locations of the competitor’ facilities. 
    
    The number of candicate facilities for the entrant is 50 - K.

## Replicating

- To run the code, you will need to make sure that you have already installed **Anaconda3**.

- You also need to install the solver **Gurobi** (license required)

