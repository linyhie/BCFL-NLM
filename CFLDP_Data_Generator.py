import numpy as np
import gurobipy as grb

'''The ogrinal CFLDP is available at http://old.math.nsc.ru/AP/benchmarks/Design/design_en.html '''

def CFLDP():
   I, J, K, R = 50, 50, 50, 3   
   a = np.zeros((J,R))   
   a[:,0] = 3
   a[:,1] = np.array([7.31,7.14,4.43,7.21,4.16,4.36,6.63,4.89,6.93,5.95,5.22,4.65,5.90,5.50,7.74,5.01,5.35,4.76,5.65,5.61,8.97,5.86,5.98,4.60,4.28,
                      4.26,6.01,7.14,5.98,4.76,6.11,5.93,5.34,4.94,4.37,4.51,5.62,4.56,6.56,7.92,6.98,5.81,7.40,4.80,6.62,8.93,7.38,8.66,5.49,5.23])
   a[:,2] = np.array([14.35,10.41,6.25,12.57,11.50,6.77,13.63,6.16,12.90,9.82,8.19,13.12,13.17,7.50,9.30,6.05,10.35,7.15,9.19,14.32,12.93,13.04,14.29,11.48,9.89,
                      5.31,11.17,9.95,9.19,13.25,8.19,9.93,13.93,8.32,8.58,8.58,8.16,10.33,7.92,16.48,12.84,11.57,12.45,11.05,8.61,11.76,14.59,11.27,8.06,11.40])
   b = a  
   d = np.array([3,9,6,4,4,3,4,9,2,6,10,6,10,8,2,7,2,3,7,5,4,4,2,2,6,8,3,7,8,4,2,6,2,9,3,4,6,8,7,5,5,2,4,1,4,3,7,6,8,4])
   cost_follower = np.zeros((J,R))    
   cost_follower[:,0] = np.array([12.70,13.67,13.20,13.03,14.58,10.26,10.49,13.07,16.65,14.86,12.90,12.50,18.76,19.63,23.37,19.28,11.69,11.17,10.74,17.06,13.30,10.90,12.94,12.30,
                                    22.82,20.50,13.32,17.04,18.18,13.33,8.18,9.30,10.58,15.18,15.08,10.03,18.82,14.08,13.92,16.08,15.29,18.97,17.51,15.12,18.69,16.75,18.76,11.16,18.34,18.25])
   cost_follower[:,1] = np.array([25.75,33.94,11.91,37.58,27.63,9.96,25.28,19.89,26.25,34.10,20.61,24.02,24.12,28.00,36.27,20.00,35.63,29.47,22.22,21.77,32.47,20.49,22.73,17.13,
                                    29.16,13.67,20.25,31.14,18.25,26.81,39.76,18.06,19.97,18.75,26.37,17.78,16.54,22.27,32.62,17.31,32.82,34.00,43.23,29.63,33.23,36.22,45.20,26.68,29.54,23.82])
   cost_follower[:,2] = np.array([60.05,45.15,17.13,46.85,24.94,32.24,35.73,28.53,66.88,54.43,47.22,62.47,59.14,42.23,43.87,28.22,48.83,22.53,20.72,72.05,54.42,46.20,62.78,21.80,
                                    34.72,25.78,35.01,45.76,41.56,28.67,46.08,52.03,52.17,32.50,16.99,26.01,24.26,32.44,33.39,47.45,51.17,26.90,67.03,44.91,29.94,65.61,39.84,47.90,26.95,50.04])
   cost_leader = cost_follower
   x_axis = np.array([81.423,23.986,58.360,70.994,16.525,33.446,35.339,82.131,17.120,89.266,22.486,50.849,66.774,34.897,63.452,3.999,25.426,
                      85.620,60.199,88.603,19.818,89.291,91.544,35.724,40.176,43.333,37.977,80.580,43.137,67.976,12.704,38.914,60.930,35.754,98.959,
                      39.608,16.806,57.243,93.555,13.555,46.719,23.052,56.692,38.770,27.251,2.071,68.009,41.762,9.797,95.959])
   y_axis = np.array([63.358,24.907,94.707,95.909,18.016,55.179,61.412,82.424,30.418,36.745,28.671,62.173,83.822,
                      22.596,8.297,8.276,49.606,81.129,9.892,50.964,39.411,99.867,33.614,96.915,88.541,70.414,94.200,29.840,70.615,54.672,76.245,
                      92.317,43.457,32.293,20.722,48.879,26.254,29.399,42.752,30.350,7.948,28.537,80.636,59.428,86.887,59.893,76.120,57.527,41.724,73.750])        
   l = np.zeros((I,I))
   for i in range(I):
       l[i] = np.sqrt(np.square(x_axis[i] - x_axis) + np.square(y_axis[i]-y_axis))           
   u = np.zeros((I,J,R))
   for j in range(J):
       for r in range(R):
           u[:,j,r] = a[j,r]/(l[:,j]+1)
   v = np.zeros((I,K,R))
   for k in range(K):
       for r in range(R):
           v[:,k,r] = b[k,r]/(l[:,k]+1)
   return d, u, v, cost_leader, cost_follower


''' Solve the problem below to get K reasonable locations of competitor’ facilities.
    After optimization, the entrant has 50 - K candicate facilities and the competitor operates 5 facilities.
'''
    
def generator_instance(lambda_c,K,o):
    d, u, v, c, g = CFLDP()
    I = d.shape[0]
    c = np.sort(c, axis=1)
    g = np.sort(g, axis=1)
    v_gt = v[:,:,0]
    beta_gt  = np.exp(v_gt/lambda_c)    
    mgt = grb.Model()
    y = mgt.addVars(50, vtype=grb.GRB.BINARY)
    w = mgt.addVars(I, ub=d)
    mgt.update()
    mgt.addConstr(y.sum() == K)
    mgt.setObjective(w.sum(), grb.GRB.MAXIMIZE)

    def SP_OA_Cut(y_last):
        phi = np.zeros(I)
        der_y = np.zeros((I,50))
        for i in range(I): 
            sum_attraction  = (beta_gt[i]@y_last)**lambda_c + o
            der_y[i] = d[i]*o*lambda_c*beta_gt[i]*(beta_gt[i]@y_last)**(lambda_c-1) / sum_attraction**2
            phi[i] = d[i]*(sum_attraction-o)/sum_attraction
        return(phi, der_y)
    
    def lazy_cut(model, where):    
        if where == grb.GRB.Callback.MIPSOL:
          y_vals = mgt.cbGetSolution(mgt._y)
          y_last = np.zeros(50)
          for f in range(50):
              y_last[f] = y_vals[f]
          phi, der_y = SP_OA_Cut(y_last)
          w_vals = mgt.cbGetSolution(mgt._w)
          for i in range(I):
              if w_vals[i] > phi[i]*(1+1e-5):
                  mgt.cbLazy(w[i] <= phi[i] + grb.quicksum(der_y[i,f]*(y[f] - y_last[f]) for f in range(50)))                 

    mgt._y, mgt._w = y, w    
    mgt.Params.OutputFlag = 0
    mgt.Params.lazyConstraints = 1
    mgt.optimize(lazy_cut) 
    
    y_open = [f for f in range(50) if y[f].x > 0.5]   
    v = v[:,y_open,:]
    g = g[y_open,:]
    for k in range(K):
        g[k] -= g[k,0]
    mask = np.ones(u.shape[1], dtype=bool)
    mask[y_open] = False
    u = u[:, mask, :]
    c = c[mask, :]    
    return(d, u, v, c, g)


if __name__ == "__main__":
     lambda_c = 0.9   # Dissimilarity factor of the competitor’ facilities
     K = 5            # Number of facilities to select for the competitor. 
                      # Number of candicate facilities for the entrant is 50 - K
     o = 1            # Attraction of the outside option, possible value 1 and 10

     d, u, v, c, g = generator_instance(lambda_c,K,o)
     
