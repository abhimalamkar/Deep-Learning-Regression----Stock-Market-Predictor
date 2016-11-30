import numpy as np
import time

#variables
n_hidden = 10
n_in = 10
#outputs
n_out =10
#sample data
n_sample = 300

#hyperparameter
learning_rate = 0.01
momentum = 0.9

#seeding manually
np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
      return 1-np.tanh(x)**2

def train(x,t,V,W,bv,bw):#x is input data ,t is tranpose,v,W layers,bwbv are biases
     #forward propagation -- matrix multiply + biases
     A = np.dot(x,V) + bv
     Z = np.tanh(A)
     
     B = np.dot(Z,W) + bw
     Y = sigmoid(B)

     #backword propagation
     Ew = Y-t
     Ev = tanh_prime(A) * np.dot(W,Ew)
     #predict loss
     dW = np.outer(Z,Ew)
     dV = np.outer(X,Ev)
     #cross entropy
     loss = -np.mean(t*np.log(Y) + (1-t) * np.log(1-Y))
     
     return loss,(dV,dW,Ev,Ew)

def predict(x,t,V,W,bv,bw):
     
     A = np.dot(x,V) + bv  
     B = np.dot(np.tanh(A),W) + bw
     return (sigmoid(B)>0.5).astype(int)

#creating layers
V = np.random.normal(scale = 0.1,size = (n_in,n_hidden))
W = np.random.normal(scale = 0.1,size = (n_hidden,n_out)) 

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

#generate data
X = np.random.binomial(1,0.5,(n_sample,n_in))
T = X ^ 1

#training time
for epoch in range(100):
    err = []
    upd = [0]*len(params)
    
    t0 = time.clock()
    #for each point update weights
    for i in range(X.shape[0]):
       loss,grad = train(X[i],T[i],*params)
       #update loss
       for j in range(len(params)):
          params[j] -=upd[j]

       for j in range(len(params)):
          upd[j] = learning_rate*grad[j] + momentum * upd[j]
   
       err.append(loss)

       print('Epoch: %d, Loss: %.8,Time: %.4fs' % ( epoch,np.mean(err),time.clock()-t0))
   
x = np.random.binomial(1,0.5,n_in)
print('XOR Prediction')
print(x)
print(predict(x,*params))
 