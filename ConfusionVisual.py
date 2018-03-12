#provide data
true_data = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1] #0 or 1
pred_data = [0.3,0.6,0.7,0.1,0.4,0.3,0.5,1-.1,1-.2,1-.3,1-.5,1.3,1-.8,1-.6,1-.3] #[0,1]

#make ROC data
from numpy import arange
import numpy as np
def ROC(pred,actual,bins=100):
    low = min(pred)
    high = max(pred)
    step = (low+high)/bins
    confs = []
    fpr=[]
    tpr=[]

    for thresh in arange(low-step,high+step,step):
        conf = ConfusionMatrix.fromThresh(pred,actual,thresh) #be sure to get module from github
        confs.append(conf)
        tp,fp,tn,fn = conf
        fpr.append(fp/(fp+tn)) if fp+tn!=0 else fpr.append(0)
        tpr.append(tp/(tp+fn)) if tp+fn!=0 else tpr.append(0)

    fpr=np.array(fpr)[::-1] #reverses data
    tpr=np.array(tpr)[::-1]
    return fpr,tpr,confs
fpr,tpr,confs = ROC(pred_data,true_data,bins=50)

##plot ROC data
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

fig,ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(fpr,tpr,color="green")
ax.plot(fpr,fpr,color="gray",linestyle="--")
ax.set_xlim(0-.01,1)
ax.set_ylim(0,1+.01)
ax.fill_betweenx(tpr,fpr,tpr,color="green",alpha=.2)

ax.fill_between(fpr,0,fpr,color="gray",alpha=.2)




%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from mpldatacursor import datacursor


data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
lines = ax.plot(data)
ax.set_title('Click somewhere on a line')

datacursor(lines)

plt.show()
