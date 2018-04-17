
# coding: utf-8

# In[ ]:


from math import sqrt
def isConfusionMatrix(*args, **kwargs):
    if len(kwargs)==4:
        for key, value in kwargs.items():
            if key not in ["tp","fp","tn","fn"]:
                raise ValueError("Provided key \""+str(key)+"\" not in [\"tp\",\"fp\",\"tn\",\"fn\"] ")
            if int(value)!=value:
                raise ValueError("Confusion matrix parameters must be integers not "+str(value.__class__))
        return kwargs["tp"],kwargs["fp"],kwargs["tn"],kwargs["fn"]
    if len(args)==4:
        for arg in args:
            if int(arg)!=arg:
                raise ValueError("Confusion matrix parameters must be integers not "+str(arg.__class__))
        return args
    else:
        raise ValueError(str(len(args)) + " arguments were given, a confusion matrix allows only 4")

def TPR(*args,**kwargs):
    """Returns the true positive rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return tp / (tp + fn) if (tp + fn)!=0 else 0

def TNR(*args,**kwargs):
    """Returns the true negative rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return tn / (tn + fp) if (tn + fp)!=0 else 0

def PPV(*args,**kwargs):
    """Returns the positive predictive value"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return tp / (tp + fp) if (tp + fp)!=0 else 0

def NPV(*args,**kwargs):
    """Returns the negative predictive value"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return tn / (tn + fn) if (tn + fn)!=0 else 0

def FNR(*args,**kwargs):
    """Returns the false negative rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return fn / (fn + tp) if (fn + tp)!=0 else 0

def FPR(*args,**kwargs):
    """Returns the false positive rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return fp / (fp + tn) if (fp + tn)!=0 else 0

def FDR(*args,**kwargs):
    """Returns the false discovery rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return fp / (fp + tp) if (fp + tp)!=0 else 0

def FOR(*args,**kwargs):
    """Returns the false omission rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return fn / (fn + tn) if (fn + tn)!=0 else 0

def ACC(*args,**kwargs):
    """Returns the accuracy"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn)!=0 else 0

def F1(*args,**kwargs):
    """Returns the harmonic mean of precision and score"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn)!=0 else 0

def MCC(*args,**kwargs):
    """Returns the Matthews correlation coefficient"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / sqrt( denom ) if denom!=0 else 0

def BM(*args,**kwargs):
    """Returns the Bookmaker Informedness"""
    c = isConfusionMatrix(*args,**kwargs)
    return TPR(*c) + TNR(*c) - 1 

def MK(*args,**kwargs):
    """Returns the Markedness"""
    c = isConfusionMatrix(*args,**kwargs)
    return PPV(*c) + NPV(*c) -1

def ERROR(*args,**kwargs):
    """Returns the Error Rate"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return 1 - (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn)!=0 else 0

def STAT(*args,function = lambda tp,fp,tn,fn: 0,**kwargs):
    """Returns the function of a Confusion Matrix"""
    tp,fp,tn,fn = isConfusionMatrix(*args,**kwargs)
    return function(tp,fp,tn,fn)

def RIE(actives, scores, alpha):
    """Returns Robust Initial Enhancement"""
    N=len(actives)
    n=sum(scores)
    x = np.array(actives)
    e = np.exp(-alpha*x)
    summation = sum(e)
    RIE = ((1/n)*summation)/((1/N)*((1-np.exp(-alpha))/np.exp(alpha/N-1)))
    return RIE

def BEDROC(actives, scores, alpha):
    """Returns Boltzmann-Enhanced Discrimination of ROC"""
    N=len(actives)
    n=sum(scores)
    RIE=RIE(actives, scores, alpha)
    BEDROC = (RIE*((1/N)*np.sinh(alpha/2))/(np.cosh(alpha/2)-np.cosh(alpha/2-alpha*(n/N))))+1/(1-np.exp(alpha*((N-n)/N)))
    return BEDROC
