
# coding: utf-8

# In[ ]:


from math import sqrt
class stat:
    def isConfusionMatrix(*args, **kwargs):
        if len(kwargs)==4:
            for key, value in kwargs.items():
                if key not in ["tp","fp","tn","fn"]:
                    raise ValueError("provided key \""+str(key)+"\" not in [\"tp\",\"fp\",\"tn\",\"fn\"] ")
                if value.__class__!=int:
                    raise ValueError("confusion matrix parameters must be integers")
            return kwargs["tp"],kwargs["fp"],kwargs["tn"],kwargs["fn"]
        if len(args)==4:
            for arg in args:
                if arg.__class__!=int:
                    raise ValueError("confusion matrix parameters must be integers")
            return args
        else:
            raise ValueError(str(len(args)) + " arguments were given, a confusion matrix has 4")
        return None

    def TPR(*args,**kwargs):
        """Returns the true positive rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return tp / (tp + fn)

    def TNR(*args,**kwargs):
        """Returns the true negative rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return tn / (tn + fp)

    def PPV(*args,**kwargs):
        """Returns the positive predictive value"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return tp / (tp + fp)

    def NPV(*args,**kwargs):
        """Returns the negative predictive value"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return tn / (tn + fn)

    def FNR(*args,**kwargs):
        """Returns the false negative rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return fn / (fn + tp)

    def FPR(*args,**kwargs):
        """Returns the false positive rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return fp / (fp + tn)

    def FDR(*args,**kwargs):
        """Returns the false discovery rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return fp / (fp + tp)

    def FOR(*args,**kwargs):
        """Returns the false omission rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return fn / (fn + tn)

    def ACC(*args,**kwargs):
        """Returns the accuracy"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (tp + tn) / (tp + tn + fp + fn)

    def F1(*args,**kwargs):
        """Returns the harmonic mean of precision and score"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (2 * tp) / (2 * tp + fp + fn) 

    def MCC(*args,**kwargs):
        """Returns the Matthews correlation coefficient"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (tp * tn - fp * fn) / sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) )

    def BM(*args,**kwargs):
        """Returns the Bookmaker Informedness"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (tp/(tp + fn)) + (tn/(tn + fp)) - 1

    def MK(*args,**kwargs):
        """Returns the Markedness"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (tp/(tp + fp)) + (tn/(tn + fn)) - 1

    def ERROR(*args,**kwargs):
        """Returns the Error Rate"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return (1 - (tp + tn) / (tp + tn + fp + fn))

    def makeStat(function,tp,fp,tn,fn):
        """Returns a user-defined lambda function"""
        tp,fp,tn,fn = stat.isConfusionMatrix(*args,**kwargs)
        return function(tp,fp,tn,fn)
