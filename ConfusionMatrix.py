
# coding: utf-8

# In[6]:


from numpy import nan

class ConfusionMatrix:
    
    def __init__(self,pred=[],actual=[],tp=0,fp=0,tn=0,fn=0):
        if len(pred)!=0:
            pred_vals,actual_vals = set(pred), set(actual)
            
            if len(pred_vals)!=2:
                raise ValueError("Non-Binary Values")

            c1,c2 = sorted(list(pred_vals))     #classifications like 0 or 1, cat or dog, etc. 
            
            for prd,act in zip(pred,actual):    #note that pred and actual do not have to be equal length
                if prd==act:                    #predicted correctly
                    if prd==c2:                 
                        tp+=1              
                    else:                       
                        tn+=1
                else:
                    if prd==c2:                 
                        fp+=1              
                    else:                       
                        fn+=1 
        else:
            if (tp+tn+fn+fp).__class__!=int:
                raise ValueError("")
        
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
    

    def __iter__(self):        
        return (x for x in [self.tp,self.fp,self.tn,self.fn])
        
        
    #also could use (x > 0.5) where x is numpy array 
    def fromThresh(pred,actual,thresh):
        actual_vals = set(actual)
        if len(actual_vals)!=2:
            raise ValueError("Non-Binary Values")
        
        true_val=sorted(list(actual_vals))[1]
        
        tp=fp=tn=fn=0
        for aval,pval in zip(actual,pred):
                if aval==true_val:  #true
                    if pval>thresh: #positive
                        tp+=1
                    else:
                        fn+=1
                else:               #false
                    if pval<thresh: 
                        tn+=1
                    else: 
                        fp+=1
        return ConfusionMatrix(tp=tp,fp=fp,tn=tn,fn=fn)


    
  
      

