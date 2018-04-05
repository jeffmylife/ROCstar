
from numpy import nan,arange 

class ConfusionMatrix: 
    
    def calcConfusion(pred,actual,threshold=.5): 
        actual_vals = set(actual) 
        if len(actual_vals)!=2: 
            raise ValueError("non-binary values not allowed ") 
                  
        tp=fp=tn=fn=0 
        for actu_val,pred_val in zip(actual,pred): 
                if actu_val==1:      
                    if pred_val>threshold: 
                        tp+=1 
                    else: 
                        fn+=1 
                else:                
                    if pred_val<threshold:  
                        tn+=1 
                    else:  
                        fp+=1 
        return ConfusionMatrix(tp=tp,fp=fp,tn=tn,fn=fn) 
     
    def __init__(self,pred=[],actual=[],tp=0,fp=0,tn=0,fn=0,bins=100): 
        assert len(pred)==len(actual)
        
        self.tp = tp 
        self.fp = fp
        self.tn = tn 
        self.fn = fn 
        self.confs = None
        
        '''Calculate all confusion matrices'''
        if len(pred)!=0 : 
            low = min(pred)
            high = max(pred)
            step = (low+high)/bins
            
            self.tp = self.fp = self.tn = self.fn = None
            confs = {}
            for thresh in arange(high,low-step,-step):
                conf = ConfusionMatrix.calcConfusion(pred,actual,threshold=thresh) 
                confs.update({thresh:conf})
            self.confs=confs
            
    
    def getConfs(self):
        confs={}
        for thresh in self.confs:
            confs[thresh]= tuple([i for i in self.confs[thresh]])
        return confs
    
    def __iter__(self):         
        return (x for x in [self.tp,self.fp,self.tn,self.fn]) 
         