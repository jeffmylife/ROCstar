import ConfusionMatrix
import ConfusionVisual
import ConfusionStatistics
from numpy import *
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


class ROCstar:
    import matplotlib.pyplot as plt
    
    def fluff(x1, y1,rand_interpol=True):
    	og = sorted(list(set([(x,y) for x,y in zip(x1,y1)])))
    	tol = 0.001
    	rounder = abs(int(log10(tol)))
    	d = dict()
    	for i,j in og:
        	if i in d:
            	d[i].append(j)
        	else:
            	d[i]=[j]
    	dout=dict()
                
    	left = ( 0.0, max(d[0.0]) )
    	d.pop(left[0])
    	l=[(0.0,0.0),left]
    	for x in d:
        	right = ( x, min(d[x]) )  
        	'''Equation of the line:  x | (y2-y1) / (x2-x1) * (x-x0) + y1'''
        	m = (right[1]-left[1]) / (right[0]-left[0]) if rand_interpol else 1
        	line = lambda x: round( m * (x-left[0]) + left[1], rounder )
        	for t in arange(left[0],right[0]+tol,tol): 
            	y_t = line(t)
            	if  round(t,rounder)>1 :
                	break
            	if y_t>1:
                	dout[round(t,rounder)] = 1
            	else:
                	dout[round(t,rounder)] = y_t
        	left = ( x, max(d[x]) )

    	x,y = list(dout),list(dout.values())
    	return [0]+x, [0]+y
    
    @staticmethod
    def removePrivateMethods(methodDict):
        for key in list(methodDict): 
            if key.startswith("__"):
                del methodDict[key] 
        return methodDict
    __removePrivateMethods__ = removePrivateMethods.__func__


    @staticmethod
    def binarize(lst):
        labels = set(lst)
        if len(labels)>2:
            raise ValueError("Non-binary data not allowed")
        return label_binarize(lst,list(labels))[:,0]  
    
    STATS = __removePrivateMethods__(dict(ConfusionStatistics.__dict__))
    MATRIX = __removePrivateMethods__(dict(ConfusionMatrix.ConfusionMatrix.__dict__))
    VISUAL = __removePrivateMethods__(dict(ConfusionVisual.ConfusionVisual.__dict__))
    
    
    def __init__(self,actual = [], preds: dict ={}):
        self.setActual(actual)
        self.addPreds(preds)
        self.METHODS = dict(pair for d in [ROCstar.STATS,ROCstar.MATRIX] for pair in d.items())
        self.CONFS = dict()
        self.__update__()
        
    def __update__(self):
        '''Updates self.CONFS to new ACTUAL or PREDS values'''
        for model in self.PREDS:
            assert len(self.ACTUAL)==len(self.PREDS[model]),"Predicted and actual data must be of equal length"
            if len(set(self.PREDS[model]))>2:
                self.CONFS.update( {model: ConfusionMatrix.ConfusionMatrix(actual=self.ACTUAL,pred=self.PREDS[model],bins=1000).getConfs()} )
            else:
                self.CONFS.update( {model: ConfusionMatrix.ConfusionMatrix(actual=self.ACTUAL,pred=self.PREDS[model])} )
                
    def setActual(self,actual):
        self.ACTUAL = ROCstar.binarize(actual)  
        try:
            if len(self.PREDS)>0:
                self.__update__()
        except AttributeError: 
            return 
        
    def addPreds(self,preds: dict):
        assert len(preds)>=1
        try:
            if len(self.PREDS)>0:
                    self.PREDS.update(preds)
        except AttributeError:
            self.PREDS=preds
        try:
            if len(self.ACTUAL)>0:
                self.__update__()
        except AttributeError:
            pass 
        processPreds = lambda pred_dict: {model:array(list(map(float,self.PREDS[model]))) for model in self.PREDS}
        self.PREDS = processPreds(preds)
        
    def apply(self,function: str) -> dict:
        if list(self.CONFS.items())[0][1].__class__==dict:
            return { model: array([self.STATS[function](*self.CONFS[model][thresh]) for thresh in self.CONFS[model]]) for model in self.CONFS }
        else:
            return {model : self.STATS[function](*self.CONFS[model]) for model in self.CONFS}
                
    def plot(self,x_axis: str ="FPR",y_axis: str ="TPR",**kwargs):
        x_data = self.apply(x_axis)
        y_data = self.apply(y_axis)
        vis = ConfusionVisual.ConfusionVisual()
        for model in self.CONFS:
            vis.add(x=x_data[model],x_title=x_axis,y=y_data[model],y_title=y_axis,title=model) 
        return vis.plot(**kwargs),vis
    
    def ROC(self,**kwargs):
        vis = self.plot(x_axis ="FPR",y_axis ="TPR",**kwargs)
        return vis 
      
     
       
