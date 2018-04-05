import ConfusionStatistics 
import ConfusionMatrix
import ConfusionVisual
import numpy  
from sklearn.preprocessing import label_binarize


def removePrivateMethods(methodDict):
    for key in list(methodDict): 
        if key.startswith("__"):
            del methodDict[key] 
    return methodDict

def binarize(lst):
    labels = set(lst)
    if len(labels)>2:
        raise ValueError("Non-binary data not allowed")
    return label_binarize(lst,list(labels))[:,0]




class ROCstar:
    STATS = removePrivateMethods(dict(ConfusionStatistics.stat.__dict__))
    MATRIX = removePrivateMethods(dict(ConfusionMatrix.ConfusionMatrix.__dict__))
    VISUAL = removePrivateMethods(dict(ConfusionVisual.ConfusionVisual.__dict__))
    
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
                self.CONFS.update( {model: ConfusionMatrix.ConfusionMatrix(actual=self.ACTUAL,pred=self.PREDS[model],bins=10).getConfs()} )
            else:
                self.CONFS.update( {model: ConfusionMatrix.ConfusionMatrix(actual=self.ACTUAL,pred=self.PREDS[model])} )
                
    def setActual(self,actual):
        self.ACTUAL = binarize(actual)  
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
        processPreds = lambda pred_dict: {model:numpy.array(list(map(float,self.PREDS[model]))) for model in self.PREDS}
        self.PREDS = processPreds(preds)
        
    
    def apply(self,function: str) -> dict:
        if list(self.CONFS.items())[0][1].__class__==dict:
            return { model: [self.STATS[function](*self.CONFS[model][thresh]) for thresh in self.CONFS[model]] for model in self.CONFS }
        else:
            return {model : self.STATS[function](*self.CONFS[model]) for model in self.CONFS}
            
        
    
    def plot(self,x_axis: str ="FPR",y_axis: str ="TPR"):
        x_data = self.apply(x_axis)
        y_data = self.apply(y_axis)
        vis = ConfusionVisual.ConfusionVisual()
        for model in self.CONFS:
            vis.add(x=x_data[model],x_title=x_axis,y=y_data[model],y_title=y_axis,title=model)
    
        vis.plot()
        plt.show()
        
    
'''    
r = ROCstar((1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1),{"pred1":(1,0.6,1.0,.6,.7,1,1,0,1,1,0,1,.5,0,.1,1,0.7,1,.1,0,1,1,0,1)})
r.plot()
''' 