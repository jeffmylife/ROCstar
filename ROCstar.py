import ConfusionMatrix
import ConfusionVisual
import ConfusionStatistics
from numpy import *
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
#%matplotlib inline


class ROCstar:
    import matplotlib.pyplot as plt
    
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
    
    STATS = __removePrivateMethods__(dict(ConfusionStatistics.stat.__dict__))
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
      
     
        
    
'''   
r = ROCstar( (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,0,0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1),{"pred1":(0.584624535745324, 0.6076533871509906, 0.5652611635439775, 0.5354354483589192, 0.5614486550293746, 0.9551569363906147, 0.94191370849109028, 0.8739227945870757, 0.35415203900622083, 0.2007607753715377, 0.46104766520661145, 0.8452862420667127, 0.3679877712602706, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, .999,0,0,.5,.5,1,0.6,1.0,.6,.7,1,0,0,1,.9,0,1,.5,0,.1,1,0.7,1,.1,0,1,1,0,1),"pred2":(0.8267628604678204, 0.396587273899628, 0.7796929785329103, 0.7735648848402202, 0.8768144491877597, 0.5680912422826304, 0.805368149664956, 0.40569857780971774, 0.789423761169201, 0.4543712390728748, 0.6306063372510953, 0.322895054804039, 0.9885000568328947, 0.7632365939616852, 0.7227807323167601, 0.23190179168212866, 0.3247882445170872, 0.5864080080315711, 0.5299825143473527, 0.18884690237163226, 0.3054881142694532, 0.7689708861708133, 0.4935897534639603, 0.31238441130545147, 0.7797401378218456, 0.4504988270061461, 0.9170965683894869, 0.6083621655027964, 0.774934349651232,0,0,0,.8,.5,1,0.8,1.0,.6,.7,1,1,0,1,1,0,.1,.5,0,.1,.1,0.7,1,.1,0,.1,.1,.8,.1)} )
plt, vis = r.plot()
vis.AUC()
'''