import matplotlib.pyplot as plt
from sklearn import metrics
from math import log

class ConfusionVisual:
    class Axes:
        def __init__(self,x,x_title,y,y_title,title):
            self.x = x
            self.y = y
            self.y_title = y_title
            self.x_title = x_title
            self.title = title 

        def invertAxes(self):
            tmp=self.y
            self.y=self.x
            self.x=tmp
            
        def axesToLog(self,base=10):
            self.x = log(self.x,base)
            self.y = log(self.y,base)
            
        def AUC(self,**kwargs):
            return metrics.auc(self.x,self.y,**kwargs)
                
    def __init__(self):
        self.PLOTS = []
        
    def add(self,x,x_title,y,y_title,title):
        self.PLOTS.append( ConfusionVisual.Axes(x, x_title, y, y_title, title) )
    
    def plot(self,show=True,show_legend=True,show_random=True, title="ROC",figsize=5,**kwargs):
        fig, ax = plt.subplots(figsize=(figsize, figsize)) 
        
        for axes in self.PLOTS:
            ax.plot(axes.x,axes.y,label=axes.title)
            ax.scatter(axes.x,axes.y)
            
        plt.title(title)
        plt.xlabel(axes.x_title)
        plt.ylabel(axes.y_title)
        
        lim_offset = 0.01 if "lim_offset" not in kwargs else kwargs["lim_offset"]
        plt.xlim(0-lim_offset,1+lim_offset)
        plt.ylim(0-lim_offset,1+lim_offset)
        
        if show_legend: plt.legend(loc="lower right" if  "legend_loc" not in kwargs else kwargs["legend_loc"])    
        diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        
        return plt
    
    def AUC(self):
        out = dict()
        for axes in self.PLOTS:
            out[axes.title] = axes.AUC()
        return out
