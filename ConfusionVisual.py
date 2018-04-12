import matplotlib.pyplot as plt
from sklearn import metrics

class ConfusionVisual:
    class Axes:
        def __init__(self,x,x_title,y,y_title,title):
            self.x = x
            self.y = y
            self.y_title = y_title
            self.x_title = x_title
            self.title = title 

        def invertAxes(self):
            tmp=self.Y
            self.Y=self.X
            self.X=tmp
            
        def AUC(x,y,**kwargs):
            return auc(x,y,**kwargs)
                
    def __init__(self):
        self.PLOTS = []
        
    def add(self,x,x_title,y,y_title,title):
        self.PLOTS.append(ConfusionVisual.Axes(x,x_title,y,y_title,title))
    
    def plot(self):
        for axes in self.PLOTS:
            plt.plot(axes.x,axes.y)
        
        if __name__=='__main__':
            plt.show()
        else:
            return plt
        
        


        
        



'''
import matplotlib.pyplot as plt
import numpy as np
from mpldatacursor import DataCursor
from mpldatacursor import HighlightingDataCursor
%matplotlib notebook

fig,ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(fpr,tpr,color="green")
ax.plot(fpr,fpr,color="gray",linestyle="--")
ax.set_xlim(0-.01,1)
ax.set_ylim(0,1+.01)
ax.fill_betweenx(tpr,fpr,tpr,color="green",alpha=.2)
ax.fill_between(fpr,0,fpr,color="gray",alpha=.2)
rocTitle = 'ROC Plot Title'
ax.set_title(rocTitle)
xLabel = 'x axis title'
ax.set_xlabel(xLabel)
yLabel = 'y axis title'
ax.set_ylabel(yLabel)
ax.legend(ax.plot(fpr,tpr,color="green"), "roc1", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

DataCursor(ax.plot(fpr,tpr,color="green"), display='single', draggable=True, hide_button=1)
HighlightingDataCursor(ax.plot(fpr,tpr,color="green"), highlight_color='darkgreen')
'''





