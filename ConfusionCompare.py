
# coding: utf-8

# In[ ]:


#This is the ConfusionCompare class.
# Stories related to this class include :
#1) F4 : Comprehensive overview of classifier(s) performance
class ConfusionCompare:

    def percentDifference(auc1, auc2):
        """Returns the percent difference between two AUC's"""
        return abs(auc1-auc2) / (auc1+auc2) * 2
