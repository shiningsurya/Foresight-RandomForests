# Read dataset
class DataSet(object):
    '''
    One class to hold the entire dataset
    '''
    def __init__(self, X=None,y=None,full=None):
        if X == None and y == None:
            if full == None:
                raise ValueError('Wrong inputs given to DataSet')
            else:
                self.X = full[:,:-1]
                self.y = full[:,-1].reshape(-1,1) 
        elif full == None:
            if X == None and y == None:
                raise ValueError('Wrong inputs given to DataSet')
            else:
                self.X = X
                self.y = y
    def set_x_labels(self,xl):
        self.xl = xl
        if len(xl) != self.X.shape[1]:
            raise ValueError('Input x label length is not same as dataset shape')
    def set_y_labels(self,yl):
        self.yl = yl
        if len(yl) != self.y.shape[1]:
            raise ValueError('Input y label length is not same as dataset shape')


def SpamBase(path="./datasets/spambase"):
    '''
    Gives SpamBase dataset 
    '''
    import os
    import pandas as pd
    full = pd.read_csv(os.path.join(path,"spambase.data"))
    sb = DataSet(full=full.as_matrix())
    xl = ['make','address','all','3d','our','over','remove','internet']
    xl.append('order')
    xl = xl + ['mail', 'receive', 'will','people','report','addresses']
    xl = xl + ['free','business','email','you','credit','your','font']
    xl = xl + ['000','money','hp','hpl','geogre','650','lab','labs']
    xl = xl + ['telnet','857','data','415','85','technology','1999']
    xl = xl + ['parts','pm','direct','cs','meeting','original','project']
    xl = xl + ['re','edu','table','conference',';','(','[','!','$','#']
    xl = xl + ['c_average','c_longest','c_total']
    yl = ['spam']
    sb.set_x_labels(xl)
    sb.set_y_labels(yl)
    return sb 

def RCV1(path="./datasets"):
    '''
    Gives RCV dataset
    For now, using scikit-learn's RCV1
    '''
    import os
    from sklearn.datasets import fetch_rcv1
    return fetch_rcv1(data_home=path)
