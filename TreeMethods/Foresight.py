'''
Class to determine weights when performing feature selection for e.g. when growing decision tree s

Authors: Suryarao Bethapudi: ep14btech11008@iith.ac.in

Ben Hoyle benhoyle1212@gmail.com




'''
import numpy as np

#use SKlearn, which accepts discreet and continuous variables now (as of 0.19.0)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

__all__=['fit', 'select_n_features','lazy_select']

EPS = np.finfo(float).eps


class Foresight(object):
    """Good class"""
    def __init__(self, X, y, k=None, aux_data=None, number_rows=50000, ydiscrete=None, verbose=False):
        """Initializing function"""
        self.n, self.d = X.shape
        self.X = X.copy() # deep copy
        self.y = y.copy() # deep copy
        '''
        doubt: Should I remove one of `aux_data` or `X` since at any point we're
        using only one of them?
        '''
        #number of data to use when estimating MIs [randomly selected]
        """
        doubt: Why do we have to restrict with the number of rows, sir?
        To speed up the process. Let's say we have 2M science smaple objects, and we didn't randomly sample from them. The MI determination would take *far* too long. Recall, we are just trying to best-guess what are the good features, we don't need exactness.
        """
        self.number_rows = number_rows
        self.rnd_ind = np.arange(self.n)

        #use this to show debugging output to screen.
        self.verbose = verbose

        if self.n > self.number_rows:
            self.rnd_ind = np.random.choice(self.n, size=self.number_rows, replace=False)

        # There will be two tables
        #set mi_features to NaN, so that we know if we have tried
        #to calculate MI(x1,x2) already
        self.mi_features = np.zeros((self.d, self.d)) - np.nan
        self.mi_features_y = np.zeros(self.d)

        self.k = k
        if k is None:
            self.k = int(np.sqrt(self.d))

        self.aux_data = aux_data
        if aux_data is not None:
            assert(aux_data.shape == X.shape)
            '''
            doubt:
            shouldn't we just check if the dimensions of `X` and `aux_data` match?
            number of rows of both the matrices can be different right?
            -- okay I'm fine with this. Yes the # of rows will almost certainly be different.
            '''
            self.aux_data = aux_data.copy()
            self.aux_n, self.aux_d = self.aux_data.shape
            self.ind_aux = np.arange(len(self.aux_data[:, 0]))
            if len(ind_aux) > self.number_rows:
                self.ind_aux = np.random.choice(len(ind_aux), size=self.number_rows, replace=False)

        if ydiscrete is None:
            self.ydiscrete = self.is_array_discreet(y)
        else:
            # Boolean variable to see if y is discrete
            self.ydiscrete = ydiscrete

        # Boolean variable to check if fit has been called
        self.fitdone = False

    def is_array_discreet(self, arr):
        """best guess if array is continuous of discreet"""
        return (len(np.unique(arr)) < len(arr)*0.1) or np.all([i in np.arange(np.amin(arr), np.amax(arr)) for i in np.unique(arr)])

    def MI(self, x1, x2):
        """ uses the appropiate MI, depending on type of x2 array
        """
        if self.ydiscrete:
            return mutual_info_classif(x1.reshape(-1, 1), x2)
        return mutual_info_regression(x1.reshape(-1, 1), x2)

    # main method
    def fit(self):
        """This function does the magic"""

        for i in xrange(self.d):
            self.mi_features_y[i] = self.MI(self.X[self.rnd_ind, i], self.y[self.rnd_ind])

        #ensure there are no negative MIs
        self.mi_features_y[self.mi_features_y < 0] = 0

        self.fitdone = True


    def select_n_features(self, n, nn=5, met='random'):
        """
        Calling the optimized function `select_n_features`

        -- BH: Note, we can be clever, and there is no need to perform a normal NF*NF fit at all.
        """

        '''
        TODO: What if some features in `X` are also discrete?
        '''
        #ensure there are no negative MIs
        # ABOVE code comes from `fit`
        # BELOW code comes from `select_n_features`
        wgts = self.mi_features_y / np.sum(self.mi_features_y)
        # simple logic to prevent ValueError while drawing samples
        sample_size = nn*n
        if sample_size > self.d:
            sample_size = self.d

        if sample_size > np.sum(wgts > 0):
            sample_size = np.sum(wgts > 0)

        if self.verbose:
            print 'Diag1: ', wgts.shape
            print 'Diag2: ', self.d
            print 'Diag3: ', sample_size
            print 'Diag4: ', wgts

        get_a_heap_of_features = np.random.choice(self.d, size=sample_size, replace=False, p=wgts)
        get_1_feature = np.random.choice(self.d, size=1, replace=False, p=wgts)[0]

        """
        We have a heap of features sampled using weights provided by MI.
        Our job now is to select `n` best features from the heap of features.
        """
        if met.upper() == "RANDOM":
            final_features = [get_1_feature]
            for i in range(n-1):
                # reduce size of get_a_heap_of_features by removing all final_features from it.
                #List comprehension e.g. [i for i in range] is *insanely* fast in python
                get_a_heap_of_features = np.array([ft for ft in get_a_heap_of_features if ft not in final_features])

                #determine MI between final_features[-1] and get_a_heap_of_features
                #first do any of the MI(F1,F2) need to be computed?
                nan_mif1f2 = np.isnan(self.mi_features[final_features[-1], get_a_heap_of_features])

                #this loop will not be executed if all nan_mif1f2 are False, so that
                #mi_features[final_features[-1], get_a_heap_of_features] have been calculated aready
                for j, ft2 in enumerate(get_a_heap_of_features[nan_mif1f2]):
                    if self.aux_data is not None:
                        tmi = mutual_info_regression(self.aux_data[self.ind_aux, i].reshape(-1, 1),
                                                     self.aux_data[self.ind_aux, j].reshape(-1, 1))
                    else:
                        tmi = mutual_info_regression(self.X[self.rnd_ind, i].reshape(-1, 1),
                                                     self.X[self.rnd_ind, j].reshape(-1, 1))
                    if tmi < 0:
                        tmi = 0.0
                    self.mi_features[final_features[-1], ft2] = self.mi_features[final_features[-1], ft2] = tmi

                #get weights [inverse MI between F1 and F2s]
                wgts = 1. / self.mi_features[final_features[-1], get_a_heap_of_features]
                wgts /= np.sum(wgts)

                #add a randomly selected feature from get_a_heap_of_features to final_features
                final_features.append(np.random.choice(get_a_heap_of_features, size=1, p=wgts)[0])

            return np.sort(final_features)
        else:
            # not yet implemented
            ValueError('No other method is implemented yet!')
