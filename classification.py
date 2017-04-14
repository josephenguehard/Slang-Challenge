# -*-coding:Latin-1 -*
import numpy as np

class Classifier:
    
    def __init__(self, C=None):
        self.C = C
        if self.C is not None: self.C = float(self.C)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = sp.sparse.csc_matrix.dot(X[i], X[j].T).toarray()
        
        # Lagrange Multipliers
        P = sp.sparse.csc_matrix(np.outer(y,y) * K)
        q = sp.sparse.csc_matrix(np.ones(n_samples) * -1)
        A = sp.sparse.csc_matrix(y, (1,n_samples))
        b = sp.sparse.csc_matrix(0.0)

        if self.C is None:
            G = sp.sparse.csc_matrix(np.diag(np.ones(n_samples) * -1))
            h = sp.sparse.csc_matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = sp.sparse.csc_matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = sp.sparse.csc_matrix(np.hstack((tmp1, tmp2)))
        
        # Function to minimize in order to solve the dual problem
        def fun_to_minimize(x):
            z = (1/2)*np.dot(x,P.dot(x)) + q.dot(x)
            return(z)
        
        cons = ({'type': 'eq', 'fun': lambda x: A.dot(x)},
                {'type': 'ineq', 'fun': lambda x: np.asarray((h.todense() - G.dot(x))).reshape(-1)})
        bnds = ((0, None),)*n_samples
        
        #Use the scipy library to find the minimum
        res = minimize(fun_to_minimize, (0,)*n_samples, method='SLSQP', bounds=bnds, constraints=cons)
        self.res = res
        a = res.x

    # Support vectors have non zero lagrange multipliers
        sv = a > 1e-13
        ind = np.arange(len(a))
        self.ind = ind
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def project(self, X):
        return X.dot((self.w).T) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
    def score(self, X, y):
        y_pred = np.squeeze(np.asarray(svm.predict(X)))
        return np.mean(y_pred == y)