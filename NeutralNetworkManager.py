from dependency import *
from MathUtils import *
from PaintUtils import *

kernel = ConstantKernel(constant_value=1) * RBF(length_scale=1)


class NeutralNetworkManager(object):
    def __init__(self, sample_num, dim, func, min, max) -> None:
        # sampling by Latin Hypercube
        self.archive = []
        self.dim = dim
        self.func = func
        self.min = min
        self.max = max
        self.model = GaussianProcessRegressor( kernel,
                               alpha=0.1,
                               n_restarts_optimizer=5,
                               normalize_y=True)
        # self.std = StandardScaler()
        train_data = LatinHypercube(min, max, sample_num, dim)
        # train_data_std = self.std.fit_transform(train_data)
        train_target = calculateAll(train_data, func)
        self.model.fit(train_data, train_target)
        pass

    def update(self, X):
        # train_data_std = self.std.transform(X)
        train_target = calculateAll(X, self.func)
        self.model.fit(X, train_target)

    def predict(self, X):
        # X_std = self.std.transform(X)
        return self.model.predict(X)
    
    def plot(self, point_num):
        if self.dim!=2:
            return
        X = []
        for i in range(point_num):
            x = []
            for j in range(self.dim):
                x.append(random.uniform(self.min, self.max))
            X.append(np.array(x, dtype=float))
        X = np.array(X)
        x_dim = X[:,0]
        y_dim = X[:,1]
        # z = self.predict(X)
        z = calculateAll(X, func=self.func)
        paint3D(x_dim, y_dim, z)

