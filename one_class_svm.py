from sklearn.svm import OneClassSVM

def train_one_class_svm(X):
    model = OneClassSVM(kernel='rbf', nu=0.05)
    preds = model.fit_predict(X)
    return (preds == -1).astype(int)
