def run_grid(knl, X_train, y_train, X_test, y_test, X_entire, y_entire):
    
#   Given a kernel function specification, knl, 
#   run a parameter tuning analysis using an SVM for a training set
#   X_train, y_train.
#   3-fold cross-validation is used for each parameter combination.

#   Returns a dictionary containing the model performance
#   results for both the test data set as well as the entire data set.

    from report_model_performance import report_model_performance
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC 
	
    target_names = ["Non-Affluent", "Affluent"]

    param_grid = {'C': [1, 2, 3, 4, 5, 10, 20, 30, 50, 100] }
    if knl == 'rbf':
        param_grid['gamma'] = [10 **(-4 + 0.25*j) for j in range(12)]
  
    model = SVC(kernel = knl)

#   Note the n_jobs = -1 requests that the job be run on all available local
#   processors.

    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=-1, cv=3)
    grid.fit(X_train, y_train)
    
    model = SVC(kernel = knl, max_iter = 1000000, **grid.best_params_)

    model.fit(X_train, y_train)
    
    res = {}
    
    x1, x2, x3, x4 = report_model_performance(model, X_test, y_test, target_names)
    res['test_acc'] = x1
    res['test_rec'] = x2
    res['test_prc'] = x3
    res['test_f1' ] = x4
    
    x1, x2, x3, x4 = report_model_performance(model, X_entire, y_entire, target_names)
    res['entire_acc'] = x1
    res['entire_rec'] = x2
    res['entire_prc'] = x3
    res['entire_f1' ] = x4
    
    res['C'] = grid.best_params_['C']
    
    if knl == 'rbf':
        res['gamma'] = grid.best_params_['gamma']
    
    return res