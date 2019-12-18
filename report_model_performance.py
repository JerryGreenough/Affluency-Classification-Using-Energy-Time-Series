def report_model_performance(model, X, y, target_names):
    
    from collections import Counter
    from sklearn.metrics import classification_report
    
    # For a pre-trained model ('model'), assess a model's performance on a test set of 
    # features, X together with a set of observed classifications y.   
    # Return a 4-tuple of calculations related to model perfomance as follows:
    
    # 1. Overall accuracy (ratio of correct predctions to number of tests)
    # 2. The 'recall' of the 'Affluent' class.
    # 3. The 'precision' of the 'Affluent' class.
    # 4. The 'f1-score' of the 'Affluent' class.
    
    predictions = model.predict(X)

    cc = Counter(y-predictions)
    acc = cc[0] / sum(cc.values())
    rec = classification_report(y, predictions, target_names=target_names, output_dict=True)['Affluent']['recall']
    prc = classification_report(y, predictions, target_names=target_names, output_dict=True)['Affluent']['precision']
    f1s = classification_report(y, predictions, target_names=target_names, output_dict=True)['Affluent']['f1-score']
    
    return acc, rec, prc, f1s   


def report_model(model, X, y, target_names):

    from collections import Counter
    from sklearn.metrics import classification_report
    
    predictions = model.predict(X)
    
    cc = Counter(y-predictions)
    print("Number of accurate predictions = ", cc[0])
    rat = cc[0] / sum(cc.values())
    
    print("Ratio of accurate predictions = ", round(rat,2), '\n')
    
    print(classification_report(y, predictions, target_names=target_names))
    print(classification_report(y, predictions, target_names=target_names, output_dict=True)['Affluent']['recall'])