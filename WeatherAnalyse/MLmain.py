import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from IPython.display import Image
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

#Input data
df = pd.read_csv ('weather.csv')

#Prepare data
df ['Rain'] = df ['RainTomorrow'] == 'Yes'
#FN1 = ['MinTemp', 'MaxTemp', 'Sunshine', 'WindGustSpeed', 'WindSpeed3pm', 'WindSpeed9am','Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'] 
FN2 = ['Temp3pm', 'Cloud3pm', 'Humidity3pm', 'WindSpeed3pm','Pressure3pm', 'Sunshine', 'WindGustSpeed', 'Rainfall']
#FN3 = ['Pressure3pm', 'WindGustSpeed', 'Sunshine', 'MaxTemp', 'Humidity3pm']
prediction = np.array ([10, 3, 30, 2, 1018, 2, 40, 10])
y = df ['Rain'].values
kf_splits = 5
def specificity_score (y_test, y_pred):
    p, r, f, s = precision_recall_fscore_support (y_test, y_pred)   
    return r [0]
    
def Accuracy_score (model, X, y):
    y_pred = model.predict_proba (X) [:, 1] > 0.75
    y_pred_roc = model.predict_proba (X)
    print ('Accuracy:', accuracy_score (y, y_pred))
    print ('ConfusionMatrix:', '\n', confusion_matrix (y, y_pred))
    print('Precision:', precision_score (y, y_pred))
    print ('Recall:', recall_score (y, y_pred))
    print ('F1score:', f1_score (y, y_pred))
    print ('Sensitivity:', recall_score (y, y_pred))
    print ('Specificity:', specificity_score (y, y_pred))
    print ('AUCscore:', roc_auc_score (y, y_pred_roc [:, 1])) 
    #ROC CURVE
    fpr, tpr, thresholds = roc_curve (y, y_pred_roc [:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.scatter (1 - specificity_score (y, y_pred), recall_score (y, y_pred), c = 'red')
    plt.xlim ([0.0, 1.0])                                              
    plt.ylim ([0.0, 1.0])
    plt.xlabel ('1 - specificity')
    plt.ylabel ('sensitivity')
    plt.show ()  
   
#NEURAL NETWORKS
def NN_model_TTS (FN, y, prediction):
    X = df [FN].values
    X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=3)
    model = MLPClassifier (max_iter=1000, hidden_layer_sizes=(150, 100), alpha=0.0001, solver='adam', random_state=10)
    model.fit (X_train, y_train)
    print ('RainTomorrow:', [[model.predict ([prediction])]])
    Accuracy_score (model, X, y)    
print ('Neural Networks model (TTS) #2:')
NN_model_TTS (FN2, y, prediction)

#RANDOM FOREST MODEL
def RF_model_GSCV (FN, y, prediction):
    X = df [FN].values
    n_estimators = list (range (1,101, 10))
    param_grid = {
        'n_estimators': n_estimators,
        }
    for criterion  in ['gini', 'entropy']:
        print ('Decision Tree - {}'.format (criterion))
        model = RandomForestClassifier (criterion = criterion)
        a = ['accuracy', 'recall', 'f1']
        for i in a:
            gs = GridSearchCV (model, param_grid, scoring =  i, cv=5)
            gs.fit (X,y)
            print ('Best params:', gs.best_params_)
            print ('Best', i, ':', gs.best_score_)
        #PlotAccuracy
            scores = gs.cv_results_ ['mean_test_score'] 
            plt.plot (n_estimators, scores)
        print ('RainTomorrow:', [[gs.predict ([prediction])]])
        plt.xlabel ('n_estimators')
        plt.ylabel ('Accuracy')
        plt.xlim (0, 100)
        plt.ylim (0.8, 1)
        plt.show ()
#print('Random Forest model (GSCV) #2:')
#RF_model_GSCV (FN2, y, prediction)

def RF_model_TTS (FN, y, prediction):
    X = df [FN].values
    X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=101)
    model = RandomForestClassifier (n_estimators=10, random_state=111)
    model.fit (X_train, y_train)
    print ('RainTomorrow:', [[model.predict ([prediction])]])
    first_row = X_test [0]
    Accuracy_score (model, X_test, y_test) 
    #ft_imp = pd.Series (model.feature_importances_, index=FN).sort_values (ascending=False)
#print ('Random Forest model (TTS) #2:')
#RF_model_TTS (FN2, y, prediction)


#DECISION TREE MODEL
def DT_model_GSCV (FN, y, prediction):
    X = df [FN].values
    max_depth = list (range (1,100,10))
    min_samples_leaf = list (range(1,100, 10))
    max_leaf_nodes = list (range (2, 100, 10))
    param_grid = {
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'max_leaf_nodes': max_leaf_nodes}
    for criterion  in ['gini', 'entropy']:
        print ('Decision Tree - {}'.format (criterion))
        model = DecisionTreeClassifier (criterion = criterion)
        a = ['accuracy', 'recall', 'f1']
        for i in a:
            gs = GridSearchCV (model, param_grid, scoring = i, cv = 5)
            gs.fit (X,y)
            #print ('RainTomorrow:', [[gs.predict ([prediction])]])
    	    #BestParams 
            print ('BestParams:', gs.best_params_)
            print ('Best', i, ':', gs.best_score_)
        print ('RainTomorrow:', [[gs.predict ([prediction])]])
#print ('Decision Tree model (GSCV) #2:')
#DT_model_GSCV (FN2, y, prediction)

def DT_model_Kf (FN, y, kf_split, prediction):
    X = df [FN].values
    for criterion in ['gini', 'entropy']:
        print ('Decision Tree - {}'.format (criterion))
        accuracy = []
        precision = []
        recall = []
        f1 = []
        sensitivity = []
        specificity = []
        roc_auc = []
        kf = KFold (n_splits = kf_splits, shuffle = True)
        c = 0
        for train_index, test_index in kf.split (X):
            c += 1
            X_train, X_test = X [train_index], X [test_index]
            y_train, y_test = y [train_index], y [test_index]
            model = DecisionTreeClassifier (criterion=criterion, max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
            model.fit (X_train, y_train)
            print ('RainTomorrow:', [[model.predict ([prediction])]])
            y_pred = model.predict (X_test)
            y_pred_roc = model.predict_proba (X_test)
            #Accuracy
            accuracy.append (accuracy_score (y_test, y_pred))
            precision.append (precision_score (y_test, y_pred))
            recall.append (recall_score (y_test, y_pred))
            f1.append (f1_score (y_test, y_pred))
            specificity.append (specificity_score (y_test, y_pred))
            roc_auc.append (roc_auc_score (y_test, y_pred_roc [:, 1]))
            if c == kf_splits:
                print ('Accuracy:', np.mean (accuracy))
                print ('Precision:', np.mean (precision))
                print ('Recall:', np.mean (recall))
                print ('F1score:', np.mean (f1))
                print ('Sensitivity:', np.mean (recall))  
                print ('Specificity:', np.mean (specificity)) 
                print ('AUCscore:', np.mean (roc_auc))
                #ROC CURVE
                fpr, tpr, thresholds = roc_curve (y_test, y_pred_roc [:,1])
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], linestyle='--')
                plt.scatter (1 - specificity_score (y_test, y_pred), recall_score (y_test, y_pred), c = 'red') 
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('1 - specificity')
                plt.ylabel('sensitivity')
                plt.show()
            dot_file = export_graphviz (model, feature_names = FN)
            graph = graphviz.Source (dot_file)
            graph.render (filename = 'DTM_Kf ' + criterion, format = 'png', cleanup = True)
#print ('Decision Tree model (KfoldVal) #2:')
#DT_model_Kf (FN2, y, kf_splits, prediction)

def DT_model_TTS (FN, y, prediction):
    X = df [FN].values
    for criterion in ['gini', 'entropy']:
        print ('Decision Tree - {}'.format (criterion)) 
        X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=101)
        model = DecisionTreeClassifier (criterion=criterion, max_depth=5, min_samples_leaf=5, max_leaf_nodes=10)
        model.fit (X_train, y_train)
        print ('RainTomorrow:', [[model.predict ([prediction])]])
        Accuracy_score (model, X_test, y_test)
        dot_file = export_graphviz (model, feature_names = FN)
        #Graph
        graph = graphviz.Source (dot_file)                         
        graph.render (filename = 'DTM_TTS ' + criterion, format = 'png', cleanup = True) 
#print('Decision Tree Model (TTS) #2:')
#DT_model_TTS (FN2, y, prediction)



#LINEAR REGRESSION MODEL

def LR_model_Kf (FN, y, kf_scores, prediction):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    sensitivity_scores = []
    specificity_scores = []
    roc_auc_scores = []
    c = 0
    kf = KFold (n_splits = kf_splits, shuffle = True)
    X = df [FN].values
    for train_index, test_index in kf.split (X):
        c += 1
        X_train, X_test = X [train_index], X [test_index]
        y_train, y_test = y [train_index], y [test_index]
        model = LogisticRegression (solver = 'liblinear')
        model.fit (X_train, y_train)
        print ('RainTomorrow:', [[model.predict ([prediction])]])
        y_pred = model.predict (X_test) > 0.75
        y_pred_roc = model.predict_proba (X_test)
        #ACCURACY
        accuracy_scores.append (accuracy_score (y_test, y_pred))
        precision_scores.append (precision_score (y_test, y_pred))
        recall_scores.append (recall_score (y_test, y_pred))
        f1_scores.append (f1_score (y_test, y_pred))
        specificity_scores.append (specificity_score (y_test, y_pred))
        roc_auc_scores.append (roc_auc_score (y_test, y_pred_roc [:, 1]))
        if c == kf_splits:
           print ('Accuracy:', np.mean (accuracy_scores))
           print ('Precision:', np.mean (precision_scores))
           print ('Recall:', np.mean (recall_scores))
           print ('F1score:', np.mean (f1_scores))
           print ('Sensitivity:', np.mean (recall_scores))  
           print ('Specificity:', np.mean (specificity_scores)) 
           print ('AUCscore:', np.mean (roc_auc_scores))
           #ROC CURVE
           fpr, tpr, thresholds = roc_curve (y_test, y_pred_roc [:,1])
           plt.plot(fpr, tpr)
           plt.plot([0, 1], [0, 1], linestyle='--')
           plt.scatter (1 - specificity_score (y_test, y_pred), recall_score (y_test, y_pred), c = 'red') 
           plt.xlim([0.0, 1.0])
           plt.ylim([0.0, 1.0])
           plt.xlabel('1 - specificity')
           plt.ylabel('sensitivity')
           plt.show()   	
#print ('Logistic Regression model (KfoldVal) #2')
#LR_model_Kf (FN2, y, kf_splits, prediction)

def LR_model_TTS (FN, y, prediction):
    X = df [FN].values
    model = LogisticRegression (solver = 'liblinear')   
    X_train, X_test, y_train, y_test = train_test_split (X, y, random_state = 27 )
    model.fit (X_train, y_train)
    print ('RainTomorrow:', [[model.predict ([prediction])]])
    Accuracy_score (model, X_test, y_test)
#print ('Logistic Regression Model (TrainTestSplit) #2:')
#LR_model_TTS (FN2, y, prediction)
    
def LR_model (FN, y, prediction):
    X = df [FN].values
    model = LogisticRegression (solver = 'liblinear')
    model.fit (X, y)
    print ('RainTomorrow:', [[model.predict ([prediction])]]) 
    Accuracy_score (model, X, y)
    print (type (prediction), type (X))
#print ('Logistic Regression Model #2:')  
#LR_model (FN2, y, prediction)
