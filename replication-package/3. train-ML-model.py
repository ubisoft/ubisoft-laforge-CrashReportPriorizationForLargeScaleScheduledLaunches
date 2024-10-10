import subprocess
import sys



import pandas as pd
home = 'data/'

n_components = 400

tabular_data = pd.read_csv(home+'g1_tabular_data_'+str(n_components)+'.csv')
tabular_data_p1 = tabular_data
tabular_data_bkp = tabular_data.copy()
tabular_data_p2 = pd.read_csv(home+'g1_tabular_data_'+str(n_components)+'.csv')

max_cumsum= tabular_data[[c for c in tabular_data.columns if '_cumsum' in c]].max().reset_index()
max_cumsum.columns = ['first_component','max']


model_path = home+'models-imbalance-platform-exception-'+str(n_components)+'-components/'

import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_auc_score, roc_curve, auc



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
def predict(X_train_scaled, y_train, X_test_scaled, y_test, X_valid_scaled=None, y_valid=None, models = ['all'], random_state=0, model_name_suffix=""):
    # Create a classifier (example: Logistic Regression)
    
    li_mcc = []
    li_precision = []
    li_recall = []
    li_models = []
    li_reports = []
    li_data = []
    li_suggestions = []
    li_len_suggestions = []
    li_auroc = []
    li_fpr_tpr = []
    li_auprc = []
    li_precision_recall = []
    
    
    def check_contains_1(series):
        return any( val==1 for val in series)

    if 'all' in models or 'LogisticRegression' in models:
        
        from sklearn.linear_model import LogisticRegression   
        print("-------------------")
        print("LogisticRegression")
        print("-------------------")
        classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=random_state)
        classifier.fit(X_train_scaled, y_train)
        with open(model_path+model_name_suffix+'LogisticRegression.pkl', 'wb') as file:
            pickle.dump(classifier, file)
        

        
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        print(classifier.classes_)
        y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
        fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recalls, precisions)
        li_auroc.append(roc_auc)
        li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
        li_auprc.append(pr_auc)
        li_precision_recall.append([precisions, precisions, thresholds_prc])
    
        li_mcc.append(mcc)
        precision = precision_score(y_test, y_pred)
        li_precision.append(precision)
        recall = recall_score(y_test, y_pred)
        li_recall.append(recall)
        li_models.append(classifier)
        li_reports.append(report)

        y_suggestion = classifier.predict(X_valid_scaled)
        tabular_data_p2['y_suggestion'] = y_suggestion
        suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
        li_suggestions.append(suggestions)
        li_len_suggestions.append(len(suggestions))
    
        dataset_test_['y_pred'] = y_pred
        s=""         
        li_data.append("LogisticRegression max_iter=1000, multi_class='multinomial', random_state=0"+s)
    
    
        
        print(f"Accuracy: {accuracy:.2f}")
        print(f"MCC: {mcc:.2f}")
        print("Classification Test Report:\n", report)
    
  
        
    
    if 'all' in models or 'KNeighborsClassifier' in models:
        print("-------------------")
        print("KNeighborsClassifier")
        print("-------------------")
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X_train_scaled, y_train)
        with open(model_path+model_name_suffix+'KNeighborsClassifier.pkl', 'wb') as file:
            pickle.dump(classifier, file)
            
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        print(classifier.classes_)
        y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
        fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recalls, precisions)
        li_auroc.append(roc_auc)
        li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
        li_auprc.append(pr_auc)
        li_precision_recall.append([precisions, precisions, thresholds_prc])
    
        li_mcc.append(mcc)
        precision = precision_score(y_test, y_pred)
        li_precision.append(precision)
        recall = recall_score(y_test, y_pred)
        li_recall.append(recall)
        li_models.append(classifier)
        li_reports.append(report)
        dataset_test_['y_pred'] = y_pred

        y_suggestion = classifier.predict(X_valid_scaled)
        tabular_data_p2['y_suggestion'] = y_suggestion
        suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
        li_suggestions.append(suggestions)
        li_len_suggestions.append(len(suggestions))
    
        s="" 
        li_data.append("KNeighborsClassifier n_neighbors=10"+s)
        
        print(f"Accuracy: {accuracy:.2f}")
        print(f"MCC: {mcc:.2f}")
        print("Classification Test Report:\n", report)
    

    
    
    if 'all' in models or 'MLPClassifier' in models:
    
        recall_li = []
        precision_li = []
        auc_li = []
        auprc_li = [] 
        
        hidden_layer_sizes_li=[(16,16),(16,32,16),(32,32),(32,32,32), (32,64,32), (32,64, 128,32), (64,64), (64,64,64), (64,128,64), (64,128,128, 64), (128,128,128)]
        
        for hidden_layer_sizes in hidden_layer_sizes_li:
            print("-------------------")
            print("MLPClassifier", hidden_layer_sizes)
            print("-------------------")
            from sklearn.neural_network import MLPClassifier
            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', max_iter=1000, random_state=random_state)
            classifier.fit(X_train_scaled, y_train)
            with open(model_path+model_name_suffix+'MLPClassifier_'+str(hidden_layer_sizes)+'.pkl', 'wb') as file:
                pickle.dump(classifier, file)
            
            y_pred = classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print(classifier.classes_)
            y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
            fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(recalls, precisions)
            li_auroc.append(roc_auc)
            li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
            li_auprc.append(pr_auc)
            li_precision_recall.append([precisions, precisions, thresholds_prc])
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)            
            auc_ = roc_auc_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
    
            li_mcc.append(mcc)
            li_precision.append(precision)
            li_recall.append(recall)
            li_models.append(classifier)
            li_reports.append(report)
    
            dataset_test_['y_pred'] = y_pred

            y_suggestion = classifier.predict(X_valid_scaled)
            tabular_data_p2['y_suggestion'] = y_suggestion
            suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
            li_suggestions.append(suggestions)
            li_len_suggestions.append(len(suggestions))
    
            s="" 
            li_data.append("MLPClassifier hidden_layer_sizes="+str(hidden_layer_sizes)+", activation='relu', max_iter=1000, random_state=0" +s)
                    
            precision_li.append(precision)
            recall_li.append(recall)
            auc_li.append(auc_)
                    
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            auprc = auc(recall, precision)
            auprc_li.append(auprc)
        
            print(f"Accuracy: {accuracy:.2f}")
            print(f"MCC: {mcc:.2f}")
            print("Classification Test Report:\n", report)
    


        

    
    n_estimators_li =  [10,20,50,60,70,80,90,100,110,120,150,200]

    if 'all' in models or 'GradientBoostingClassifier' in models:
        
        recall_li = []
        precision_li = []
        auc_li = []
        auprc_li = []     
        for n_estimators in n_estimators_li:
            try:
                print("-------------------")
                print("GradientBoostingClassifier",n_estimators)
                print("-------------------")
                from sklearn.ensemble import GradientBoostingClassifier
                classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, random_state=random_state)
                classifier.fit(X_train_scaled, y_train)
                with open(model_path+model_name_suffix+'GradientBoostingClassifier_'+str(n_estimators)+'.pkl', 'wb') as file:
                    pickle.dump(classifier, file)
                    
                y_pred = classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                print(classifier.classes_)
                y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
                fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
                pr_auc = auc(recalls, precisions)
                li_auroc.append(roc_auc)
                li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
                li_auprc.append(pr_auc)
                li_precision_recall.append([precisions, precisions, thresholds_prc])
            
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)            
                auc_ = roc_auc_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
        
                li_mcc.append(mcc)
                li_precision.append(precision)
                li_recall.append(recall)
                li_models.append(classifier)
                li_reports.append(report)

                y_suggestion = classifier.predict(X_valid_scaled)
                tabular_data_p2['y_suggestion'] = y_suggestion
                suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
                li_suggestions.append(suggestions)
                li_len_suggestions.append(len(suggestions))
        
                dataset_test_['y_pred'] = y_pred
                s=''
                li_data.append("GradientBoostingClassifier n_estimators="+str(n_estimators)+", learning_rate=0.1, random_state=0" + s)
                        
                precision_li.append(precision)
                recall_li.append(recall)
                auc_li.append(auc_)
                        
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
                auprc = auc(recall, precision)
                auprc_li.append(auprc)
            
                print(f"Accuracy: {accuracy:.2f}")
                print(f"MCC: {mcc:.2f}")
                print("Classification Report:\n", report)
            except:
                pass
    
        try:
            plt.plot(n_estimators_li,recall_li, label="Recall")
            plt.plot(n_estimators_li,precision_li, label="Precision")
            plt.plot(n_estimators_li,auc_li, label="AUROC")
            plt.plot(n_estimators_li,auprc_li, label="AUPRC")
            plt.xlabel("Number of estimators in GradientBoostingClassifier with lr=0.1")
            plt.ylabel("Metric")
            plt.legend()
            plt.show()
        except:
            pass

    if 'all' in models or 'XGboost' in models:
        
        try:
            print("-------------------")
            print("XGboost")
            print("-------------------")
            
            import xgboost as xgb
            # Convert data into DMatrix format
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            # Set the hyperparameters
            params = {
                'objective': 'multi:softmax',  # Classification problem
                'num_class': 2,                # Number of classes in the target variable
                'tree_method': 'gpu_hist'      # Use GPU for training
            }
            
            # Train the XGBoost model
            num_round = 5
            bst = xgb.train(params, dtrain, num_round, random_state=random_state)
            with open(model_path+model_name_suffix+'XGboost'+'.pkl', 'wb') as file:
                pickle.dump(bst, file)
            
            # Make predictions
            y_pred = bst.predict(dtest)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            print(classifier.classes_)
            y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
            fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(recalls, precisions)
            li_auroc.append(roc_auc)
            li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
            li_auprc.append(pr_auc)
            li_precision_recall.append([precisions, precisions, thresholds_prc])
        
            li_mcc.append(mcc)
            precision = precision_score(y_test, y_pred)
            li_precision.append(precision)
            recall = recall_score(y_test, y_pred)
            li_recall.append(recall)
            li_models.append(classifier)
            li_reports.append(report)
            dataset_test_['y_pred'] = y_pred

            y_suggestion = classifier.predict(X_valid_scaled)
            tabular_data_p2['y_suggestion'] = y_suggestion
            suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
            li_suggestions.append(suggestions)
            li_len_suggestions.append(len(suggestions))
    
            s=''
            li_data.append("XGboost"+s)
            
            print(f"Accuracy: {accuracy:.2f}")
            print(f"MCC: {mcc:.2f}")
            print("Classification Report:\n", report)
        except:
            pass
    
    if 'all' in models or 'RandomForestClassifier' in models:
        recall_li = []
        precision_li = []
        auc_li = []
        auprc_li = []  
        for n_estimators in n_estimators_li:
            try:
                print("-------------------")
                print("RandomForestClassifier", n_estimators)
                print("-------------------")
            
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                classifier.fit(X_train_scaled, y_train)
                with open(model_path+model_name_suffix+'RandomForestClassifier_'+str(n_estimators)+'.pkl', 'wb') as file:
                    pickle.dump(classifier, file)
                    
                y_pred = classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                print(classifier.classes_)
                y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
                fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
                pr_auc = auc(recalls, precisions)
                li_auroc.append(roc_auc)
                li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
                li_auprc.append(pr_auc)
                li_precision_recall.append([precisions, precisions, thresholds_prc])
                
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)            
                auc_ = roc_auc_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
        
                li_mcc.append(mcc)
                li_precision.append(precision)
                li_recall.append(recall)
                li_models.append(classifier)
                li_reports.append(report)
        
                dataset_test_['y_pred'] = y_pred

                y_suggestion = classifier.predict(X_valid_scaled)
                tabular_data_p2['y_suggestion'] = y_suggestion
                suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
                li_suggestions.append(suggestions)
                li_len_suggestions.append(len(suggestions))
    
                s=''
                li_data.append("RandomForestClassifier n_estimators="+str(n_estimators)+", random_state=0"+s)
                        
                precision_li.append(precision)
                recall_li.append(recall)
                auc_li.append(auc_)
                        
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
                auprc = auc(recall, precision)
                auprc_li.append(auprc)
            
                print(f"Accuracy: {accuracy:.2f}")
                print(f"MCC: {mcc:.2f}")
                print("Classification Report:\n", report)
            except:
                pass
    
        try:
            plt.plot(n_estimators_li,recall_li, label="Recall")
            plt.plot(n_estimators_li,precision_li, label="Precision")
            plt.plot(n_estimators_li,auc_li, label="AUROC")
            plt.plot(n_estimators_li,auprc_li, label="AUPRC")
            plt.xlabel("Number of estimators in RandomForestClassifier")
            plt.ylabel("Metric")
            plt.legend()
            plt.show()
        except:
            pass

    if 'all' in models or 'KNeighborsClassifier' in models:
    
        recall_li = []
        precision_li = []
        auc_li = []
        auprc_li = []  
        n_estimators_li = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        for n_estimators in n_estimators_li:
    
            try:
        
                print("-------------------")
                print("KNeighborsClassifier", n_estimators)
                print("-------------------")
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier(n_neighbors=n_estimators)
                classifier.fit(X_train_scaled, y_train)
                with open(model_path+model_name_suffix+'KNeighborsClassifier'+str(n_estimators)+'.pkl', 'wb') as file:
                    pickle.dump(classifier, file)
                    
                y_pred = classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                print(classifier.classes_)
                y_scores = classifier.predict_proba(X_test_scaled)[:, 1] 
                fpr, tpr, thresholds_auroc = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                precisions, recalls, thresholds_prc = precision_recall_curve(y_test, y_scores)
                pr_auc = auc(recalls, precisions)
                li_auroc.append(roc_auc)
                li_fpr_tpr.append([fpr, tpr, thresholds_auroc ])
                li_auprc.append(pr_auc)
                li_precision_recall.append([precisions, precisions, thresholds_prc])
            
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)            
                auc_ = roc_auc_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
        
                li_mcc.append(mcc)
                li_precision.append(precision)
                li_recall.append(recall)
                li_models.append(classifier)
                li_reports.append(report)
                
                dataset_test_['y_pred'] = y_pred

                y_suggestion = classifier.predict(X_valid_scaled)
                tabular_data_p2['y_suggestion'] = y_suggestion
                suggestions  = tabular_data_p2[tabular_data_p2['y_suggestion']==True]['CrashType']
                li_suggestions.append(suggestions)
                li_len_suggestions.append(len(suggestions))
    
                s=''
                li_data.append("KNeighborsClassifier n_estimators="+str(n_estimators)+s)
                        
                precision_li.append(precision)
                recall_li.append(recall)
                auc_li.append(auc_)
                        
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
                auprc = auc(recall, precision)
                auprc_li.append(auprc)
                
                print(f"Accuracy: {accuracy:.2f}")
                print(f"MCC: {mcc:.2f}")
                print("Classification Report:\n", report)
            except:
                pass
    
        try:
            plt.plot(n_estimators_li,recall_li, label="Recall")
            plt.plot(n_estimators_li,precision_li, label="Precision")
            plt.plot(n_estimators_li,auc_li, label="AUROC")
            plt.plot(n_estimators_li,auprc_li, label="AUPRC")
            plt.xlabel("Number of estimators in RandomForestClassifier")
            plt.ylabel("Metric")
            plt.legend()
            plt.show()
        except:
            pass



    df = pd.DataFrame({
        "mcc": li_mcc,
        "auroc": li_auroc,
        "auprc":li_auprc,
        "precision": li_precision,
        "recall": li_recall,
        "model":li_models,
        "report":li_reports,
        "data": li_data,
        "suggestions":li_suggestions,
        "len_suggestions":li_len_suggestions
    })

    return df

def create_train_test_validate_datasets(p1_data, p2_data, label_column, test_ratio=0.30, total_count_threshold = 0, 
                               is_and = True, daily_max_count_threshold=0, random_state=0,
                                stratifiied=False, cumsum=True, mean = True, oversampling=False,sampling_strategy=1,
                               viral_callstack_data = False, 
                                        callstack_features = False, platform_features = False, 
                                        subtype_features = False, application_context_features = False
                              ):
    data_ = p1_data.copy()

    if is_and:
        data_[label_column] = (data_['total_count']>=total_count_threshold) & (data_['daily_max_count']>=daily_max_count_threshold)
    else:
        data_[label_column] = (data_['total_count']>=total_count_threshold) | (data_['daily_max_count']>=daily_max_count_threshold)

    if viral_callstack_data:
        data_,p2_data = get_viral_callstack_data(data_,label_column, project_2 = True, tabular_data_p2=p2_data)

    data_ = data_.sort_values('CrashType')


    if cumsum==False:
        data_ = data_[[c for c in data_.columns if "_cumsum" not in c]]
    
    if mean==False:
        data_ = data_[[c for c in data_.columns if "_mean_" not in c]]    

    n=int(len(data_[['CrashType']].drop_duplicates())*test_ratio)
    print("n", n)
    dataset_sample = data_[['CrashType']].drop_duplicates().sample(n=n, random_state=random_state)
    
    print(len(data_[['CrashType']].drop_duplicates()))
    
    dataset_test_ = data_.merge(dataset_sample, on='CrashType')    
    dataset_sample['test_data'] = True
    dataset_train_ = data_.merge(dataset_sample, on='CrashType', how='outer')
    dataset_train_ = dataset_train_[dataset_train_['test_data'].isna()]
    dataset_train_ = dataset_train_.drop(['test_data'], axis=1)
    dataset_test_ = dataset_test_.groupby('CrashType').head(1)

    
    dataset_test = dataset_test_.drop([
    'total_count',
    'daily_max_count',
     'CrashType'
    ], axis=1).fillna(0)
    
    dataset_train = dataset_train_.drop([
    'total_count',
    'daily_max_count',
     'CrashType'
    ], axis=1).fillna(0)

    
    for c in dataset_train.columns:
        try:
            if 'cumsum' in c:
                cumsum = max_cumsum[max_cumsum['first_component']==c]['max'].values[0]
        except:
            print('Error',c)
        if c not in p2_data.columns:
            p2_data[c] = 0
        else:
            if 'cumsum' in c:
                p2_data[c] = p2_data[c].map(lambda x: x+cumsum if x>0 else 0)
    
    # dataset_valid  =   p2_data[ dataset_train.columns]
    
    dataset_train.columns = [str(i) for i in dataset_train.columns]
    dataset_test.columns = [str(i) for i in dataset_test.columns]
    

    for c in max_cumsum['first_component'].values:
        p2_data[c] = p2_data[c] + max_cumsum[max_cumsum['first_component']==c]['max'].values[0]
    
    X_train = dataset_train.drop([label_column], axis=1)
    y_train = dataset_train[label_column]
    if oversampling=='SMOTE':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    elif oversampling=='RandomOverSampler':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_train, y_train = ros.fit_resample(X_train, y_train) 
    X_test = dataset_test.drop([label_column], axis=1)
    y_test = dataset_test[label_column]
    X_valid = p2_data[X_test.columns].fillna(0)
    try:
        
        y_valid = p2_data[label_column]
    except:
        y_valid = None

    li = []

    if callstack_features:
        li+=[c for c in dataset_train.columns if 'first_component' in c ]

    if platform_features:
        li+=[c for c in tabular_data.columns if 'Platform_' in c ]

    if subtype_features:
        li+=[c for c in tabular_data.columns if 'ExceptionType_' in c]

    if application_context_features:
        li+=[c for c in tabular_data.columns if 'ApplicationContext_' in c]

    

    X_train = X_train[li]
    X_test = X_test[li]
    X_valid = X_valid[li]
    
    print("X_test",len(X_test))
    print("X_train",len(X_train))
    # print("X_valid",len(X_valid))
    # Perform feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)
    
    return dataset_test_,dataset_train_, X_train_scaled, y_train, X_test_scaled,y_test, X_valid_scaled, y_valid
	
	
li = []
oversampling_li = ["SMOTE", "RandomOverSampler"]
sampling_strategy_li = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


for oversampling in oversampling_li:
    for sampling_strategy in sampling_strategy_li:

        print("---",oversampling,sampling_strategy,"---")

        dataset_test_,dataset_train_, X_train_scaled, y_train, X_test_scaled, y_test, X_valid_scaled, y_valid = create_train_test_validate_datasets(
            tabular_data, tabular_data_p2, 'Class',test_ratio=0.20, total_count_threshold = t_g1_total_count, is_and = False, 
            daily_max_count_threshold=t_g1_daily_max_count, random_state=10, stratifiied=False,
            cumsum = False, mean=False, oversampling = oversampling, sampling_strategy = sampling_strategy, viral_callstack_data=False, callstack_features = True, platform_features=True,
            subtype_features = True, application_context_features = False
        )
        res = predict(X_train_scaled, y_train, X_test_scaled, y_test, X_valid_scaled=X_valid_scaled, random_state=10, models = [
           'LogisticRegression',
           'MLPClassifier',
           'GradientBoostingClassifier',
           'XGboost',
           'RandomForestClassifier'
          ], model_name_suffix = str(oversampling)+'_'+str(sampling_strategy)+'_')
        
        li_ = [oversampling,sampling_strategy, res]

        df = li_[2]
        df['oversampling'] = li_[0]
        df['sampling_strategy'] = li_[1]
        df.to_json(home+'scores-imbalance-platform-exception-'+str(n_components)+'-components/g2-suggestions-'+str(oversampling)+'-'+str(sampling_strategy)+'.json')

