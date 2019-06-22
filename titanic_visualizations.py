import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_data(data, condition):
    
    field, op, value = condition.split(" ")
    
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")
    
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")
    
    data = data[matches].reset_index(drop = True)
    return data

def survival_stats(data, outcomes, key, filters = []):
    
    if key not in data.columns.values :
        print ("'{}' is not a feature of the Titanic data. Did you spell something wrong?".format(key))
        return False

    if(key == 'Cabin' or key == 'PassengerId' or key == 'Ticket'):
        print ("'{}' has too many unique categories to display! Try a different feature.".format(key))
        return False

    all_data = pd.concat([data, outcomes], axis = 1)
    
    for condition in filters:
        all_data = filter_data(all_data, condition)

    all_data = all_data[[key, 'Survived']]
    
    plt.figure(figsize=(8,6))

    if(key == 'Age' or key == 'Fare'):
        
        all_data = all_data[~np.isnan(all_data[key])]
        
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value

        if(key == 'Fare'):
            bins = np.arange(0, all_data['Fare'].max() + 20, 20)
        if(key == 'Age'):
            bins = np.arange(0, all_data['Age'].max() + 10, 10)
        
        nonsurv_vals = all_data[all_data['Survived'] == 0][key].reset_index(drop = True)
        surv_vals = all_data[all_data['Survived'] == 1][key].reset_index(drop = True)
        plt.hist(nonsurv_vals, bins = bins, alpha = 0.6,
                 color = 'red', label = 'Did not survive')
        plt.hist(surv_vals, bins = bins, alpha = 0.6,
                 color = 'green', label = 'Survived')
    
        plt.xlim(0, bins.max())
        plt.legend(framealpha = 0.8)
    
    else:
       
        if(key == 'Pclass'):
            values = np.arange(1,4)
        if(key == 'Parch' or key == 'SibSp'):
            values = np.arange(0,np.max(data[key]) + 1)
        if(key == 'Embarked'):
            values = ['C', 'Q', 'S']
        if(key == 'Sex'):
            values = ['male', 'female']

        frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Survived','NSurvived'))
        for i, value in enumerate(values):
            frame.loc[i] = [value, \
                   len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)]), \
                   len(all_data[(all_data['Survived'] == 0) & (all_data[key] == value)])]

        bar_width = 0.4

        for i in np.arange(len(frame)):
            nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['NSurvived'], width = bar_width, color = 'r')
            surv_bar = plt.bar(i, frame.loc[i]['Survived'], width = bar_width, color = 'g')

            plt.xticks(np.arange(len(frame)), values)
            plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
    plt.show()

    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print ("Passengers with missing '{}' values: {} ({} survived, {} did not survive)").format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0))
                                                                                                   
                                                                                                   
