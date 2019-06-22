import numpy as np
import pandas as pd

from titanic_visualizations import survival_stats
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

display(full_data.head())

outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

display(data.head())


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    if len(truth) == len(pred): 
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
predictions = pd.Series(np.ones(5, dtype = int))
print (accuracy_score(outcomes[:5], predictions))

def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        predictions.append(0)
    
    return pd.Series(predictions)

predictions = predictions_0(data)


print (accuracy_score(outcomes, predictions))
# Answer: 61.62%

survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)
    
    return pd.Series(predictions)

predictions = predictions_1(data)
print (accuracy_score(outcomes, predictions))
# Answer: 78.68%

survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            if passenger['Age'] < 10:
                predictions.append(1)
            else:
                predictions.append(0)
    return pd.Series(predictions)

predictions = predictions_2(data)

print (accuracy_score(outcomes, predictions))
# Answer: 79.35%

survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age < 18"])
survival_stats(data, outcomes, 'SibSp', ["Sex == 'female'"])


def predictions_3(data):
    """ Model which makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            if passenger['SibSp'] > 2:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if passenger['Age'] < 10:
                predictions.append(1)
            else:
                predictions.append(0)
    
    return pd.Series(predictions)

predictions = predictions_3(data)

print (accuracy_score(outcomes, predictions))
# Answer: 80.36%
