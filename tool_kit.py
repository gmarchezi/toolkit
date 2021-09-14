import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from dotmap import DotMap
#from pandas_profiling import ProfileReport


##### DATABASES #####
# Some of the databases in this project were extracted from the basedosdados.org project. Access and learn about it.


# Returns a Pandas Dataframe
# Returns False if a database is not loaded
def load_database(name = None):
    name.lower()

    # (PT-BR) Índice de Desenvolvimento da Educação Brasileira Básica (IDEB) por Município
    # (EN-US) Basic Brazilian Education Progress Index by Municipality
    if name == 'ideb':
        try:
            database = pd.read_csv('https://raw.githubusercontent.com/gmarchezi/toolkit/main/databases/Base_IDEB.csv')
        except:
            return False

    # (PT-BR) Campeonato Brasileiro de Futebol Série A
    # (EN-US) Brazilian Championship A Series
    elif name == 'brasileirao':
        try:
            database = pd.read_csv('https://raw.githubusercontent.com/gmarchezi/toolkit/main/databases/Brasileirao_Serie_A.csv')
        except:
            return False

    # (PT-BR) Dados Demográficos de Municípios Brasileiros
    # (EN-US) Demographic Data of Brazilian Municipalities
    elif name == 'municipios':
        try:
            database = pd.read_csv('https://raw.githubusercontent.com/gmarchezi/toolkit/main/databases/Dados_Demograficos_Municipios_2010.csv')    
        except:
            return False

    # (PT-BR) Desmatamento no Brasil por Município
    # (EN-US) Deforestation in Brazil By Municipality    
    elif name == 'desmatamento':
        try:
            database = pd.read_csv('https://raw.githubusercontent.com/gmarchezi/toolkit/main/databases/Desmatamento_Municipios.csv')
        except:
            return False

    # (PT-BR) Dados dos Passageiros do Titanic
    # (EN-US) Titanic Passengers Data
    elif name == 'titanic':
        try:
            database = pd.read_csv('https://raw.githubusercontent.com/gmarchezi/toolkit/main/databases/Titanic.csv')
        except:
            return False

    else:
        return False

    return database


# Plots a graphic that shows the missing values for each column
# Params
    # database = Pandas Dataframe
    # size = list of 2 values (default = [20,5])
    # color = string (default = 'viridis'; alternatives = 'magma','flare','crest',...) see more: https://seaborn.pydata.org/tutorial/color_palettes.html
def null_map(database,size = [20,5],color = 'viridis'):
    fig, ax = plt.subplots(figsize=(size[0],size[1])) 
    sns.heatmap(database.isnull(), 
    yticklabels=False, 
    cbar=False, 
    cmap=color,
    ax=ax)


# Returns a balanced database with a binary target column
# Params
    # database = Pandas Dataframe
    # target = string (name of database's target column)
    # target_values = list of target column's values (only binary values)
def under_sample(database, target, target_values):
    count_x = len(database[database[target] == target_values[0]])
    count_y = len(database[database[target] == target_values[1]])
    if count_x > count_y:
        value1 = target_values[0]
        value2 = target_values[1]
    else:
        value1 = target_values[1]
        value2 = target_values[0]
    
    count_2 = len(database[database[target] == value2])

    balanced_database = database[database[target] == value1].sample(count_2, replace=True)
    balanced_database = pd.concat([balanced_database,database[database[target] == value2]], axis=0)
    
    return balanced_database

# Returns 4 pandas dataframes or 1 dot_map class variable
# Params
    # database = Pandas Dataframe
    # target = String
    # size = Float (percentage of rows in train database)
    # random_st = Integer (Random State)
    # shuffle_data = True/False (Shuffle data before split?)
    # dot_map = True/False (Return a dot_map dict instead of 4 Dataframes?)
def train_test(database,target,size = 0.8, random_st = 0,shuffle_data = True, dot_map = False):
    x_train,x_test,y_train,y_test = train_test_split(
        database.drop([target], axis=1), 
        database[target], 
        train_size = size, test_size = 1 - size,
        random_state=random_st, shuffle=shuffle_data)

    if dot_map == False:
        return x_train,x_test,y_train,y_test
    else:
        return DotMap({'x_train' : x_train,'x_test' : x_test,'y_train' : y_train,'y_test' : y_test})

# Plots a box_plot
# Params
    # database = Pandas Dataframe
    # target = String (Target Column)
    # String (Default = 'red')
def box_plot(database,target,dot_color = 'red'):
    sns.boxplot(x=database[target])
    sns.swarmplot(x=database[target], color=dot_color)