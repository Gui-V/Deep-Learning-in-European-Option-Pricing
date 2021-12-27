#%% Main code
#Imports
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
from numpy.random import seed
seed(123)
import random as python_random
python_random.seed(123)
import tensorflow as tf
tf.set_random_seed(1234)

#pip show tensorflow
import os 
#os.getcwd()

#pip list
#tf.test.is_gpu_available() 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
from keras.callbacks import EarlyStopping
import random

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import numpy as np, pandas as pd
from numpy import log
import numpy.random as npr
import scipy.linalg
import csv
import seaborn as  sns
from functools import reduce
from operator import add


#%% Create Dataset
pd.DataFrame(np.random.uniform(0.4, 1.0, size=(20, 1)))


#%% Black-Scholes functions to generate synthetic market data
def BS_d1(S, X, r, tau, sigma):
    # d1 Function for BS model
    d1 = (np.log(S/X)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return(d1)


def BS_d2(S, X, r, tau, sigma):
    # d2 Function for BS model
    d2 = BS_d1(S, X, r, tau, sigma)-sigma*np.sqrt(tau)
    return(d2)

#d1 and d2
def BS(S, X, r, tau, sigma, opt_type):
    """Standard Black-Scholes formula for call and put options"""
    d1 = BS_d1(S, X, r, tau, sigma)
    d2 = BS_d2(S, X, r, tau, sigma)

    return np.where(opt_type == "C",(S*scipy.stats.norm.cdf(d1)-X*np.exp(-r*tau) * \
            scipy.stats.norm.cdf(d2)), (-S*scipy.stats.norm.cdf(-d1)+X *\
                np.exp(-r*tau)*scipy.stats.norm.cdf(-d2)))

#Black-Scholes - Main function to get call option price
def df_BS_function(df):
    """Function indexing columns in pandas dataframe"""
    S = df['S']  # close
    X = df['K']  # strike
    r = df['risk_free']  # /100 #discount-monthly
    tau = df['maturity']  # "normmat"
    sigma = df['implied_vol'] # volat100
    opt_type = df['opt_type'] # call or put
    C = BS(S, X, r, tau, sigma, opt_type)
    return(C)


# %% Load Stock data price
def load_data(filename):
    """Loads dataset from .csv file"""
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row[4])
    return(rows[1:])

#import file
price_list=load_data("TSLA.csv")
price_list=[float(i) for i in price_list]

#len(load_data("TSLA.csv"))

dataset_size=100000 #Predefine the dataset size

k_size=int(dataset_size/len(price_list))

dataset=pd.DataFrame(np.zeros((k_size*len(price_list),1)))

#Loop to create dataset 
for idx in range(0,len(price_list)):
    seed(123)
    print(idx)
    price_list[idx]*np.random.uniform(0.4,1,k_size)
    if idx== 0:
        
        dataset.loc[idx*k_size:idx+k_size-1, 'S']=[price_list[idx]]*k_size
        dataset.loc[idx*k_size:idx+k_size-1, 'K']=price_list[idx]*np.random.uniform(0.4,1.4,k_size)
        dataset.loc[idx*k_size:idx+k_size-1, 'maturity']=np.random.uniform(1/250,1,k_size)
        dataset.loc[idx*k_size:idx+k_size-1, 'risk_free']=np.random.uniform(0,0.05,k_size)
        dataset.loc[idx*k_size:idx+k_size-1, 'implied_vol']=np.random.uniform(0,0.05,k_size)

    else:
        
        dataset.loc[idx*k_size:idx*k_size+k_size-1, 'S']=[price_list[idx]]*k_size
        dataset.loc[idx*k_size:idx*k_size+k_size-1, 'K']=price_list[idx]*np.random.uniform(0.4,1.4,k_size)
        dataset.loc[idx*k_size:idx*k_size+k_size-1, 'maturity']=np.random.uniform(1/250,1,k_size)
        dataset.loc[idx*k_size:idx*k_size+k_size-1, 'risk_free']=np.random.uniform(0,0.05,k_size)
        dataset.loc[idx*k_size:idx*k_size+k_size-1, 'implied_vol']=np.random.uniform(0,0.05,k_size)


dataset['S/K']=dataset['S']/dataset['K'] #Unused normalization
dataset['opt_type']='C' #Call price 
#dataset['opt_type'][2]='P' # If one want to use put option price instead call price (Please adjuste the Black Scholes Formula in the function)

dataset['opt_price']=df_BS_function(dataset)
dataset['O/K']=dataset['opt_price']/dataset['K']
dataset['O']=dataset['opt_price']

dataset=dataset.sample(frac=1) #Shuffle the dataset

#Histogram dataset
variaveis_plot=dataset.columns.tolist()
#pivot1[numerical].hist(bins=20, figsize=(10, 10), layout=(40, 40), xlabelsize=8, ylabelsize=8);
dataset[variaveis_plot].hist(bins=20, figsize=(20, 30), layout=(8, 8), xlabelsize=8, ylabelsize=8);
# %%
n_train =  int(0.5 * dataset_size)

train = dataset[0:n_train]

#Normalization
train_S_mean=train['S'].mean(axis=0)
train_O_mean=train['O'].mean(axis=0)
train_K_mean=train['K'].mean(axis=0)

train_S_std=train['S'].std(axis=0)
train_O_std=train['O'].std(axis=0)
train_K_std=train['K'].std(axis=0)

train['S']-=train_S_mean
train['S']/=train_S_std

train['O']-=train_O_mean
train['O']/=train_O_std

train['K']-=train_K_mean
train['K']/=train_K_std
train=train[(train['K'] >= 0) & (train['S'] >= 0) & (train['O'] >= 0)] #Remove the negative values
X_train = train[['S','maturity','K','risk_free','implied_vol']].values
y_train = train['O'].values

#Train data pairplot
sns.pairplot(train, x_vars=['S','maturity','K','risk_free','implied_vol'], y_vars=['O'])

# %%

test = dataset[n_train:dataset_size]
 
 #Normalization
test['S']-=train_S_mean
test['S']/=train_S_std

test['O']-=train_O_mean
test['O']/=train_O_std

test['K']-=train_K_mean
test['K']/=train_K_std 

test=test[(test['K'] >= 0) & (test['S'] >= 0)& (test['O'] >= 0)] #Remove the negative values

X_test = test[[ 'S','maturity','K','risk_free','implied_vol']].values
y_test = test['O'].values

#Test data pairplot
sns.pairplot(test, x_vars=['S','maturity','K','risk_free','implied_vol'], y_vars=['O'])

# %% Activation function - To ensure the output is allway positive

def custom_activation(x):
    return backend.exp(x)

# %% Baseline model
nodes = 32

def build_model():
    model = Sequential()

    model.add(Dense(nodes, input_dim=X_train.shape[1]))
    #model.add(Dropout(0.25)) #Dropout if needed to reduce overfiting

    model.add(Dense(nodes, activation='relu'))
    #model.add(Dropout(0.25)) #Dropout if needed to reduce overfiting

    model.add(Dense(nodes, activation='relu'))
    #model.add(Dropout(0.25)) #Dropout if needed to reduce overfiting

    model.add(Dense(nodes, activation='relu'))
    #model.add(Dropout(0.25)) #Dropout if needed to reduce overfiting

    model.add(Dense(1))
    #model.add(Activation(custom_activation)) # Without activation function
            
    model.compile(loss='mse',optimizer='rmsprop',metrics=['mae']) 
    return model

#Fit the model###
epoc=5
model=build_model()
model.fit(X_train, y_train, batch_size=32, epochs=epoc, validation_split=0.1, verbose=2)

dic=build_model().fit(X_train, y_train, batch_size=5, epochs=epoc, validation_split=0.1, verbose=2).history


#Function to check scores
def Checkscores(y,y_hat):

    stats = dict()
    
    stats['diff'] = y - y_hat
    
    stats['mse'] = np.mean(stats['diff']**2)
    print ("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = np.sqrt(stats['mse'])
    print ("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(stats['diff']))
    print ("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print ("Mean Percent Error:      ", stats['mpe'])
    
    #plots
    mpl.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14,10))
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price',fontsize=20,)
    plt.ylabel('Predicted Price',fontsize=20) 
    plt.show()
    
    plt.figure(figsize=(14,10))
    plt.hist(stats['diff'], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()
    
    return stats

#Training data evaluation
y_train_hat = model.predict(X_train)
y_train_hat = np.squeeze(y_train_hat)
Checkscores(y_train, y_train_hat)
######

#Test data evaluation
y_test_hat = model.predict(X_test)
y_test_hat = np.squeeze(y_test_hat)
Checkscores(y_test, y_test_hat)
######


#Training and Validation loss
dic.keys()
loss_values=dic['loss']
val_loss_values=dic['val_loss']
epochs=range(1,epoc+1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Training and Validation MAE
dic.keys()
mae_values=dic['mae']
val_mae_values=dic['val_mae']
epochs=range(1,epoc+1)
plt.plot(epochs, mae_values, 'bo', label='Training MAE')
plt.plot(epochs, val_mae_values, 'b', label='Validation MAE')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()


#OR

#Cross validation - 10 fold ###
estimator = KerasRegressor(build_fn=build_model, epochs=epoc, batch_size=64, verbose=2)
kfold = KFold(n_splits=5, shuffle=True,random_state=1234)
results = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
print(results)

#Manual KFOLD
k=5
num_val_samples=len(train)//k
#epoc
all_scores=[]
all_mae_histories=[]

for i in range(k):
    print('processing fold #',i)
    val_data=X_train[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=y_train[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data=np.concatenate([X_train[:i*num_val_samples],X_train[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([y_train[:i*num_val_samples],y_train[(i+1)*num_val_samples:]],axis=0)

    model=build_model()
    history=model.fit(partial_train_data,partial_train_targets,epochs=epoc,batch_size=64,verbose=0)
    #val_mse, val_mae=model.evaluate(val_data,val_targets,verbose=0)
    #all_scores.append(val_mae)
    mae_history=history.history['mae']
    all_mae_histories.append(mae_history)


#Plot validation MAE
average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(epoc)]
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# %% Brute force grid search

def generate_network_list(nn_param_choices):
    """Generate a list of all possible networks.
    Args:
        nn_param_choices (dict): The parameter choices
    Returns:
        networks (list): A list of network objects
    """
    networks = []

    # This is silly.
    for nbn in nn_param_choices['nb_neurons']:
        for nbl in nn_param_choices['nb_layers']:
            for a in nn_param_choices['activation']:
                for o in nn_param_choices['optimizer']:

                    # Set the parameters.
                    network = {
                        'nb_neurons': nbn,
                        'nb_layers': nbl,
                        'activation': a,
                        'optimizer': o,
                    }
                    networks.append(network)
    return networks

#Hyperparameter tunning
#Parameters to be tunned
nn_param_choices = {
        'nb_neurons': [25,32,64,120],
        'nb_layers': [1,2,3],
        'activation': ['relu'],
        'optimizer': ['rmsprop','adam']
    }

networks=generate_network_list(nn_param_choices)


#Callback to Early stopping.
early_stopper = EarlyStopping(patience=2) 

#Train the network and save scores and parameters
def train_networks(networks, x_train, y_train,x_test, y_test):

    s=[]
    net=[]
    for network in networks:
        sc=train_and_score(network, x_train, y_train,x_test, y_test)
        print (network)
        print (sc)
        s.append(sc)
        net.append(network)
    return [s,net]

#Compile the model
def compile_model(network, nb_outputs, input_shape):
    
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        #model.add(Dropout(0.2))  # Dropout if needed

    # Output layer.
    model.add(Dense(nb_outputs))
    #model.add(Activation(custom_activation))

    model.compile(loss='mse', optimizer=optimizer)

    return model

#Train and score the network
def train_and_score(network, x_train, y_train,x_test, y_test):
    
    batch_size = 64
    input_shape = (X_train.shape[1],)
    nb_outputs=1

    model = compile_model(network,nb_outputs, input_shape)
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping - large number
              verbose=2,
              validation_split=0.1,
              callbacks=[early_stopper])

    score = model.evaluate(X_test, y_test, verbose=0)
    
    return score  # 1 is mae. 0 is mse.

#Run Brute force
brute_score=train_networks(networks, X_train, y_train,X_test, y_test)
#Example: First score -> brute_score[0][0] and First network ->brute_score[1][0]

#Network with best loss - After brute force search
#Best loss value
best_value=min(brute_score[0])
#Best network parameters
best_network=brute_score[1][brute_score[0].index(min(brute_score[0]))] 

#%% Genetic Algorithms

#Generate Random pop based on parameters
def create_random(nn_param_choices):
    network={}
    for key in nn_param_choices:
        network[key] = random.choice(nn_param_choices[key])
    return network

#Generate list of dict with parameters 
def create_population(count, nn_param_choices):

        pop = []
        
        for _ in range(0, count):
            # Create a random network.
            net=create_random(nn_param_choices)

            pop.append(net)

        return pop

#fitness function to get scores (based on loss function=mse)
def fitness(network,x_train, y_train,x_test, y_test):
    fit_score=train_and_score(network, x_train, y_train,x_test, y_test)
    print(fit_score)
    return fit_score

 
def grade(population,x_train, y_train,x_test, y_test):
    summed = reduce(add, (fitness(network,x_train, y_train,x_test, y_test) for network in population))
    return summed / float((len(population)))

 
#%%
def mutate(network):

    # Choose a random key.
    mutation = random.choice(list(nn_param_choices.keys()))

    # Mutate one of the params.
    network.network[mutation] = random.choice(nn_param_choices[mutation])

    return network


#%% 

#GA parameters
mutate_chance=0.02
retain=0.5
random_select=0.1

def breed(mother, father):
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for child.
            for param in nn_param_choices:
                child[param] = random.choice([mother[param], father[param]])

            # Create a network object.
            net=child

            # Randomly mutate some of the children.
            if mutate_chance > random.random():
                net = mutate(child)

            children.append(net)

        return children


def evolve(pop,x_train, y_train,x_test, y_test):
    
    # Get scores for each network.
    graded = [(fitness(network,x_train, y_train,x_test, y_test), network) for network in pop]
    sorted_scores=[x[0] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
    print(sorted_scores)

    # Sort on the scores.
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]

   
    # Get the number we want to keep for the next gen.
    retain_length = int(len(graded)*retain)

    # The parents are every network we want to keep.
    parents = graded[:retain_length]

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:
        print('while')
        # Get a random mom and dad.
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]

            # Breed
            babies = breed(male, female)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

    parents.extend(children)

    return parents #parents from gen 


def generate(generations, population, nn_param_choices, x_train, y_train,x_test, y_test):
    networks = create_population(population,nn_param_choices) 

    # Evolve the generation.
    for i in range(generations):
        print("***Doing generation %d of %d***" %(i + 1, generations))

        # Train and get accuracy for networks.
        #train_networks(networks, x_train, y_train,x_test, y_test)

        # Print out the average accuracy each generation.
        #print("Generation average: %.2f%%" % (average_fit_score * 100))

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = evolve(networks,x_train, y_train,x_test, y_test)

        if i==(generations-1): #if last gen print
            
            final=[(fitness(network,x_train, y_train,x_test, y_test), network) for network in networks]
            print(final)
            sorted_final=[x[0] for x in sorted(final, key=lambda x: x[0], reverse=False)]
            print(sorted_final)


"""Evolve a network."""
generations = 3  # Number of times to evole the population.
population = 5  # Number of networks in each generation.


nn_param_choices = {
        'nb_neurons': [25,32,64,120],
        'nb_layers': [1,2,3],
        'activation': ['relu'],
        'optimizer': ['rmsprop','adam']
    }


generate(generations, population, nn_param_choices,  X_train, y_train,X_test, y_test)


#%% Final model using Genetic Algorithms

# %% Baseline model
nodes = 32

def build_model_final():
    model = Sequential()

    model.add(Dense(nodes, input_dim=X_train.shape[1]))
    #model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='relu'))
    #model.add(Dropout(0.25))

    model.add(Dense(1))
    #model.add(Activation(custom_activation)) # Without activation function
            
    model.compile(loss='mse',optimizer='adam',metrics=['mae']) 
    return model

#Fit###
epoc=10000
model=build_model_final()
tf.set_random_seed(1234)
model.fit(X_train, y_train, batch_size=64, epochs=epoc, validation_split=0.1, callbacks=[early_stopper],verbose=2)


#Test data evaluation
y_test_hat = model.predict(X_test)
y_test_hat = np.squeeze(y_test_hat)
Checkscores(y_test, y_test_hat)
######




