
# coding: utf-8

# # PUBG Finish Placement Prediction

# ### File descriptions
# * train_V2.csv - the training set
# * test_V2.csv - the test set
# 
# ### Data fields
# * DBNOs - Number of enemy players knocked.
# * assists - Number of enemy players this player damaged that were killed by teammates.
# * boosts - Number of boost items used.
# * damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# * headshotKills - Number of enemy players killed with headshots.
# * heals - Number of healing items used.
# * Id - Player’s Id
# * killPlace - Ranking in match of number of enemy players killed.
# * killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * kills - Number of enemy players killed.
# * longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# * matchDuration - Duration of match in seconds.
# * matchId - ID to identify match. There are no matches that are in both the training and testing set.
# * matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# * rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# * revives - Number of times this player revived teammates.
# * rideDistance - Total distance traveled in vehicles measured in meters.
# * roadKills - Number of kills while in a vehicle.
# * swimDistance - Total distance traveled by swimming measured in meters.
# * teamKills - Number of times this player killed a teammate.
# * vehicleDestroys - Number of vehicles destroyed.
# * walkDistance - Total distance traveled on foot measured in meters.
# * weaponsAcquired - Number of weapons picked up.
# * winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# * groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# * numGroups - Number of groups we have data for in the match.
# * maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# * winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

#common imports 
import pandas as pd
import numpy as np
import os
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns



train = pd.read_csv("../pubg-finish-placement-prediction/train_V2.csv")
test = pd.read_csv("../pubg-finish-placement-prediction/test_V2.csv")

train.head()
train.info()

train = train.dropna()
test = test.dropna()

# ## Correlational Matrix
data = train.copy()
f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.6, fmt= '.2f',ax=ax)
plt.savefig("Correlational_Matrix.png")
plt.show()

print("The average walk distance, in meters, is:", data["walkDistance"].mean())
data = train.copy()
data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]
plt.figure(figsize=(12,10))
plt.title("Walking Distance Distribution",fontsize=30)
sns.distplot(data['walkDistance'])
plt.savefig("Walking_Distance_Distribution.png")
plt.show()


# The correlation between walk distance and the win place percentage fundamentally makes sense. Since the goal of the game is to survive as long as possible, a player with a longer walk distance implies he has been alive for a longer time than most. As a result, the odds of victory for such a player is increased.
data = train.copy()
import seaborn as sns
print(sns.__version__)
plt.figure(figsize=(30, 15))
plt.title("Kills Distribution", fontsize=30)
sns.scatterplot(x="winPlacePerc", y="kills", data=data[data["kills"] < 30], color="red")
plt.savefig("Kills_Distribution.png")
plt.show()

data = train.copy()

plt.figure(figsize=(30, 15))
plt.title("Kill Distance", fontsize = 30)
sns.distplot(data["longestKill"])
plt.savefig("Kill_Distance_Distribution.png")
plt.show()


# On the graph above, we can see some outliers being detected on the far right, just after the 1000 meter mark. Seeing as these data points are anomalies, it gives us an initial suspicions that these players may in fact be cheaters. For the sake of our project, we decided to remove these potential cheaters from our data set.
print("The average amount of weapons acquired, per player, in a game is", data["weaponsAcquired"].mean())

data = train.copy()

data.hist('weaponsAcquired', figsize = (20,10), range=(0, 10))
plt.savefig("Weapons_Acquired.png")

data = train.copy()

plt.figure(figsize=(30, 15))
sns.scatterplot(x="winPlacePerc", y="weaponsAcquired", data=data[data["weaponsAcquired"] < 30], color = "blue")
plt.savefig("winPlacePerc_weaponsAcquired.png")
plt.show()

#inspired from other noteboos
train['playersInGame'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(25,10))
sns.countplot(train[train['playersInGame']>60]['playersInGame'])
plt.title("Players in game", fontsize=20)
plt.savefig("PlayersInGame_bigger_than_60.png")
plt.show()

# we will create normalized feautures for several of the features to normalize based on the players in the game
train['killsNorm'] = train['kills']*((100-train['playersInGame'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersInGame'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersInGame'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersInGame'])/100 + 1)

#compare the normalized ones and regular ones
to_show = ['playersInGame','kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']

# get the total distance by adding all the dinstances together
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']


# ## Outliers (cheaters) 

# We now find outliers and cheaters in our data and remove them from the dataset

# We find the players who got kills without even moving. It is very unlikely that a player hasn't moved in a game and got a kill, these players are most likely cheaters, so we remove them.

train['killsWithoutMoving'] = ((train['kills']>0)&(train['totalDistance'] == 0))
# remove people who got kills without moving
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)


# We remove the players that have more than 40 kills. These players are probably cheaters since according to pubg.me leaderboards, most of these players came from very early releases of the game when it was still in beta. During this time there were more cheaters and PubG hasnt seen these types of of kill numbers since then.
train.drop(train[train['kills'] > 40].index, inplace=True)


# We now calculate the headshot rate of the players kills. Players who didn't get any headshots will be assigned the value 0. It is very suspect if a player has over 9 kills and a headshot rate of 100%
#people who have 100% headshot with over 9 kills
train['headshot_rate'] = train['headshotKills']/train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)
# display(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)
train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head(10)


# We display and the distance of players kills. We see most kills tend to be under 200 meters(?). We see a little spike after 1000m which is highly suspect. We remove those players.
plt.figure(figsize=(12,4))
sns.distplot(train['longestKill'], bins=10)
plt.savefig("LongestKill_Distrubution.png")
plt.show()

# drop the people who have kills from an insane distance. it makes no sense how they would get these kills. prob cheaters. 
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)


# We look at the type of matches in pubg. The most common are quad-fpp, duo-fpp, squad, solo-fpp, duo, and solo. fpp means "first-person" as the game has both third-person and first-person perspectives. We then have several other game modes that could be private lobbies and games that are different then the main pubg games. We plan on removing those.
print('There are {} different Match types in the dataset.'.format(train['matchType'].nunique()))
print(train['matchType'].value_counts())

# We turn groupId and matchId into categorical types in order for the ML algorithms to better process the data.
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')
train['groupId_cat'] = train['groupId'].cat.codes
train['matchId_cat'] = train['matchId'].cat.codes

train.drop(columns=['groupId', 'matchId'], inplace=True)

train[['groupId_cat', 'matchId_cat']].head()
train.drop(['Id','matchType'], axis=1, inplace=True)
train.head()
train.info()


# Machine Learning
# We will now do some machine learning on our data to predict the winPlacePerc. We will use Keras, Tensorflow and sklearn.
X = train.copy()
y = X['winPlacePerc']
X = X.drop(columns = ['winPlacePerc']) 
#X.shape
print("the size of the traning data is {0} x {1} and label size {2}".format(X.shape[0], X.shape[1], y.shape[0]))
#print("the size of the target is {0} ".format(y.shape))
y.head(10)


# We split our training data (80:20) to have a cross validation set to fine tune our algorithm without looking at the testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

X_train.info()


# We encode our labels so that the data is not continuous which causes problems for the ML algorithms.

from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)
# X_train_encoded = ohe.transform(X_train)
# X_test_encoded = ohe.transform(X_test)


# Models we no longer are useing have been commented out. We were going to run a basic logistic regression

# from sklearn.linear_model import LogisticRegression


# bin_clf = LogisticRegression().fit(X_train, y_train_encoded)

# from sklearn.model_selection import cross_val_score
# from sklearn.metric import mean_absolute_error

# bin_clf_predictions = cross_val_predict(bin_clf, X_train, y_train_encoded, cv=3)
# print('Classifier scores on training set: ')
# print('MAE is {0} '.format(mean_absolute_error(y_train, bin_clf_predictions)))

# #cross_val_score(bin_clf, X_train, y_train_encoded, cv=3, scoring="mean_squared_error")


# We run the Support Vector Machine algorithm on our data

# from sklearn.svm import SVC 

# svm_model_linear = SVC(kernel = 'rbf', C = 1, verbose=True).fit(X_train, y_train_encoded) 
# svm_predictions = svm_model_linear.predict(X_test) 


# from sklearn.model_selection import cross_val_score
# from sklearn.metric import mean_absolute_error

# svm_train_predictions = cross_val_predict(svm_model_linear, X_train, y_train_encoded, cv=3)
# print('Classifier scores on training set: ')
# print('MAE is {0} '.format(mean_absolute_error(y_train, svm_train_predictions)))

# cross_val_score(m1, X_train, y_train_encoded, cv=3, scoring="mean_squared_error")

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import cross_val_score


# rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
#                           n_jobs=-1, verbose=True).fit(X_train, y_train_encoded)
# #cross_val_score(m1, X_train, y_train_encoded, cv=3, scoring="mean_squared_error")


# from sklearn.model_selection import cross_val_score, cross_val_predict

# bin_clf_predictions = cross_val_predict(rf, X_train, y_train_encoded, cv=3)

# print('Classifier scores on training set: ')
# print('MAE is {0} '.format(mean_absolute_error(y_train_encoded, bin_clf_predictions)))


# The perceptron model written in keras' sequential API instead of SKLEARN

# from sklearn.neural_network import MLPClassifier

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.optimizers import SGD, Adam

# model = Sequential()

# model.add(Dense(64, activation='relu', input_shape=(34,)))
# #model.add(Flatten())
# # model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1, activation='sigmoid'))


# # sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss='mse',
#               optimizer=adam,
#               metrics=['mae'])

# history = model.fit(X_train, y_train, batch_size=32, epochs=10)

# print(history.history.keys())
# #  "Accuracy"
# plt.plot(history.history['mean_absolute_error'])
# #plt.plot(history.history['val_mean_absolute_error'])
# plt.title('model Mean Absolute Error')
# plt.ylabel('Mean Absolute Error')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig("result_1.png")
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# #plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig("result_2.png")
# plt.show()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  BatchNormalization

model2 = Sequential()

model2.add(Dense(32, input_shape=(X_train.shape[1],) , activation='selu'))
model2.add(Dense(64, activation='selu'))
model2.add(Dense(128, activation='selu'))
model2.add(Dropout(0.1))
model2.add(Dense(128, activation='selu'))
model2.add(BatchNormalization())
model2.add(Dense(64, activation='selu'))
model2.add(Dense(32,  activation='selu'))
model2.add(Dense(16, activation='selu'))
model2.add(BatchNormalization())
model2.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model2.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(X_train.shape)
print(X_test.shape)

history = model2.fit(X_train, y_train, batch_size=4, validation_data=(X_test, y_test), epochs=10) # , validation_steps=222269, steps_per_epoch=592717
model2.save('model.h5')

# load model
# model2 = load_model('model.h5')

# pred_mod =model2.predict_generator(validation_generator,verbose=1)

# predicted_class_indices=np.argmax(pred_mod,axis=1)

# labels = (X_train.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]

# print(history.history.keys())

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['val_mean_absolute_error'])
plt.title('model Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



test.info()

X_train.info()

test['playersInGame'] = test.groupby('matchId')['matchId'].transform('count')
test['killsNorm'] = test['kills']* (( 100-test['playersInGame']) / 100 + 1)
test['damageDealtNorm'] = test['damageDealt']* (( 100-test['playersInGame']) / 100 + 1)
test['maxPlaceNorm'] = test['maxPlace']* (( 100-train['playersInGame']) / 100 + 1)
test['matchDurationNorm'] = test['matchDuration']* (( 100-test['playersInGame']) / 100 + 1)

test['headshot_rate'] = test['headshotKills'] / test['kills']
test['headshot_rate'] = test['headshot_rate'].fillna(0)
test['totalDistance'] =  test['walkDistance'] + test['rideDistance'] + test['swimDistance']

test['killsWithoutMoving'] = ((( test['totalDistance'] == 0) & test['kills'] > 0))

test['groupId'] = test['groupId'].astype('category')
test['matchId'] = test['matchId'].astype('category')
test['groupId_cat'] = test['groupId'].cat.codes
test['matchId_cat'] = test['matchId'].cat.codes
test.drop(columns=['groupId', 'matchId'], inplace=True)
test.drop(['matchType'], axis=1, inplace=True)

test.info()

Id = pd.DataFrame({'Id' : test['Id']})

#test_pred = test[to_keep].copy()
test.drop(['Id'], axis=1, inplace=True)
# Fill NaN with 0 (temporary)
test.fillna(0, inplace=True)
test.head()

predictions = np.clip(a = model.predict(test), a_min = 0.0, a_max = 1.0)
print(predictions.shape)
print(Id['Id'].reshape([-1, 1]).shape)

pred_df = pd.DataFrame({'Id' : Id['Id'].reshape([-1]), 'winPlacePerc' : predictions.reshape([-1])})

# Create submission file
pred_df.to_csv("submission.csv", index=False)
