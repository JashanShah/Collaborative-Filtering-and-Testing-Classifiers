import sys
import pandas as pd
import glob
import os
import numpy as np
import nltk
import math
import sklearn
from sklearn.metrics.pairwise import cosine_similarity



def collaborativeFiltering():
    trainingData = pd.read_csv("C:/HW2/TrainingRatings.txt", sep = ',', names = ['MovieID', 'UserID', 'Rating'], dtype = {'UserID': 'int32', 'MovieID': 'int32', 'Rating': 'float32'})
    testingData = pd.read_csv("C:/HW2/TestingRatings.txt", sep = ',', names = ['MovieID', 'UserID', 'Rating'], dtype = {'UserID': 'int32', 'MovieID': 'int32', 'Rating': 'float32'})
    print(trainingData)
    print(testingData)

    theText = getTrainingData()
    dataArray = []  #For getting all the actual data into an array
    for i in theText.split('\n'):   #Gets every line
        temp = []   #Resets everytime
        for j in i.split(","):  #MovieID,UserID,Rating
            if not j:
                continue
            temp.append(j)
        if not temp:
            continue
        dataArray.append(temp)

    temp = []
    temp1 = []
    for i in dataArray:
        temp.append(int(i[0]))
        temp1.append(int(i[1]))

    userId = set(temp1)
    users = list(userId)
    users.sort()
    users = dict(zip(users, list(range(len(users)))))
    usersReview = dict(zip(list(range(len(users))), users))
    # print(user)
    lengthOfUser = len(users)


    movieId = set(temp) #Gets unique movie Ids
    mov = list(movieId)
    mov.sort()
    movies = dict(zip(mov, list(range(len(mov)))))
    movieReview = dict(zip(list(range(len(mov))), mov))
    #print(movie)
    lengthOfMovies = len(mov)

    dataset = np.zeros((lengthOfUser, lengthOfMovies), dtype='int')
    reviewedMovies = [[-1] * 1 for i in range(lengthOfUser)]

    for i in dataArray:
        movieID = int(i[0])
        userID = int(i[1])

        movieRating = int(float(i[2]))
        movieIndex = movies[movieID]
        userIndex = users[userID]
        dataset[userIndex][movieIndex] = movieRating

        if 0 != movieRating:
            if reviewedMovies[userIndex][0] != -1:
                reviewedMovies[userIndex].append(movieIndex)
            else:
                reviewedMovies[userIndex].append(movieIndex)
                reviewedMovies[userIndex].remove(-1)

    amtOfRatingsGiven = []
    lengthOfDataset = len(dataset)
    for i in range(lengthOfDataset):
        count = 0
        for j in range(len(dataset[i])):
            if dataset[i][j] != 0:
                count += 1
        amtOfRatingsGiven.append(count)
    meanOfRatingsNP = np.array([])  # Does not include 0s or moveis not rated.
    sumOfRatingsNP = np.array([])
    amtOfRatingsGivenNP = np.array(amtOfRatingsGiven)

    for i in range(lengthOfDataset):
        sumOfRatingsNP = np.append(sumOfRatingsNP, np.sum(dataset[i]))
    meanOfRatingsNP = sumOfRatingsNP / amtOfRatingsGivenNP

    centeredDataset = dataset





    #Testing

    theText = getTestingData()

    dataArrayTesting = []
    for i in theText.split('\n'):
        temp = []
        for j in i.split(","):
            if not j:
                continue
            temp.append(j)
        if not temp:
            continue
        dataArrayTesting.append(temp)

    temp = []
    temp1 = []
    for i in dataArrayTesting:
        temp1.append(int(i[1]))
        temp.append(int(i[0]))


    movieIdY = set(temp)
    movieY = list(movieIdY)
    movieY.sort()

    userIdY = set(temp1)
    userY = list(userIdY)
    lengthOfUserY = len(userY)
    userY.sort()
    usersY = dict(zip(userY, list(range(lengthOfUserY))))
    usersReviewY = dict(zip(list(range(lengthOfUserY)), userY))
    #print(userY)


    #print(movieY)
    #print(mov)

    for i in userY:
        if i not in users:
            print("User ID in test but not in training:")
            print(i)

    for i in movieY:
        if i not in mov:
            print("Movie ID in test but not in training:")
            print(i)
    needPrediction = [[-1] * 1 for i in range(lengthOfUserY)]
    needPredictionCounter = 0
    predictionY = np.zeros((lengthOfUserY, len(mov)), dtype='int')
    trueY = np.zeros((lengthOfUserY, len(mov)), dtype='int')
    lengthOfTrueY = len(trueY)

    for i in dataArrayTesting:
        movieID = int(i[0])
        userID = int(i[1])
        movieRating = int(float(i[2]))
        movieIndex = movies[movieID]
        userIndex = usersY[userID]

        if 0 != movieRating:
            needPredictionCounter += 1
            if (needPrediction[userIndex][0] != -1):
                needPrediction[userIndex].append(movieIndex)
            else:
                needPrediction[userIndex].append(movieIndex)
                needPrediction[userIndex].remove(-1)

        try:
            trueY[userIndex][movieIndex] = movieRating
        except Exception as e:
            print(e)
            continue


    needPredictionDict = dict(zip(userY, needPrediction))
    centeredY = trueY
    for i in range(lengthOfTrueY):
        mean = np.mean(trueY[i])
        for j in range(len(trueY[i])):
            if trueY[i][j] != 0:
                centeredY[i][j] = trueY[i][j] - mean

    weightsAI = sklearn.metrics.pairwise.cosine_similarity(centeredDataset, centeredY)
    lengthOfWeightsAI = len(weightsAI)
    knn = []
    for a in range(lengthOfWeightsAI):
        c = []  # Resets it after each iteration
        for i in range(len(weightsAI[a])):
            if weightsAI[a][i] > 0.18:
                c.append(i)
        knn.append(c)

    for temp in userY:
        a = users[temp]
        indexY = usersY[temp]
        for j in needPrediction[indexY]:
            k = np.sum(weightsAI[indexY])
            s = 0
            if k != 0:
                k = np.reciprocal(k)
            for i in knn[a]:
                t = np.multiply(k, weightsAI[i][j])
                c = dataset[i][j] - meanOfRatingsNP[i]
                s += np.multiply(t, c)
            ans = meanOfRatingsNP[a] + s
            try:
                predictionY[indexY][j] = ans
            except Exception as e:
                predictionY[indexY][j] = meanOfRatingsNP[a]

    #MAE
    calcAns = np.abs(predictionY - trueY)
    calcAns2 = np.sum(calcAns)
    theAns = np.sqrt(calcAns2 / needPredictionCounter)
    print(theAns)
    #RMSE
    calcAns = np.abs(np.square(predictionY) - np.square(trueY))
    calcAns2 = np.sum(calcAns)
    theAns = np.sqrt(calcAns2 / needPredictionCounter)
    print(theAns)


def getTrainingData():
    path = "C:/HW2/TrainingRatings.txt"
    f = open(path, "r")
    theText = f.read()
    return theText


def getTestingData():
    path = "C:/HW2/TestingRatings.txt"
    f = open(path, "r")
    theText = f.read()
    return theText


if __name__ == '__main__':
    #arg_list = sys.argv
    # path = os.getcwd()
    #path = str(arg_list[1])
    collaborativeFiltering()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
