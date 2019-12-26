'''
 Optimización de una máquina de soporte vectorial con un algoritmo genético
'''
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.read_csv("ENB2012_data.csv")

# shuffle data randomly
data = data.sample(frac = 1)

# Independent variables
X1 = pd.DataFrame(data, columns = ['X1','X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])

# Dependent variable
Y = pd.DataFrame(data, columns = ['Y1']).values

# 'X6' and 'X8' are categorical variables, so we generate dummy variables
xbef = pd.get_dummies(X1, columns = ['X6','X8'])

# Scaling the variables between 0 and 1

min_max_scalar = preprocessing.MinMaxScaler() # Creating the object
X = min_max_scalar.fit_transform(xbef) # Reescaling the variables (returns an np array)

# Getting the number of datapoints
Cnt1 = len(X) # Prints 768 (number of samples in the data frame )

# Getting started with the Genetic Algorithm GA

# Parameters of the GA
p_c = 1 # Probability of the crossOver
p_m = 0.2 # Probability of the mutation
pop = 100 # Size of the population
gen = 50 # Generations
kfold = 3 #

# first 15 bits represent the parameter Gamma and the others the parameter C
XY0 = np.array([1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,0])

n_list = np.empty((0,len(XY0)))

# Creating the population randomly
for i in range(pop):
    rd.shuffle(XY0)
    n_list = np.vstack((n_list, XY0))

# Calculating the precision
# X = C
a_X = 10
b_X = 1000
l_X = (len(XY0)/2)

# Y = Gamma
a_Y = 0.05
b_Y = 0.99
l_Y = (len(XY0)/2)

# Calculating the precision
precision_X = (b_X - a_X)/((2**l_X)-1) # Precision for the parameter C
precision_Y = (b_Y - a_Y)/((2**l_Y)-1) # Precision for the parameter Gamma

# Decoding the parameter C
z = 0
t = 1
X0_num_Sum = 0

for i in range(len(XY0)//2):
    x0_num = XY0[-t] * (2 ** z)
    X0_num_Sum+=x0_num
    t = t+1
    z = z+1

# Decoding the parameter Gamma
p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum = 0

for i in range(len(XY0)//2):
    y0_sum = XY0[-u] * (2 ** p)
    Y0_num_Sum += y0_sum
    u = u+1
    p = p+1

# Getting the decoded values
decodedX = (X0_num_Sum * precision_X) + a_X
decodedY = (Y0_num_Sum * precision_Y) + a_Y

print(decodedX)
print(decodedY)

# Getting started with THE algorithm
final_best_in_generation_X = []
worst_best_in_generation_X = []

one_final_guy = np.empty((0, len(XY0) + 2))
one_final_guy_final = []

min_for_all_generations_for_mut1 = np.empty((0, len(XY0) + 1))
min_for_all_generations_for_mut2 = np.empty((0, len(XY0) + 1))

min_for_all_generations_for_mut1_1 = np.empty((0, len(XY0) + 2))
min_for_all_generations_for_mut2_2 = np.empty((0, len(XY0) + 2))

min_for_all_generations_for_mut1_1_1 = np.empty((0, len(XY0) + 2))
min_for_all_generations_for_mut2_2_2 = np.empty((0, len(XY0) + 2))

generation = 1

for i in range(gen):

    new_population = np.empty((0, len(XY0)))

    all_in_generation_x_1 = np.empty((0, len(XY0)+1))
    all_in_generation_x_2 = np.empty((0, len(XY0)+1))

    min_in_generation_x1 = []
    min_in_generation_x2 = []

    save_best_in_generation_x = np.empty((0, len(XY0) + 1))
    final_best_in_generation_X = []
    worst_best_in_generation_X = []

    print("Generation: #", generation)

    family = 1
    for j in range(int(pop / 2)):
        print("Family: ", family)

        parents = np.empty((0, len(XY0)))

        for i in range(2):

            battle_troops = []
            warrior_1_index = np.random.randint(0, len(n_list))
            warrior_2_index = np.random.randint(0, len(n_list))
            warrior_3_index = np.random.randint(0, len(n_list))

            # Esta mujer no sabe programar en lo  mínimo
            while warrior_1_index == warrior_2_index:
                warrior_1_index = np.random.randint(0, len(n_list))
            while warrior_2_index == warrior_3_index:
                warrior_3_index = np.random.randint(0, len(n_list))
            while warrior_1_index == warrior_3_index:
                warrior_3_index = np.random.randint(0, len(n_list))

            warrior_1 = n_list[warrior_1_index]
            warrior_2 = n_list[warrior_2_index]
            warrior_3 = n_list[warrior_3_index]

            # Warrior 1

            # Decoding the parameter C
            z = 0
            t = 1
            X0_num_Sum_w1 = 0

            for i in range(len(XY0)//2):
                x0_num_w1 = warrior_1[-t] * (2 ** z)
                X0_num_Sum_w1+=x0_num_w1
                t = t+1
                z = z+1

            # Decoding the parameter Gamma
            p = 0
            u = 1 + (len(XY0)//2)
            Y0_num_Sum_w1 = 0

            for i in range(len(XY0)//2):
                y0_sum_w1 = warrior_1[-u] * (2 ** p)
                Y0_num_Sum_w1 += y0_sum_w1
                u = u+1
                p = p+1

            # Getting the decoded values
            decodedX_w1 = (X0_num_Sum_w1 * precision_X) + a_X
            decodedY_w1 = (Y0_num_Sum_w1 * precision_Y) + a_Y

            # Testing the parameters C and Gamma in the SVM
            p_1 = 0
            kf = cross_validation.KFold(Cnt1, n_folds = kfold)
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                model1 = svm.SVR(kernel = "rbf", C = decodedX_w1, gamma = decodedY_w1)
                model1.fit(X_train, Y_train)
                pl1 = model1.predict(X_test)

                ac1 = model1.score(X_test, Y_test)
                of_so_far_1 = 1 - ac1
                p_1 += of_so_far_1
            of_so_far_w1 = p_1/kfold

            # Warrior 2

            # Decoding the parameter C
            z = 0
            t = 1
            X0_num_Sum_w2 = 0

            for i in range(len(XY0)//2):
                x0_num_w2 = warrior_2[-t] * (2 ** z)
                X0_num_Sum_w2+=x0_num_w2
                t = t+1
                z = z+1

            # Decoding the parameter Gamma
            p = 0
            u = 1 + (len(XY0)//2)
            Y0_num_Sum_w2 = 0

            for i in range(len(XY0)//2):
                y0_sum_w2 = warrior_2[-u] * (2 ** p)
                Y0_num_Sum_w2 += y0_sum_w2
                u = u+1
                p = p+1

            # Getting the decoded values
            decodedX_w2 = (X0_num_Sum_w2 * precision_X) + a_X
            decodedY_w2 = (Y0_num_Sum_w2 * precision_Y) + a_Y

            # Testing the parameters C and Gamma in the SVM
            p_2 = 0
            kf = cross_validation.KFold(Cnt1, n_folds = kfold)
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                model2 = svm.SVR(kernel = "rbf", C = decodedX_w2, gamma = decodedY_w2)
                model2.fit(X_train, Y_train)
                pl2 = model2.predict(X_test)

                ac2 = model2.score(X_test, Y_test)
                of_so_far_2 = 1 - ac2
                p_2 += of_so_far_2
            of_so_far_w2 = p_2/kfold
