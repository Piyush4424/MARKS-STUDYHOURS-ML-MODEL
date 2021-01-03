#importing the needed modules 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Importing the csv file
#since This  file is a txt file we need to convert it to csv .

#pf = pd.read_csv(r'C:\Users\carnivore\OneDrive\Documents\machine learning\stuscore.txt')
#pf.to_csv(r'C:\Users\carnivore\OneDrive\Documents\machine learning\stuscore.csv',index=None)


#storing the variable hours and Scores in a list .   
df = pd.read_csv(r'C:\Users\carnivore\OneDrive\Documents\machine learning\stuscore.csv')
list_hours = []
list_scores = []
for i in df['Hours']:
    list_hours.append(i)
for j in df['Scores']:
    list_scores.append(j)

#Training the model with the test variables (Hours and Scores) and finding the relationship between them using Linear Regression     
slope , intercept , r , p , std_err = stats.linregress(list_hours,list_scores)

def myfunc(list_hours):
    return slope * list_hours + intercept

mymodel = list(map(myfunc,list_hours))

plt.scatter(list_hours,list_scores)
plt.plot(list_hours,mymodel)
plt.show()

#checking the R-Square value 
print("R-square value :- %f"%(r))

#asking for the user input for predicting the value
x = float(input("Enter the number of hours for which you want the predict the scores"))
score_main = myfunc(x)
print(score_main)

#score_main = myfunc(9.25)
#print("Score when the time of study is 9.25 hours is %f"%(score_main))
    





