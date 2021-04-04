# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:27:13 2021
one liner
@author: hycbl
"""
q = lambda l: q( [x for x in l[1:] if x <= l[0]]) + [l[0]] + q([x for x in l if x > l[0]]) if l else []

#Quick sort


lst = [x**2 for x in range(10)]

whales = [x for x,y in  [("John", 240000), ("Alice", 120000), ("Ann", 1100000), ("Zach", 44000)] if y>1000000]

[x * 2 for x in range(3)]

print([(x, y) for x in range(3) for y in range(3)])

print([x ** 2 for x in range(10) if x % 2 > 0])

print([x.lower() for x in ['I', 'AM', 'NOT', 'SHOUTING']])

## Data
employees = {'Alice' : 100000,
 'Bob' : 99817,
 'Carol' : 122908,
 'Frank' : 88123,
 'Eve' : 93121}
## One-Liner
top_earners = [(k, v) for k, v in employees.items() if v >= 100000]
## Result
print(top_earners)

## Data
text = '''
Call me Ishmael. Some years ago - never mind how long precisely - having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part
of the world. It is a way I have of driving off the spleen, and regulating
the circulation. - Moby Dick'''
## One-Liner
w = [[x for x in line.split() if len(x)>3] for line in text.split('\n')]
## Result
print(w)

print([line.strip() for line in open("readFile.py")])     #open file

## Data
txt = ['lambda functions are anonymous functions.',
 'anonymous functions dont have a name.',
 'functions are objects in Python.']
## One-Liner
mark = map(lambda s: (True, s) if 'anonymous' in s else (False, s), txt)
## Result
print(list(mark))

## Data
letters_amazon = '''
We spent several years building our own database engine,
Amazon Aurora, a fully-managed MySQL and PostgreSQL-compatible
service with the same or better durability and availability as
the commercial engines, but at one-tenth of the cost. We were
not surprised when this worked.
'''
## One-Liner
find = lambda x, q: x[x.find(q)-18:x.find(q)+18] if q in x else -1
## Result
print(find(letters_amazon, 'SQL'))

## Data (daily stock prices ($))
price = [[9.9, 9.8, 9.8, 9.4, 9.5, 9.7],
 [9.5, 9.4, 9.4, 9.3, 9.2, 9.1],
 [8.4, 7.9, 7.9, 8.1, 8.0, 8.0],
 [7.1, 5.9, 4.8, 4.8, 4.7, 3.9]]
## One-Liner
sample = [line[::2] for line in price]
## Result
print(sample)

## Data
visitors = ['Firefox', 'corrupted', 'Chrome', 'corrupted',
 'Safari', 'corrupted', 'Safari', 'corrupted',
 'Chrome', 'corrupted', 'Firefox', 'corrupted']
## One-Liner
visitors[1::2] = visitors[::2]
## Result
print(visitors)

## Dependencies
import matplotlib.pyplot as plt
## Data
cardiac_cycle = [62, 60, 62, 64, 68, 77, 80, 76, 71, 66, 61, 60, 62]
## One-Liner
expected_cycles = cardiac_cycle[1:-2] * 10
## Result
plt.plot(expected_cycles)
plt.show()

## Data
companies = {
 'CoolCompany' : {'Alice' : 33, 'Bob' : 28, 'Frank' : 29},
 'CheapCompany' : {'Ann' : 4, 'Lee' : 9, 'Chrisi' : 7},
 'SosoCompany' : {'Esther' : 38, 'Cole' : 8, 'Paris' : 18}}
## One-Liner
illegal = [x for x in companies if any(y<9 for y in companies[x].values())]
## Result
print(illegal)

## Data
column_names = ['name', 'salary', 'job']
db_rows = [('Alice', 180000, 'data scientist'),
 ('Bob', 99000, 'mid-level manager'),
 ('Frank', 87000, 'CEO')]
## One-Liner
db = [dict(zip(column_names, row)) for row in db_rows]
## Result
print(db)

import numpy as np
## Data: yearly salary in ($1000) [2017, 2018, 2019]
alice = [99, 101, 103]
bob = [110, 108, 105]
tim = [90, 88, 85]
salaries = np.array([alice, bob, tim])
taxation = np.array([[0.2, 0.25, 0.22],
 [0.4, 0.5, 0.5],
 [0.1, 0.2, 0.1]])
## One-liner
max_income = np.max(salaries - salaries * taxation) 
## Result
print(max_income)

import numpy as np
## Data: yearly salary in ($1000) [2025, 2026, 2027]
dataScientist = [130, 132, 137]
productManager = [127, 140, 145]
designer = [118, 118, 127]
softwareEngineer = [129, 131, 137]
employees = np.array([dataScientist,
 productManager,
 designer,
 softwareEngineer])
## One-liner
employees[0,::2] = employees[0,::2] * 1.1
## Result
print(employees)

## Dependencies
import numpy as np
## Data: air quality index AQI data (row = city)
X = np.array(
 [[ 42, 40, 41, 43, 44, 43 ], # Hong Kong
 [ 30, 31, 29, 29, 29, 30 ], # New York
 [ 8, 13, 31, 11, 11, 9 ], # Berlin
 [ 11, 11, 12, 13, 11, 12 ]]) # Montreal
cities = np.array(["Hong Kong", "New York", "Berlin", "Montreal"])
## One-liner
polluted = set(cities[np.nonzero(X > np.average(X))[0]])         
# finds the cities with above-average observed AQI values
## Result
print(polluted)

## Dependencies
import numpy as np
## Data: popular Instagram accounts (millions followers)
inst = np.array([[232, "@instagram"],
 [133, "@selenagomez"],
 [59, "@victoriassecret"],
 [120, "@cristiano"],
 [111, "@beyonce"],
 [76, "@nike"]])
## One-liner
superstars = inst[inst[:,0].astype(float) > 100, 1]
#find the names of the Instagram superstars with more than 100 million followers
## Results
print(superstars)

## Dependencies
import numpy as np
## Sensor data (Mo, Tu, We, Th, Fr, Sa, Su)
tmp = np.array([1, 2, 3, 4, 3, 4, 4,
 5, 3, 3, 4, 3, 4, 6,
 6, 5, 5, 5, 4, 5, 5])
## One-liner
tmp[6::7] = np.average(tmp.reshape((-1,7)), axis=1)
## Result
print(tmp)
#given an array of temperature values, replace every seventh temperature value with the average of the last seven days (including the seventh day’s temperature value)

## Dependencies
import numpy as np
## Data: SAT scores for different students
sat_scores = np.array([1100, 1256, 1543, 1043, 989, 1412, 1343])
students = np.array(["John", "Bob", "Alice", "Joe", "Jane", "Frank", "Carl"])
## One-liner
top_3 = students[np.argsort(sat_scores)][:-4:-1]
## Result
print(top_3)

## Dependencies
import numpy as np
## Data (row = [title, rating])
books = np.array([['Coffee Break NumPy', 4.6],
 ['Lord of the Rings', 5.0],
 ['Harry Potter', 4.3],
 ['Winnie-the-Pooh', 3.9],
 ['The Clown of God', 2.2],
 ['Coffee Break Python', 4.7]])
## One-liner
predict_bestseller = lambda x, y : x[x[:,1].astype(float) > y]
## Results
print(predict_bestseller(books, 3.9))
#create a filter function that takes a list of books x and a minimum rating y and returns a list of potential bestsellers that have higher than minimum rating, y'>y.

#all outlier days for which the statistics deviate more than the standard deviation from their mean statistics.
## Dependencies
import numpy as np
## Website analytics data:
## (row = day), (col = users, bounce, duration)
a = np.array([[815, 70, 115],
 [767, 80, 50],
 [912, 74, 77],
 [554, 88, 70],
 [1008, 65, 128]])
mean, stdev = np.mean(a, axis=0), np.std(a, axis=0)
# [811.2 76.4 88. ], [152.97764543 6.85857128 29.04479299]
## One-liner
outliers = ((np.abs(a[:,0] - mean[0]) > stdev[0])
 * (np.abs(a[:,1] - mean[1]) > stdev[1])
 * (np.abs(a[:,2] - mean[2]) > stdev[2]))
## Result
print(a[outliers])

## Dependencies
import numpy as np
## Data: row is customer shopping basket
## row = [course 1, course 2, ebook 1, ebook 2]
## value 1 indicates that an item was bought.
basket = np.array([[0, 1, 1, 0],
 [0, 0, 0, 1],
 [1, 1, 0, 0],
 [0, 1, 1, 1],
 [1, 1, 1, 0],
 [0, 1, 1, 0],
 [1, 1, 0, 1],
 [1, 1, 1, 1]])
## One-liner
copurchases = np.sum(np.all(basket[:,2:], axis = 1)) / basket.shape[0]
## Result
print(copurchases)
#what fraction of customers bought two ebooks together

## Data: row is customer shopping basket
## row = [course 1, course 2, ebook 1, ebook 2]
## value 1 indicates that an item was bought.
basket = np.array([[0, 1, 1, 0],
                   [0, 0, 0, 1],
                   [1, 1, 0, 0],
                   [0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [0, 1, 1, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 1]])

## One-liner
copurchases = [(i,j,np.sum(basket[:,i] + basket[:,j] == 2)) for i in range(4) for j in range(i+1,4)]

## Result
print(max(copurchases, key=lambda x:x[2]))

print([(i,j) for i in range(4) for j in range(i+1,4)])

#machine learning
from sklearn.linear_model import LinearRegression
import numpy as np
## Data (Apple stock prices)
apple = np.array([155, 156, 157])
n = len(apple)
## One-liner
model = LinearRegression().fit(np.arange(n).reshape((n,1)), apple)
## Result & puzzle
print(model.predict([[3],[4]]))

from sklearn.linear_model import LogisticRegression
import numpy as np
## Data (#cigarettes, cancer)
X = np.array([[0, "No"],
 [10, "No"],
 [60, "Yes"],
 [90, "Yes"]])
## One-liner
model = LogisticRegression().fit(X[:,0].reshape(-1,1), X[:,1])
## Result & puzzle
print(model.predict([[2],[12],[13],[40],[90]]))

## Dependencies
from sklearn.cluster import KMeans
import numpy as np
## Data (Work (h) / Salary ($))
X = np.array([[35, 7000], [45, 6900], [70, 7100],
 [20, 2000], [25, 2200], [15, 1800]])
## One-liner
kmeans = KMeans(n_clusters=2).fit(X)
## Result & puzzle
cc = kmeans.cluster_centers_
print(cc)

## Dependencies
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
## Data (House Size (square meters) / House Price ($))
X = np.array([[35, 30000], [45, 45000], [40, 50000],
 [35, 35000], [25, 32500], [40, 40000]])
## One-liner
KNN = KNeighborsRegressor(n_neighbors=3).fit(X[:,0].reshape(-1,1), X[:,1])
## Result & puzzle
res = KNN.predict([[30]])
print(res)

## Dependencies
from sklearn.neural_network import MLPRegressor
import numpy as np
## Questionaire data (WEEK, YEARS, BOOKS, PROJECTS, EARN, RATING)
X = np.array(
 [[20, 11, 20, 30, 4000, 3000], 
 [12, 4, 0, 0, 1000, 1500],
 [2, 0, 1, 10, 0, 1400],
 [35, 5, 10, 70, 6000, 3800],
 [30, 1, 4, 65, 0, 3900],
 [35, 1, 0, 0, 0, 100],
 [15, 1, 2, 25, 0, 3700],
 [40, 3, -1, 60, 1000, 2000],
 [40, 1, 2, 95, 0, 1000],
 [10, 0, 0, 0, 0, 1400],
 [30, 1, 0, 50, 0, 1700],
 [1, 0, 0, 45, 0, 1762],
 [10, 32, 10, 5, 0, 2400],
 [5, 35, 4, 0, 13000, 3900],
 [8, 9, 40, 30, 1000, 2625],
 [1, 0, 1, 0, 0, 1900],
 [1, 30, 10, 0, 1000, 1900],
 [7, 16, 5, 0, 0, 3000]])
## One-liner
neural_net = MLPRegressor(max_iter=10000).fit(X[:,:-1], X[:,-1]) 
## Result
res = neural_net.predict([[0, 0, 0, 0, 0]])
print(res)

## Dependencies
from sklearn import tree
import numpy as np
## Data: student scores in (math, language, creativity) --> study field
X = np.array([[9, 5, 6, "computer science"],
 [1, 8, 1, "linguistics"],
 [5, 7, 9, "art"]])
## One-liner
Tree = tree.DecisionTreeClassifier().fit(X[:,:-1], X[:,-1])
## Result & puzzle
student_0 = Tree.predict([[8, 6, 5]])
print(student_0)
student_1 = Tree.predict([[3, 7, 9]])
print(student_1)

## Dependencies
import numpy as np
## Data (rows: stocks / cols: stock prices)
X = np.array([[25,27,29,30],
 [1,5,3,2],
 [12,11,8,3],
 [1,1,2,2],
 [2,6,2,2]])
## One-liner
# Find the stock with smallest variance
min_row = min([(i,np.var(X[i,:])) for i in range(len(X))], key=lambda x: x[1])
## Result & puzzle
print("Row with minimum variance: " + str(min_row[0]))
print("Variance: " + str(min_row[1]))

## Dependencies
import numpy as np
## Stock Price Data: 5 companies
# (row=[price_day_1, price_day_2, ...])
x = np.array([[8, 9, 11, 12],
 [1, 2, 2, 1], 
 [2, 8, 9, 9],
 [9, 6, 6, 3],
 [3, 3, 3, 3]])
## One-liner
avg, var, std = np.average(x, axis=1), np.var(x, axis=1), np.std(x, axis=1)
## Result & puzzle
print("Averages: " + str(avg))
print("Variances: " + str(var))
print("Standard Deviations: " + str(std))

## Dependencies
from sklearn import svm
import numpy as np
## Data: student scores in (math, language, creativity) --> study field
X = np.array([[9, 5, 6, "computer science"],
 [10, 1, 2, "computer science"],
 [1, 8, 1, "literature"],
 [4, 9, 3, "literature"],
 [0, 1, 10, "art"],
 [5, 7, 9, "art"]])
## One-liner
svm = svm.SVC().fit(X[:,:-1], X[:,-1])
## Result & puzzle
student_0 = svm.predict([[3, 3, 6]])
print(student_0)
student_1 = svm.predict([[8, 1, 1]])
print(student_1)

## Dependencies
import numpy as np
from sklearn.ensemble import RandomForestClassifier
## Data: student scores in (math, language, creativity) --> study field
X = np.array([[9, 5, 6, "computer science"],
 [5, 1, 5, "computer science"],
 [8, 8, 8, "computer science"],
 [1, 10, 7, "literature"],
 [1, 8, 1, "literature"],
 [5, 7, 9, "art"],
 [1, 1, 6, "art"]])
## One-liner
Forest = RandomForestClassifier(n_estimators=10).fit(X[:,:-1], X[:,-1])
## Result
students = Forest.predict([[8, 6, 5],
 [3, 7, 9],
 [2, 2, 1]])
print(students)


## The Data
n = 5
## The One-Liner
factorial = lambda n: n * factorial(n-1) if n > 1 else 1
## The Result
print(factorial(n))

## The Data
a = "cat"
b = "chello"
c = "chess"
## The One-Liner
ls = lambda a, b: len(b) if not a else len(a) if not b else min(
   ls(a[1:], b[1:])+(a[0] != b[0]), 
   ls(a[1:], b)+1, 
   ls(a, b[1:])+1) 
## The Result
print(ls(a,b))
print(ls(a,c))
print(ls(b,c))
# calculates the Levenshtein distance of strings

#power set
# Dependencies
from functools import reduce
# The Data
s = {1, 2, 3}
# The One-Liner
ps = lambda s: reduce(lambda P, x: P + [subset | {x} for subset in P], s, [set()])
# The Result
print(ps(s))

## Data
abc = "abcdefghijklmnopqrstuvwxyz"
s = "xthexrussiansxarexcoming"
## One-Liner
rt13 = lambda x: "".join([abc[(abc.find(c) + 13) % 26] for c in x])
## Result
print(rt13(s))
print(rt13(rt13(s)))
#encrypting string s with the ROT13 algorithm

## Dependencies
#from functools import reduce
## The Data
n=100
## The One-Liner
primes = reduce(lambda r, x: r - set(range(x**2, n, x)) if x in r else r,
 range(2, int(n**0.5) + 1), set(range(2, n)))
## The Result
print(primes)

# Dependencies
from functools import reduce
# The Data
n = 10
# The One-Liner
fibs = reduce(lambda x, _: x + [x[-2] + x[-1]], [0] * (n-2), [0, 1])
# The Result
print(fibs)

#binary search
## The Data
l = [3, 6, 14, 16, 33, 55, 56, 89]
x = 33
## The One-Liner
bs = lambda l, x, lo, hi: -1 if lo>hi else(lo+hi)//2 if l[(lo+hi)//2] == x else bs(l, x, lo, (lo+hi)//2-1) if l[(lo+hi)//2] > x else bs(l, x, (lo+hi)//2+1, hi) 
## The Results
print(bs(l, x, 0, len(l)-1))


