import random
import numpy
import math

values=[]
limit= 100000
n=0 

while n <limit:
    x = random.uniform(0,math.pi/4)
    numerator= x*(math.sin(x))**2
    denomenator = math.cos(x)

    eq = numerator/denomenator

    curr_est = (math.pi/4)*eq

    values.append(curr_est)
    n+=1

estimate = sum(values)/limit

print(estimate)

def monte_carlo_integration(limit):
    initial_result=0
    values=[]

    while n<limit:
        x = random.uniform(0,math.pi/4)
        numerator = x*(math.sin)

def monte_carlo_pi(num_samples):

    count_inside=0
    for i in range(num_samples):
        x,y=random.uniform(0,1), random.uniform(0,1)
        if (x**2 + y**2)<=1:
            count_inside+=1
    
    estimate = 4 * (count_inside/num_samples)

    return estimate


estimate= monte_carlo_pi(1000000)
print(estimate)