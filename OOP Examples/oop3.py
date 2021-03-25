# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:33:32 2021

@author: hycbl
"""


class Person:
    number_of_people = 0
    
    def __init__(self, name):
        self.name = name
        Person.add_person()
        
    @classmethod
    def number_of_people_(cls):
        return cls.number_of_people
    
    @classmethod
    def add_person(cls):
        cls.number_of_people += 1
    
p1 = Person("dfg")

print(Person.number_of_people_())


###############################################

class Math:
    
    @staticmethod
    def add4(x):
        return x+5
    
print(Math.add4(6))
