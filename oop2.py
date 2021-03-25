# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:17:54 2021

@author: hycbl
"""


class Pet:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def show(self):
        print(f"I am {self.name} and I am {self.age} years old.")
        
class Cat(Pet):
    def meaw(self):
        print("meaw")
        
    def show(self):
        print("different because this is override")
        
class Dog(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color
     
    def show(self):
        print(f"I am {self.name} and I am {self.age} years old and {self.color}.")
        
    def bark(self):
        print("wang")
        
p = Pet("a", 5)
p.show()
c = Cat("c",6)
c.meaw()
c.show()
d = Dog("d", 8, "red")
d.bark()
d.show()