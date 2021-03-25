# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:55:17 2021

@author: hycbl
"""

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    def get_grade(self):
        return self.grade
    def set_grade(self, grade):
        self.grade = grade
        
class Course:
    def __init__(self, name, max_students):
        self.name = name
        self.max_students = max_students
        self.students = []
        self.isactive = False
        
    def add_student(self, student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        return False
    
    def get_average(self):
        value = 0
        for student in self.students:
            value += student.get_grade()
        return value/len(self.students)
        
    
s1 = Student("Sam", 20, 90)
s2 = Student("Wax", 20, 80)
s3 = Student("Abi", 20, 70)
s4 = Student("Den", 20, 85)
    
course1 = Course("Math", 3)

course1.add_student(s1)
course1.add_student(s2)
course1.add_student(s3)

print(course1.students[1].name)
print(course1.get_average())
print(course1.add_student(s4))




















