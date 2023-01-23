#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:31:20 2017
# https://www.reddit.com/r/Python/comments/6wf1uq/looking_for_help_using_scipyoptimise_fsolve_and/
@author: AaronR
"""
from math import sinh, cosh
from scipy.optimize import minimize_scalar
from scipy import integrate



def dist(x0, y0, x1, y1):
    # 2d Cartesian distance
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

def solve_catenary(x0, y0, x1, y1, length):
    """
    Given a catenary curve hanging between (x0, y0)
      and (x1, y1) with specified length,

    Return the curve parameters xm, ym, a
      such that (xm, ym) is the lowest point of the curve
      (not necessarily in the solution interval)
      and `a` is the scaling parameter.

    The curve equation is then
      y(x) = ym + a * (cosh((x - xm) / a) - 1.)

    Note: the `cosh() - 1.` means that whole subclause
      cancels out at x == xm, so the curve intersects (xm, ym)
    """
    # Solution method based on
    # https://en.wikipedia.org/wiki/Catenary#Determining_parameters

    # ensure a solution is possible
    if length < dist(x0, y0, x1, y1):
        raise ValueError("length is too short, no solution is possible", length, dist(x0, y0, x1, y1))

    # ensure that x0 is to the left of x1
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0

    # find vertical and horizontal reach
    v = y1 - y0
    h = x1 - x0   # note: h >= 0

    # solve to find `a`
    lhs = (length ** 2 - v ** 2) ** 0.5
    # rhs = 2 * a * sinh(h / (2 * a))
    # This is a transcendental function;
    #   we must solve numerically to find `a` such that lhs == rhs
    def err1(a):
        two_a = 2 * a
        rhs = two_a * sinh(h / two_a)
        return abs(rhs - lhs)
    # use bounded solver to force `a` >= 0
    a = minimize_scalar(err1, method="Bounded", bounds=(1e-5, 1e+5)).x
    
    # checks if a reached a boundry
    if abs(x1 - x0) < 1e-5 or abs(v / h) > 1e+5:
        xm = (x0 + x1) / 2
        ym = (y0 + y1 - length) / 2
        a = float("+inf")
        return xm, ym, a
    
    # Now that we have a solution for `a`, we can
    #   substitute back to find values for xm, ym:
    # ym = y0 - a * (cosh((x0 - xm) / a) - 1.)
    #    = y1 - a * (cosh((x1 - xm) / a) - 1.)
    def err2(xm):
        try:
            lhs = y0 - a * (cosh((x0 - xm) / a) - 1.)
            rhs = y1 - a * (cosh((x1 - xm) / a) - 1.)
            return abs(rhs - lhs)
        except OverflowError:
            print(x0, y0, x1, y1, length)
            print((x0 - xm) / a)
    
    xm = minimize_scalar(err2).x
    
#    print("Cat Error :", err2(xm))
    ym = y0 - a * (cosh((x0 - xm) / a) - 1.)
#    print("Inside Cat Z value 1: ", ym + a * (cosh((x0 - xm) / a) - 1))
#    print("It should be : ", y0)
#    print("Inside Cat Y value 1: ", x0)    
#    print("Inside Cat Z value 2: ", ym + a * (cosh((x1 - xm) / a) - 1))
#    print("It should be : ", y1)
#    print("Inside Cat Y value 2: ", x1)
    return xm, ym, a

def cat_func(x, xm, ym, a):
    
    return ym + a * (cosh((x - xm) / a) - 1.)


def catenary_length(x0, x1, a, xm):
    return abs(a * (sinh((x1 - xm) / a) - sinh((x0 - xm) / a)))
#print(solve_catenary(-3, -8, -2, -38, 50))


class CatenaryCurve(object):
    
    def __init__(self,p1, p2, length):
        self.p1 = p1
        self.p2 = p2
        
        self.i, self.j, self.a = solve_catenary(p1.y, p1.z, p2.y, p2.z, length)
        
        
    def arc_length(self,to_point=None):
        
        if self.a == float("+inf"):
            return self.p1.y + self.p2.y - 2 * self.j
        else:
            if to_point is None:
                to_point = self.p2
            
            return catenary_length(self.p1.y, to_point.y, self.a, self.i)
    
    #Function takes in a y and outputs a z. Should get a more descriptive name.
    def f(self, y, nothing=0):
        zcal = cat_func(y,self.i, self.j, self.a)
        return zcal