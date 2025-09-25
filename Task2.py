import numpy as np
import matplotlib.pyplot as plt


def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h


def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h


def grad(f, x, y, h):
    return ddx(f, x, y, h), ddy(f, x, y, h)

