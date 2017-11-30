#!/usr/bin/env python3 
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import numba as nb


def y_dot(t,y,x,nu,F):
    """y_dot(t,y,x,nu,F)
         Describes the differential equation for velocity as given in CW 12.
    """
    return -(nu*y)+x-(x**3)+(F*np.cos(t))

def x_dot(y):
    """x_dot(y)
         Describes the differential equation for position as given in CW 12.
    """
    return y

def rk(a, b, x0, y0, nu=0, F=0, xdot = x_dot, ydot = y_dot):
    """rk(a, b, x0, y0, nu=0, F=0, xdot = x_dot, ydot = y_dot)
          input:
             a - float, beginning point of the time domain, such that t_initial = 2pi*a
             b - float, end point of time domain, such that t_final = 2pi*b
             x0 - float, initial position of the ball
             y0 - float, initial velocity of the ball
             nu - float, constant damping parameter
             F - float, constant force amplitude parameter
             xdot - function, part of the coupled differentiial equation.  The correct function
             for xdot is passed in to rk.
             ydot - function, part of the coupled differentiial equation.  The correct function
             for ydot is passed in to rk.
          output: (t,x_vec,y_vec)
             t - numpy array, with mesh spacing dt = 0.001
             x_vec - numpy array, contains the position of the ball at each time in the linspace
             y_vec - numpy array, contains the velocity of the ball at each time in the linspace
       All output numpy arrays are the same length, as determined by the input parameters 'a' and 'b'.
             """
    # Creating linspace
    dt = 0.001
    start = 2*a*np.pi
    end = 2*b*np.pi
    n = int(np.ceil((end-start)/dt))
    t = np.linspace(start,end,n)
    # Creating empty vectors
    x_vec = np.zeros(n)
    y_vec = np.zeros(n)
    x_dot_vec = np.zeros(n)
    y_dot_vec = np.zeros(n)
    # Filling initial points
    x_vec[0] = x0
    y_vec[0] = y0
    for k in range(n):
        # Filling dot vectors
        x_dot_vec[k] = x_dot(y_vec[k])
        y_dot_vec[k] = ydot(t[k],y_vec[k],x_vec[k],nu,F)
        # Filling position vectors
        if k == n-1:
            break
        else:
            K1y = dt*ydot(t[k],y_vec[k],x_vec[k],nu,F)
            K2y = dt*ydot((t[k]+dt/2),(y_vec[k]+K1y/2),x_vec[k],nu,F)
            K3y = dt*ydot((t[k]+dt/2),(y_vec[k]+K2y/2),x_vec[k],nu,F)
            K4y = dt*ydot((t[k]+dt),(y_vec[k]+K3y),x_vec[k],nu,F)
            RKy = (K1y+(2*K2y)+(2*K3y)+K4y)/6
            y_vec[k+1] = y_vec[k]+RKy

            K1x = dt*xdot(y_vec[k])
            K2x = dt*xdot(y_vec[k]+K1x/2)
            K3x = dt*xdot(y_vec[k]+K2x/2)
            K4x = dt*xdot(y_vec[k]+K3x)
            RKx = (K1x+(2*K2x)+(2*K3x)+K4x)/6
            x_vec[k+1] = x_vec[k]+RKx
    return (t,x_vec,y_vec)

def plot(x,t,title=''):
    """plot(x,t,title='')
       input:
          x - numpy array of length n, filled with the positions of the function at
              each time in t
          t - numpy array of length n, filled with the time domain
          title - string, title of the plot
       This function prints a plot to the screen with x on the x-axis, and t on the y-axis."""
    plt.rc('text', usetex=True)
    plt.plot(x,t,color="green")
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$t$', fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(title, fontsize = 20)
    plt.show()

def parametric_plot(x,y,title = ''):
    """parametric_plot(x,y,title = '')
       input:
          x - numpy array of length n, contains the position of the ball as it evolves over time
          y - numpy array of length n, contains the velocity of the ball as it evolves over time
          title - string, desired title of the graph
       This function prints a plot to the screen with x on the x-axis, and y on the y-axis."""
    plt.rc('text', usetex=True)
    plt.plot(x,y,color="green")
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$\dot{x}$', fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(title, fontsize = 20)
    plt.show()

def scatter_plot(x,y,n,title = ''):
    """scatter_plot(x,y,n,title = '')
       input:
          x - numpy array of length n, contains the position of the ball as it evolves over time
          y - numpy array of length n, contains the velocity of the ball as it evolves over time
          n - scalar, number of points whished to be plotted
          title - string, desired title of the graph
       This function prints a scatter plot to the screen.  The points plotted are for 2pi*k, where
       k is an integer included in [0,n-1].  Essentially, it plots the state of the function at the
       start of each period."""
    plt.rc('text', usetex=True)
    ax = plt.subplot
    plt.title(title, fontsize = 20)
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$\dot{x}$', fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    for k in range(n):
        plt.scatter(x[6283*k],y[6283*k],color='g')
    plt.show()