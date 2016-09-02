#!/usr/bin/env python

__author__ = 'João Quintas'

# import sys, os, csv
# import cv2
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    debug = True


    # State transition function from POMDP model, defines each element of T(s',a,s) as P(s'|s,a)
    # a = axis 0; s' = axis 1; s = axis 2
    T = np.array([])

    # Observation function from POMDP model, defines each element of O(s,a,o) as P(o|s,a)
    # a = axis 0; o = axis 1; s = axis 2
    O = np.array([])

    # initialization (we start by assuming we did not detect any person yet)
    # for the two states in our example model S = {person_detected; person_not_detected}
    P_person_detected = 0
    P_person_not_detected = 1 - P_person_detected

    # load value functions vectors from pomdp-solve (.alpha file)
    V = np.array([[-249.99, -249.99],
                  [-107.78, -349.27]])

    Action = {0 : 'action_haar' , 1 : 'action_check_lights'} # with the numbers corresponding to the index of max V

    epoc = 0
    horizon = 100

    while(epoc < horizon):

        # control choice
        Value_belief = V[:,0]*P_person_detected + V[:,1]

        if debug:
            print(Value_belief.argmax())
            print(Action[Value_belief.argmax()])


        # sensing
        P_person_detected_observation = np.array([0.25, 0.25, 0.5])
        P_person_detected_posterior = P_person_detected_observation*P_person_detected

        if debug:
            print(P_person_detected_posterior)


        # prediction (update belief)
        # todo: put as array
        belief_update_s0 = O(a,o,0)*np.sum(T,axis=2)
        belief_update_s1 = O()



        # update time instant
        epoc = epoc+1



    # # creates xx with values between 0 and 1 with step 0.1 (belief space axis)
    # xx = np.linspace(0,1,11)
    #
    # m = -3.43
    # b = -1.43
    #
    # # calculate corresponding y
    # y = (m * xx) + b
    #
    #
    # m2 = 13.09
    # b2 = -6.55
    # y2 = (m2 * xx) + b2
    #
    #
    # m3 = 18.46
    # b3 = -11.77
    # y3 = (m3 * xx) + b3
    #
    #
    # # plot the surface
    # pyplot.plot(xx, y, 'r', xx, y2, 'g', xx, y3, 'b')
    # pyplot.axis([0,1,-5,20])
    # pyplot.show()