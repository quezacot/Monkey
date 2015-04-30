import numpy.random as npr
import numpy as np
import sys
import cPickle as pickle
import os.path

from SwingyMonkey import SwingyMonkey

learnerfile = "learner.pckl"
#treebotmin, treebotmax = 10, 140
treetopmin, treetopmax = 210, 340
treedistmin, treedistmax = -115, 360 #485 only before first tree
monkeyvelmin, monkeyvelmax = -50, 30
#monkeybotmin, monkeybotmax = 0, 343
monkeytopmin, monkeytopmax = 57, 412
monkeysize = 57

class Learner:
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.treey = treetopmax - treetopmin
        self.treeyf = 20
        self.treey = self.treey / self.treeyf + 1
        self.treex = treedistmax - treedistmin
        self.treexf = 20
        self.treex = self.treex / self.treexf + 1
        self.monkeyv = monkeyvelmax - monkeyvelmin
        self.monkeyvf = 4
        self.monkeyv = self.monkeyv / self.monkeyvf + 1
        self.monkeyy = monkeytopmax - monkeytopmin
        self.monkeyyf = 20
        self.monkeyy = self.monkeyy / self.monkeyyf + 1
        # Q table
        print self.treey, self.treex, self.monkeyv, self.monkeyy, 2
        print self.treey* self.treex* self.monkeyv* self.monkeyy* 2
        self.Q = np.zeros([self.treey, self.treex, self.monkeyv, self.monkeyy, 2], dtype='float')
        # state visited
        self.K = np.zeros([self.treey, self.treex, self.monkeyv, self.monkeyy, 2], dtype='int')
        # iteration
        self.passed = 1
        self.ii = 0
        self.score = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # convert state to dimension of our Q table
    def state2dim(self, state):
        '''
        { 'score': <current score>,
          'tree': { 'dist': <pixels to next tree trunk>,
                    'top':  <screen height of top of tree trunk gap>,
                    'bot':  <screen height of bottom of tree trunk gap> },
          'monkey': { 'vel': <current monkey y-axis speed in pixels per iteration>,
                      'top': <screen height of top of monkey>,
                      'bot': <screen height of bottom of monkey> }}'''
        # truncate the values outside our states
        treey = cutoff(state['tree']['top'], treetopmax, treetopmin) - treetopmin
        treey = treey / self.treeyf
        treex = cutoff(state['tree']['dist'], treedistmax, treedistmin) - treedistmin
        treex = treex / self.treexf
        monkeyv = cutoff(state['monkey']['vel'], monkeyvelmax, monkeyvelmin) - monkeyvelmin
        monkeyv = monkeyv / self.monkeyvf
        monkeyy = cutoff(state['monkey']['top'], monkeytopmax, monkeytopmin) - monkeytopmin
        monkeyy = monkeyy / self.monkeyyf
        return treey, treex, monkeyv, monkeyy

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''
        if (state['score'] > self.score):
            self.score = state['score']
            print "high score", self.score

        # random action for the first iteration or far ahead the first tree
        if (self.last_state is None) or (self.last_state['tree']['dist'] > treedistmax) or (state['tree']['dist'] > treedistmax):
            new_action = npr.rand() < 0.1
            self.last_action = int(new_action)
            self.last_state  = state
            return self.last_action
        # You might do some learning here based on the current state and the last state.
        # update Q table
        treey, treex, monkeyv, monkeyy = self.state2dim(state)
        #print "max", state
        #print treey, treex, monkeyv, monkeyy
        Qmax = np.max(self.Q[treey, treex, monkeyv, monkeyy])

        treey, treex, monkeyv, monkeyy = self.state2dim(self.last_state)
        # update visited times
        #print "old", self.last_state
        #print treey, treex, monkeyv, monkeyy, self.last_action
        self.K[treey, treex, monkeyv, monkeyy, self.last_action] += 1
        alpha = 1.0 / self.K[treey, treex, monkeyv, monkeyy, self.last_action]
        Qold = self.Q[treey, treex, monkeyv, monkeyy, self.last_action]
        #print Qold
        self.Q[treey, treex, monkeyv, monkeyy, self.last_action] = Qold + alpha * ( self.last_reward + Qmax - Qold )

        # You'll need to take an action, too, and return it.
        new_action = self.choose_action(state)
        # Return 0 to swing and 1 to jump.
        self.last_action = new_action
        self.last_state = state
        return self.last_action

    # epsilon-greedy
    def choose_action(self, state):
        epsilon = 1.0 / self.passed
        amax = np.argmax(self.Q[self.state2dim(state)])
        return npr.choice([amax, 1-amax], p=[1.0 - epsilon/2.0, epsilon/2.0])

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if reward > 0:
            self.passed += 1
        self.last_reward = reward

def cutoff(value, maxv, minv):
    if value < minv:
        return minv
    if value > maxv:
        return maxv
    return value

iters = 1000
learner = Learner()
ii = 0
if os.path.exists(learnerfile):
    filehandler = open(learnerfile,'rb')
    learner = pickle.load(filehandler)
    print learner.score
    ii = learner.ii
#for ii in xrange(iters):
while True:
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    if ii%100 == 0:
        filehandler = open(learnerfile,"wb")
        pickle.dump(learner,filehandler)
        filehandler.close()

    # Reset the state of the learner.
    learner.reset()

    ii += 1
    learner.ii = ii




