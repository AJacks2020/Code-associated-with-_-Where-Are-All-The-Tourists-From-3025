import math as m
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def transProb(state_index: int,
              transition_probs: list[float],
              occupancy_probs: list[float],
              step_duration: float
              ) -> float:
    '''
        Calculates the "quantity of probability" that moves to each neighbour
        from a given state in a single timestep 

        INPUTS
            state_index         : the state to calculate the transition probability of
            occupancy_probs     : the probability distribution of the Markov chain before the time step
            transition_probs    : the probability of each transition within a single step
            step_duration       : how long each step is/lasts

        OUTPUTS
            The "quantity of probability" of the state labelled with state_index in a single
            time step
    '''
    
    return transition_probs[state_index] * occupancy_probs[state_index] * step_duration 

@jit(nopython=True)
def updateSingleState(state_to_update: int,
                      occupancy_probs: list[float],
                      transition_probs: list[float],
                      step_duration: float
                      ) -> float:
    '''
        Updates the probability of being in a specific state after a single time step

        INPUTS
            state_to_update     : 
            occupancy_probs     : the probability distribution of the Markov chain before the time step
            transition_probs    : the probability of each transition within a single step
            step_duration       : how long each step is/lasts

        OUTPUTS
            The probability of being in the specified state after a
            single time step has been performed     
    '''

    # Treats 0-state differently due to its lack of neighbours below
    if state_to_update == 0:
        return occupancy_probs[0] + transProb(1,
                                              transition_probs,
                                              occupancy_probs,
                                              step_duration)

    # Treats the maximum state differently due to its lack of neighbours above
    max_state: int = len(transition_probs)- 1
    if state_to_update == max_state:
        return occupancy_probs[max_state] \
               - transProb(max_state,
                           transition_probs,
                           occupancy_probs,
                           step_duration) \
               + transProb(max_state - 1,
                           transition_probs,
                           occupancy_probs,
                           step_duration)

    # Treats all other states in the same way.
    # Updates the probability of the state being updated by adding the
    # probability of moving to that states and sibtracting the probability
    # of moving out of that state
    return occupancy_probs[state_to_update] \
           - 2.0 * transProb(state_to_update,
                             transition_probs,
                             occupancy_probs,
                             step_duration) \
           + transProb(state_to_update + 1,
                       transition_probs,
                       occupancy_probs,
                       step_duration) \
           + transProb(state_to_update - 1,
                       transition_probs,
                       occupancy_probs,
                       step_duration)

@jit(nopython=True)        
def singleStep(occupancy_probs: list[float],
               transition_probs: list[float],
               step_duration: float
               ) -> list[float]:
    '''
        Updates the probability distribution of the Markov chain specified to
        perform a single time step

        INPUTS
            occupancy_probs     : the probability distribution of the Markov chain
                                  before the time step
            transition_probs    : the probability of each transition within a
                                  single step
            step_duration       : how long each step is/lasts

        OUTPUTS
            The probability distribution of the Markov chain after a
            single time step has been performed     
    '''
        
    # Creates a empty list to hold the updated probabilities
    new_occupancy_probs: list[float] = []

    # Updates each occupancy probability based on the input occupancy
    # probabilities.
    for curr_index in range(len(occupancy_probs)):
        new_proba: float = updateSingleState(curr_index,
                                             occupancy_probs,
                                             transition_probs,
                                             step_duration)
        new_occupancy_probs.append(new_proba)
    
    return new_occupancy_probs


def fullSimulation(initial_state_index: int,
                   transition_probs: list[float],
                   step_duration: list[float],
                   simulation_duration: float
                   ) -> list[list[float]]:
    '''
        Peforms the full simulation required.

        INPUTS
            initial_state_index : the probability distribution to begin the simulation from
            transition_probs    : the probability of each transition within a single step
            step_duration       : how long each step is/lasts
            simulation_duration : how long the simulation is for, overall

        OUTPUTS
            A record of the probability distribution of the Markov chain at each step    
    '''

    # Initializes the state of the model
    occupancy_probs: list[float] = [0.0]*initial_state_index \
                                   + [1.0] \
                                   + [0.0]*(len(transition_probs) - initial_state_index - 1)

    # Initializes an array to store the full details of the simulation
    simulation_record:list[list[float]] = []

    # Calculates the required number of steps to implement in the simulation
    no_steps: int = m.ceil(simulation_duration / step_duration)

    # Updates each occupancy probability based on the input occupancy probabilities
    for _ in range(no_steps):
        simulation_record.append(occupancy_probs)
        occupancy_probs = singleStep(occupancy_probs, transition_probs, step_duration)
    simulation_record.append(occupancy_probs)

    return simulation_record

def plotSimulation(record: list[list[float]], state_to_plot: int, step_duration: float) -> None:
    '''
        Plots the probability of being in a specified state according to a provided
        record of a simulation

        INPUTS
            record        : a list of the probability distributions of each time
                            point in the simulation
            state_to_plot : the state in the Markov chain to plot the probabilities of being in
            step_duration : how long each step is/lasts


        OUTPUTS
            None

        SIDE EFFECTS
            Plots (but does not display) the probability of being in the
            specified state at each point in the simulation that the input record records        
    '''

    # Extracts the probability of being in the specified state at each recorded
    # point in time from the provided record of the simulation to plot
    state_record = [record_entry[state_to_plot] for record_entry in record]

    # Calculates the X-axis labels and then plots the specified probabilities
    time_ticks = [N * step_duration for N in range(len(record))]
    plt.plot(time_ticks, state_record, label="Construction number = " + str(state_to_plot))

@jit  
def genTransitions(num_states: int,
                   beta: float
                   ) -> list[float]:
    '''
        Peforms the full simulation required and plots the results.

        INPUTS
            num_states : how many states are in the Markov chain to simulate
            beta       : a scaling parameter of the 

        OUTPUTS
            A list of the transition probabilities generated according to the rule ...         
    '''
    return [beta * j /(2.0 * ((beta*j) + 1.0)) for j in range(num_states)]

def runAndPlotSimulation(num_states: int,
                         initial_state_index: int =None,
                         simulation_duration: float =None,
                         beta: float =1.0,
                         step_duration: float =0.1,
                         plot_modulo: int =None
                         ) -> None:
    '''
        Peforms the full simulation required and plots the results.

        INPUTS
            num_states          : how many states are in the Markov chain to simulate
            initial_state_index : the intial probability distribution 
            simulation_duration : how long the simulation is for, overall
            beta                : a scaling parameter of the 
            step_duration       : how long each step is/lasts

        OUTPUTS
            None

        SIDE EFFECTS
            Displays a plot of the simulation that has been run    
    '''

    # Just a trick to define default arguments in terms of other arguments
    if plot_modulo is None:
        plot_modulo = m.ceil(num_states / 4)
    if initial_state_index is None:
        initial_state_index = m.floor(num_states / 2)
    if simulation_duration is None:
        simulation_duration = m.ceil(num_states * 450)
        
    # Performs and records the required simulation
    sim = fullSimulation(initial_state_index,
                         genTransitions(num_states, beta),
                         step_duration,
                         simulation_duration)

    # Plots the probability of being in each state and records if the intial
    # state has been plotted
    intial_state_not_plotted: bool = True
    for k in range(m.ceil(len(sim[0]) / plot_modulo)):
        plotSimulation(sim, plot_modulo * k, step_duration)
        if plot_modulo * k == initial_state_index:
            intial_state_not_plotted = False

    #Plots the probability of being in the initial state, if have not already
    if intial_state_not_plotted == True:
        plotSimulation(sim, initial_state_index, step_duration)

    # Adds the labels and legend and displays the plotted probabilities
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Occupancy Probability")
    plt.show()


##### TESTING #####

if __name__ == "__main__":

    # Runs a test simulation, with default values of most parameters
    runAndPlotSimulation(10)
    


