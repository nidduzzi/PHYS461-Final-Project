

class State:
    def __init__(self, dr) -> None:
        ## Independent Variables
        # Radius
        
        # TODO: Time

        ## Dependent Variables 
        # Density
        self.rh:np.Array
        # Temperature
        self.T
        # Pressure
        self.p

    def getCurrentState(self):
        pass

    def getTime(self):
        pass

    