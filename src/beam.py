from typing import Generic


class AbstractBeamSearch:
    state = None
    forward = None
    beam = []

    __init__(self, forward, initial_state = None):
        self.state = initial_state
        self.forward = forward

    
