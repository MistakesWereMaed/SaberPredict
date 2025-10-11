from enum import Enum

class ActionID(Enum):
    # Direct Action
    PREPERATION     = 10
    ATTACK_SIMPLE   = 11
    ATTACK_COMPOUND = 12
    REMISE          = 13
    RIPPOSTE        = 14
    STOP_CUT        = 15
    POINT_IN_LINE   = 16
    # Bladework
    PARRY           = 20
    BEAT            = 21
    # Other
    NO_ACTION       = 30
    DISTANCE_PULL   = 31

class FencerID(Enum):
    LEFT = 0
    RIGHT = 1

class Action:
    def __init__(self, label: ActionID, blade_contact: FencerID, hit: bool, start: float, end: float):
        self.label = label
        self.blade_contact = blade_contact
        self.hit = hit
        self.start = start
        self.end = end

class Fencer:
    def __init__(self):
        self.actions: list[Action] = []

    def reset(self):
        self.actions.clear()

    def update_action(self, new_action: Action):
        self.actions.append(new_action)