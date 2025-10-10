from enum import Enum

class ActionID(Enum):
    # Offensive
    PREPERATION     = 10
    ATTACK_SIMPLE   = 11
    ATTACK_COMPOUND = 12
    ATTACK_BEAT     = 13
    REMISE          = 14
    # Defensive
    PARRY           = 20
    RIPPOSTE        = 21
    STOP_CUT        = 22
    DISTANCE_PULL   = 23
    POINT_IN_LINE   = 24
    # Other
    NO_ACTION       = 30

class FencerID(Enum):
    LEFT = 0
    RIGHT = 1

class Action:
    def __init__(self, label: ActionID, start_time: float, end_time: float, success: bool):
        self.action = label
        self.start = start_time
        self.end = end_time
        self.success = success

class Fencer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.actions = []

    def update_action(self, new_action: Action):
        self.actions.append(new_action)