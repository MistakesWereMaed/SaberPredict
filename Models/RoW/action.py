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
    NO_ACTION       = 30    # Used to mark the end of an action

class Action:
    def __init__(self, label: ActionID, start: float, end: float, success: bool):
        self.action = label
        self.start = start
        self.end = end
        self.success = success