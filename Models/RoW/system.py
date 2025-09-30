from enum import Enum

from fencer import Fencer
from action import Action, ActionID

# Used to delineate between action catagories based on the action ID
OFFENSIVE_ACTION_ID_MIN = 10
DEFENSIVE_ACTION_ID_MIN = 20
OTHER_ACTION_ID_MIN     = 30
# Padding in milliseconds to approximate real-world timing, as fencing is not a frame perfect sport
SIMUL_PADDING           = 40
ATTACK_END_PADDING      = 20

class RefCall(Enum):
    # Attacker
    ATTACK_HIT_COUNTER_ATTACK       = 10
    ATTACK_HIT_PARRY_NO_RIPOSTE     = 11
    ATTACK_NO_COUNTER_RIPOSTE       = 12
    ATTACK_NO_REMISE                = 13
    BEAT_ATTACK                     = 14
    BEAT_ATTACK_COUNTER_ATTACK      = 15
    BEAT_ATTACK_POINT_IN_LINE       = 16
    # Defender
    ATTACK_HIT_PARRY_RIPOSTE        = 20
    ATTACK_HIT_POINT_IN_LINE        = 21
    ATTACK_NO_COUNTER_ATTACK        = 22
    ATTACK_NO_COUNTER_ATTACK_REMISE = 23
    ATTACK_NO_PARRY_RIPOSTE         = 24
    ATTACK_NO_PARRY_RIPOSTE_REMISE  = 25
    # Other
    SINGLE_HIT                      = 30    # Early return case, skips RoW logic
    SIMULTANEOUS                    = 31    # No point, very rarely called at high levels

class System:
    def __init__(self):
        self.fencers: dict[str, Fencer] = {
            "LEFT": Fencer(),
            "RIGHT": Fencer()
        }

        self.right_of_way = None

    def reset(self):
        for fencer in self.fencers.values():
            fencer.reset()

        self.right_of_way = None

    def recieve_action(self, fencer: str, action: Action):
        self.fencers[fencer].update_action(action)

    # TODO: Track / update right-of-way
    # TODO: Implement FIE priority rules for each ref call (decision tree?)