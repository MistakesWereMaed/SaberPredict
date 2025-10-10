from fencer import *

# Used to delineate between action categories based on the action ID
OFFENSIVE_ACTION_ID_MAX = 14
DEFENSIVE_ACTION_ID_MAX = 26
# Padding in milliseconds to approximate real-world timing, as fencing is not a frame perfect sport
# TODO: Test
SIMUL_PADDING           = 0
ATTACK_END_PADDING      = 0

class System:
    def __init__(self):
        self.fencers: dict[FencerID, Fencer] = {
            FencerID.LEFT: Fencer(),
            FencerID.RIGHT: Fencer()
        }

        self.right_of_way: FencerID = None
        self.attacker: FencerID = None
        self.defender: FencerID = None

    def reset(self):
        for fencer in self.fencers.values():
            fencer.reset()

        self.right_of_way = None
        self.attacker = None
        self.defender = None

    def recieve_action(self, fencer: FencerID, action: Action):
        self.fencers[fencer].update_action(action)

    # TODO: Update right-of-way
    def update_right_of_way(self):
        return

    def make_call(self):
        if self.right_of_way is not None:
            self.attacker = self.right_of_way
            self.defender = (self.right_of_way - 1)**2 # Funky math to get the other fencer
        else:
            self.update_right_of_way()
            return self.make_call()

        attacker_action = self.fencers[self.attacker].actions[-1]
        defender_action = self.fencers[self.defender].actions[-1]

        # Single Hit - Easy case
        if attacker_action.success and not defender_action.success:     point = self.attacker
        elif not attacker_action.success and defender_action.success:   point = self.defender
        else:
            # Double Hit - Compare every attacker action to every defender action
            match attacker_action.label:
                case ActionID.PREPERATION:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.defender
                        case ActionID.ATTACK_COMPOUND:  point = self.defender
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.ATTACK_SIMPLE:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker   # TODO: Timing check for simuls
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.ATTACK_COMPOUND:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker   # TODO: Timing check for preperation
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker   # TODO: Timing check for simuls
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.ATTACK_BEAT:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker
                        case ActionID.ATTACK_BEAT:      point = self.defender    # Uncertain
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker 
                        case ActionID.RIPPOSTE:         point = self.defender    # Uncertain
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.REMISE:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.defender
                        case ActionID.ATTACK_COMPOUND:  point = self.defender
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.defender
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.PARRY:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.defender
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker
                        case ActionID.ATTACK_BEAT:      point = self.defender      # Uncertain
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.attacker
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.RIPPOSTE:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker
                        case ActionID.ATTACK_BEAT:      point = self.attacker    # Uncertain
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.attacker    # Uncertain
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.attacker
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.STOP_CUT:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker   # TODO: Timing check for simuls
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.DISTANCE_PULL:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.defender
                        case ActionID.ATTACK_SIMPLE:    point = self.defender
                        case ActionID.ATTACK_COMPOUND:  point = self.defender
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.POINT_IN_LINE:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.attacker
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain
                        case ActionID.NO_ACTION:        point = self.attacker
                case ActionID.NO_ACTION:
                    match defender_action.label:
                        case ActionID.PREPERATION:      point = self.attacker
                        case ActionID.ATTACK_SIMPLE:    point = self.attacker
                        case ActionID.ATTACK_COMPOUND:  point = self.attacker
                        case ActionID.ATTACK_BEAT:      point = self.defender
                        case ActionID.REMISE:           point = self.attacker
                        case ActionID.PARRY:            point = self.defender
                        case ActionID.RIPPOSTE:         point = self.defender
                        case ActionID.STOP_CUT:         point = self.attacker
                        case ActionID.DISTANCE_PULL:    point = self.attacker
                        case ActionID.POINT_IN_LINE:    point = self.defender
                        case ActionID.NO_ACTION:        point = self.attacker

        return {
            'point': point,
            'priority': self.right_of_way,
            'left_action': self.fencers[FencerID.LEFT].actions[-1],
            'right_action':self.fencers[FencerID.RIGHT].actions[-1]
        }