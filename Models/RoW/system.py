from fencer import *

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

        self.attacker: FencerID = None
        self.defender: FencerID = None

    def reset(self):
        for fencer in self.fencers.values():
            fencer.reset()

        self.attacker = None
        self.defender = None

    def recieve_action(self, fencer: FencerID, action: Action):
        self.fencers[fencer].update_action(action)
        self.update_right_of_way(fencer, action)

    def update_right_of_way(self, fencer: FencerID = None, action: Action = None):
        # Initial actions
        if self.attacker is None:
            left_action = self.fencers[FencerID.LEFT].actions[0]
            right_action = self.fencers[FencerID.RIGHT].actions[0]
            # Simultaneous starts default to left, get resolved in get_point() based on SIMUL_PADDING
            self.attacker = FencerID.LEFT if left_action.start < right_action.start else FencerID.RIGHT
            self.defender = (self.attacker - 1)**2 # Funky math to get the other fencer
        # Exchanges - RoW switches on defender-favored blade contact or attack-no from the attacker
        else:
            _attacker = self.fencers[self.attacker]
            if fencer == self.attacker:
                # Attacker preperation -> attack/parry, parry -> riposte, beat -> riposte are valid transitions, do not change RoW
                if (
                    (_attacker.actions[-2].label is ActionID.PREPERATION and (
                        action.label in {ActionID.ATTACK_SIMPLE, ActionID.ATTACK_COMPOUND, ActionID.BEAT, ActionID.PARRY}
                    ))
                    or (_attacker.actions[-2].label is ActionID.PARRY and action.label is ActionID.RIPPOSTE)
                    or (_attacker.actions[-2].label is ActionID.BEAT and action.label is ActionID.RIPPOSTE)
                ): return
                # Blade contact favoring attacker, do not change RoW
                elif action.blade_contact == self.attacker: return
                # Otherwise, attacker either got parried or missed the attack, change RoW
                else: self.attacker, self.defender = self.defender, self.attacker
            # Defender can only actively gain RoW by favorable blade contact (attack-no depends on attacker)
            elif action.blade_contact != self.attacker: 
                self.attacker, self.defender = self.defender, self.attacker
    
    def get_point(self, attacker_action: Action, defender_action: Action):
        # Single hit - return immidiately
        if attacker_action.hit and not defender_action.hit: return self.attacker
        if not attacker_action.hit and defender_action.hit: return self.defender

        # blade_contact should be the same for both actions, favor defender just in case
        blade_contact = defender_action.blade_contact

        def attacker_bladework(blade_contact): return self.attacker if blade_contact is self.attacker else self.defender
        def defender_bladework(blade_contact): return self.defender if blade_contact is self.defender else self.attacker

        # Double hit - compare every attacker action to every defender action
        # Most of these should never happen but must be included just in case
        match attacker_action.label:
            case ActionID.PREPERATION:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.defender
                    case ActionID.ATTACK_COMPOUND:  point = self.defender
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.defender
                    case ActionID.POINT_IN_LINE:    point = self.defender

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.ATTACK_SIMPLE:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker    # TODO: Timing check for simuls
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.ATTACK_COMPOUND:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker   # TODO: Timing check for preperation
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker   # TODO: Timing check for simuls
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender   # Uncertain

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.REMISE:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.RIPPOSTE:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.attacker
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.attacker

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.STOP_CUT:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker    # TODO: Timing check for simuls
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.POINT_IN_LINE:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.attacker
                    case ActionID.ATTACK_SIMPLE:    point = self.attacker
                    case ActionID.ATTACK_COMPOUND:  point = self.attacker   
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender    # Uncertain

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.PARRY:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = attacker_bladework(blade_contact)
                    case ActionID.ATTACK_SIMPLE:    point = attacker_bladework(blade_contact)
                    case ActionID.ATTACK_COMPOUND:  point = attacker_bladework(blade_contact)
                    case ActionID.REMISE:           point = attacker_bladework(blade_contact)
                    case ActionID.RIPPOSTE:         point = attacker_bladework(blade_contact)
                    case ActionID.STOP_CUT:         point = attacker_bladework(blade_contact)
                    case ActionID.POINT_IN_LINE:    point = attacker_bladework(blade_contact)

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.BEAT:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = attacker_bladework(blade_contact)
                    case ActionID.ATTACK_SIMPLE:    point = attacker_bladework(blade_contact)
                    case ActionID.ATTACK_COMPOUND:  point = attacker_bladework(blade_contact)
                    case ActionID.REMISE:           point = attacker_bladework(blade_contact)
                    case ActionID.RIPPOSTE:         point = attacker_bladework(blade_contact)
                    case ActionID.STOP_CUT:         point = attacker_bladework(blade_contact)
                    case ActionID.POINT_IN_LINE:    point = attacker_bladework(blade_contact)

                    case ActionID.PARRY:            point = defender_bladework(blade_contact)
                    case ActionID.BEAT:             point = defender_bladework(blade_contact)

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.NO_ACTION:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.defender
                    case ActionID.ATTACK_SIMPLE:    point = self.defender
                    case ActionID.ATTACK_COMPOUND:  point = self.defender
                    case ActionID.REMISE:           point = self.defender
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.defender
                    case ActionID.POINT_IN_LINE:    point = self.defender

                    case ActionID.PARRY:            point = self.defender
                    case ActionID.BEAT:             point = self.defender

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker
            case ActionID.DISTANCE_PULL:
                match defender_action.label:
                    case ActionID.PREPERATION:      point = self.defender
                    case ActionID.ATTACK_SIMPLE:    point = self.defender
                    case ActionID.ATTACK_COMPOUND:  point = self.defender
                    case ActionID.REMISE:           point = self.attacker
                    case ActionID.RIPPOSTE:         point = self.defender
                    case ActionID.STOP_CUT:         point = self.attacker
                    case ActionID.POINT_IN_LINE:    point = self.defender

                    case ActionID.PARRY:            point = self.defender
                    case ActionID.BEAT:             point = self.defender

                    case ActionID.NO_ACTION:        point = self.attacker
                    case ActionID.DISTANCE_PULL:    point = self.attacker

        return point

    def make_call(self):
        if self.attacker is None:
            self.update_right_of_way()
            return self.make_call()

        attacker_action = self.fencers[self.attacker].actions[-1]
        defender_action = self.fencers[self.defender].actions[-1]

        point = self.get_point(attacker_action, defender_action)
        return {
            'point': point,
            'left_action': self.fencers[FencerID.LEFT].actions[-1],
            'right_action': self.fencers[FencerID.RIGHT].actions[-1],
            'priority': self.attacker,
            'blade_contact': defender_action.blade_contact
        }