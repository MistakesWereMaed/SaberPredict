from action import Action

class Fencer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_action: Action = None
        self.current_action: Action = None

    def update_action(self, new_action: Action):
        self.previous_action = self.current_action
        self.current_action = new_action