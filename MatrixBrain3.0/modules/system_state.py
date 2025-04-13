import random

class SystemState:
    def __init__(self):
        self.memory = {'short_term': [], 'long_term': []}
        self.emotion = 'neutral'

    def update_emotion(self, energy):
        if energy > 0.8:
            self.emotion = 'happy'
        elif energy < 0.3:
            self.emotion = 'sad'
        else:
            self.emotion = 'neutral'

    def save_memory(self, matrix_snapshot):
        if len(self.memory['short_term']) > 20:
            self.memory['short_term'].pop(0)
        self.memory['short_term'].append(matrix_snapshot.clone())

        if random.random() < 0.1 and len(self.memory['long_term']) < 100:
            self.memory['long_term'].append(matrix_snapshot.clone())