class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}

    def store(self, experience_dict):
        # Discard the oldest experience if the buffer is full
        if len(self.buffer['state']) >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        # Add new experience at the end
        for key, value in experience_dict.items():
            self.buffer[key].append(value)

    def size(self):
        return len(self.buffer['state'])