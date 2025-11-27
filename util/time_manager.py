import time

class TimeManager():
    def __init__(self):
        self.total_budget = 24.0
        self.used_time = 0.0
        self._move_start = None
        self._deadline = None

    def remaining(self):
        return max(0.0, self.total_budget - self.used_time)
    
    def start_move(self):
        self._move_start = time.perf_counter()
    
    def end_move(self):
        dt = time.perf_counter() - self._move_start
        self.used_time += dt
        return dt
    
    def set_deadline(self, budget):
        self._deadline = self._move_start + budget

    def check_time(self):
        if time.perf_counter() >= self._deadline:
            raise TimeoutError()
    def get_move_start_time(self):
        return self._move_start
    def reset(self):
        self.total_budget = 24.0
        self.used_time = 0.0
