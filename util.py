
import time

class TimeEstimator:
    def __init__(self, total_repeat, name = ""):
        self.time_analyzed = None
        self.time_count = 0
        self.total_repeat = total_repeat
        self.name = name

    def tick(self):
        if not self.time_analyzed:
            self.time_count += 1
            if self.time_count == 3:
                self.time_begin = time.time()

            if self.time_count == 13:
                elapsed = time.time() - self.time_begin
                expected_sec = elapsed / 10 * self.total_repeat
                expected_min = int(expected_sec / 60)
                print("Expected time for {} : {} min".format(self.name, expected_min))
                self.time_analyzed = True
