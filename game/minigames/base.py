import time


class BaseMiniGame:
    name        = 'Mini Game'
    instruction = 'Follow the instructions!'
    duration    = 15.0
    grab_phase  = False  # if True, timer starts only after first grab (call _begin_timer)

    def __init__(self):
        self._start_time = None
        self._complete   = False

    def start(self):
        # If grab_phase, hold off on starting the timer until _begin_timer() is called
        self._start_time = None if self.grab_phase else time.time()
        self._complete   = False

    def _begin_timer(self):
        """Start the countdown — called once the player grabs their first tool."""
        if self._start_time is None:
            self._start_time = time.time()

    def extend_start_time(self, seconds):
        """Shift the timer's start point forward so a pause doesn't eat into time_remaining."""
        if self._start_time is not None:
            self._start_time += seconds

    @property
    def timer_started(self):
        return self._start_time is not None

    @property
    def time_remaining(self):
        if self._start_time is None:
            return self.duration
        return max(0.0, self.duration - (time.time() - self._start_time))

    @property
    def time_ratio(self):
        return self.time_remaining / self.duration

    @property
    def succeeded(self):
        return False

    @property
    def progress_text(self):
        return ''

    def check_done(self):
        if self._complete:
            return True
        # Only time-out once the timer has actually started
        if self._start_time is not None and self.time_remaining <= 0:
            self._complete = True
            return True
        return False

    def update(self, hands):
        raise NotImplementedError

    def draw(self, frame, hands):
        raise NotImplementedError
