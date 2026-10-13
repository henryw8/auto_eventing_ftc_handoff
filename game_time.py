from datetime import timedelta


def _display_football_time(half, minute=0, sec=0, time_from_zero=True, time_ms=None):
    """
    Create a string to show time

    :params half: Half
    :params minute: Minute
    :params sec: Second
    :params time_from_zero: Time starts from 0 at the beginning of each half.
    :params time_ms: Time in milliseconds to convert from
    :return: String with the GameTime shown
    """

    if time_ms is not None:
        minute = int(time_ms // (60 * 1000))
        sec = time_ms % (60 * 1000) / 1000

    sec = round(sec,2)
    # Add 0 before minute, second if < 10
    if sec < 10:
        sec = '0' + str(sec)
    else:
        sec = str(sec)
        
    if len(sec) == 4:
        sec = sec +'0'

    if time_from_zero:
        if minute < 10:
            minute = '0' + str(minute)
        return f'Half {half}, {minute}:{sec}'
    else:
        if half == 1:
            if minute < 10:
                minute = '0' + str(minute)
        elif half == 2:
            minute = 45 + minute
        elif half == 3:
            minute = 90 + minute
        elif half == 4:
            minute = 105 + minute

        return f'GameTime {minute}:{sec}'



class GameTime:

    def __init__(self, half, minute, second):
        """
        Create a new GameTime instance.

        Half is the integer of the half, starting at 1 and counting up.
        Minute (integer) is the minute of the half, starting at 0 and counting up.
        Second (float) is the second of the minute, starting at 0 and counting up.
            Valid range [0, 60).
        """
        self._td = timedelta(days=int(half), minutes=int(minute), seconds=float(second))

    def get_half(self):
        """
        Get the half of the time. Returns an integer that is at least 1.
        """
        return self._td.days

    def get_minute(self):
        """
        Get the minute of the time. Returns an integer that is at least 0.
        """
        return int((self._td.total_seconds() - self.get_half() * 24 * 60 * 60) / 60)

    def get_second(self):
        """
        Get the second of the time. Returns a float.
        """
        return round(self._td.total_seconds() % 60, 4)

    @classmethod
    def from_gametime_string(cls, string):
        """
        Convert time in event time format to standard game time format.
        """
        # Convert milliseconds to minute and second

        half = int(string.split(",")[0].split(" ")[1])
        minute = int(string.split(",")[1].split(":")[0])
        second = float(string.split(",")[1].split(":")[1])
        return cls(half, minute, second)

    @classmethod
    def from_event_time(cls, half, milliseconds):
        """
        Convert time in event time format to standard game time format.
        """
        # Convert milliseconds to minute and second
        minute = milliseconds // (60 * 1000)
        second = milliseconds % (60 * 1000) / 1000
        return cls(half, minute, second)

    @classmethod
    def from_tracking_time(cls, half, minute, second):
        """
        Convert time in tracking time format to standard game time format.
        """
        return cls(half, minute, second)

    def to_event_time(self):
        """
        Returns the stored time in event time format: (half, milliseconds), of type (str, int)
        """
        return (str(self.get_half()),
                int(self.get_minute() * 60000 + self.get_second() * 1000))

    def to_tracking_time(self):
        """
        Returns the stored time in the tracking time format: (half, minute, seconds), of type (int, int, float)
        """
        return (int(self.get_half()),
                int(self.get_minute()),
                float(self.get_second()))

    def add_delta_seconds(self, delta_seconds):
        """
        Returns a new GameTime instance that is the curent instance + the
        given delta in seconds.
        """
        # Get the current seconds into the half
        half = self.get_half()
        minute = self.get_minute()
        second = self.get_second()
        total_seconds = minute * 60.0 + second

        # Compute delta
        new_seconds = total_seconds + delta_seconds

        if new_seconds < 0:
            # Just go to start of the current half
            return GameTime(half, 0, 0)

        # Else, figure out new minute and seconds
        new_minutes = new_seconds // 60
        new_seconds = new_seconds % 60
        return GameTime(half, new_minutes, new_seconds)

    def __str__(self):
        return _display_football_time(self.get_half(), self.get_minute(), self.get_second())
        
    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.get_half() + hash(self._td)
        
    def __eq__(self, other):
        return all((self.get_half() == other.get_half(),
                    self.get_minute() == other.get_minute(),
                    self.get_second() == other.get_second()))
                    
    def __gt__(self, other):
        sh, oh = self.get_half(), other.get_half()
        sm, om = self.get_minute(), other.get_minute()
        ss, os = self.get_second(), other.get_second()
        return any(((sh > oh),
                    (sh == oh and sm > om),
                    (sh == oh and sm == om and ss > os)))

    def __ge__(self, other):
        sh, oh = self.get_half(), other.get_half()
        sm, om = self.get_minute(), other.get_minute()
        ss, os = self.get_second(), other.get_second()
        return any(((sh > oh),
                    (sh == oh and sm >= om),
                    (sh == oh and sm == om and ss >= os)))    

    def __le__(self, other):
        sh, oh = self.get_half(), other.get_half()
        sm, om = self.get_minute(), other.get_minute()
        ss, os = self.get_second(), other.get_second()
        return any(((sh < oh),
                    (sh == oh and sm <= om),
                    (sh == oh and sm == om and ss <= os)))    

    def __lt__(self, other):
        sh, oh = self.get_half(), other.get_half()
        sm, om = self.get_minute(), other.get_minute()
        ss, os = self.get_second(), other.get_second()
        return any(((sh < oh),
                    (sh == oh and sm < om),
                    (sh == oh and sm == om and ss < os)))
                    
                    
               

