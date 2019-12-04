

class LookaheadIter(object):
    # http://stackoverflow.com/a/1517965/166700
    def __init__(self, sequence):
        self.iter = iter(sequence)
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop(0)
        else:
            return next(self.iter)

    def lookahead(self, n=1):
        """Return an item n entries ahead in the iteration."""
        if n <= 0:
            raise ValueError("n=%r" % n)
        while n > len(self.buffer):
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                return None
        return self.buffer[n - 1]
