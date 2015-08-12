class PopulationTable:
    def __init__(self,coverage=range(30,61),scale=10):
        self.coverage = coverage
        self.scale = scale
        self.above=dict()
        self.below=dict()
        for i in coverage:
            self.above[i]=1
            self.below[i]=1
    def add_event(self,x0):
        x = x0 * self.scale
        for i in self.coverage:
            if x >= i:
                self.above[i] += 1
            else:
                self.below[i] += 1
    def count_event_pair(self,x0):
        lo = self.coverage[0]
        hi = self.coverage[-1]
        i = max(lo,min(hi,int(round(x0 * self.scale))))
        return (self.below[i], self.above[i])

    # multiply gradient by this factor
    # so that events with biased population will be learned
    # more slowly.
    def population_ratio(self,x0):
        b,a = self.count_event_pair(x0)
        m = float(max(a,b))
        return (b/m, a/m)


if __name__ == '__main__' :
    p = PopulationTable()
    p.add_event(3.5)
    p.add_event(3.6)
    p.add_event(4.5)
    assert (p.count_event_pair(4.2)==(3,2))
    assert (p.count_event_pair(4.51)==(3,2))
    assert (p.count_event_pair(4.55)==(4,1))
    assert (p.count_event_pair(9999)==(4,1))
    assert (p.count_event_pair(-9999)==(1,4))
