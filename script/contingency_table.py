class ContingencyTable:
    def __init__(self):
        self.counter=dict()
        for p in [False,True]:
            for o in [False,True]:
                self.counter[p,o]=0
    def add(self,p,o):
        self.counter[p,o]+=1
    def tss(self):
        tp, fp = self.counter[True,True], self.counter[True,False]
        fn, tn = self.counter[False,True], self.counter[False,False]
        return tp/max(1e-8,float(tp+fn))-fn/max(1e-8,float(fp+tn))
    def attenuate(self,factor):
        for p in [False,True]:
            for o in [False,True]:
                self.counter[p,o]*=(1-factor)
