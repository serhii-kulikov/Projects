class DaI:
    
    def __init__ (self):
        self.prevx = 0
        self.prevy = 0
        self.lastDeriv = 0
        self.lastInteg = 0
  
    def Diff (self, x, y):   	 
        self.dx = (x - self.prevx)
        self.dy = (y - self.prevy)
        if self.dx == 0:
            self.deriv = 0
        elif self.dy == 0:
            self.deriv = (self.prevy / self.dx)
        else: 
            self.deriv = (self.dy / self.dx)
        self.prevx = x                                        
        self.prevy = y
        self.lastDeriv = self.deriv
        return self.deriv
    
    def Int (self, x, y):
        self.integ = self.lastInteg + 0.5 * (self.prevy + y) * (x - self.prevx)
        self.prevx = x
        self.prevy = y
        self.lastInteg = self.integ
        return self.integ
