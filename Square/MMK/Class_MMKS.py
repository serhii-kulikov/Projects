class Plot:

    def Rectangle (self, a, b, H):
        Frect = open ("Rectangel.txt", "w")
        for i in range (1, 6):
            if i == 1 or i == 2 or i == 5:
                xr = b 
                yr = (i % 2) * H
            else:
                xr = a
                yr = ((i +1) % 2) * H
            Frect.write (f"{xr} {yr}\r\n")
        Frect.close

    def Dots (self, trials, a, b, H):
        import random as rand
        Fdots = open ("Dots.txt", "w")
        for i in range (trials):
            xd = rand.uniform (a, b)
            yd = rand.uniform (0, H)
            Fdots.write (f"{xd} {yd}\r\n")
        Fdots.close
        
class Func:

    def FtoL (self, x, y): 
        with open ("Initial_File.txt",'r') as f:
            data = f.readlines ()
            lines = [d.rstrip ('\n') for d in data]
        for line in lines:
            x.append (float (line.split () [0]))
            y.append (float (line.split () [1]))
        f.close
    
    def DtoL (self, xd, yd):
        with open ("Dots.txt",'r') as f:
            data = f.readlines ()
            lines = [d.rstrip ('\n') for d in data]
        for line in lines:
            xd.append (float (line.split () [0]))
            yd.append (float (line.split () [1]))
        f.close 
        
    def Counter (self, xf, yf, xd, yd):
        counts = 0
        for i in range (len (xf)):
            for j in range (len (xd)):
                if xf [i] <= xd [j] <= xf [i + 1] and yd [j] <= yf [i]:
                    counts += 1
        return counts
