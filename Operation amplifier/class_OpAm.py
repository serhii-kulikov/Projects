import math

y = 0

class OpAm:
    def add (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt", "a")
        x = init_x
        while x <= fin_x:
            if x == init_x:
                y += 0
            else:
                y += const
            Fout.write (f"{x} {y}\r\n")
            x += step_x
        Fout.close ()
        
    def sbt (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            if x == init_x:
                y -= 0
            else:
                y -= const
            Fout.write (f"{x} {y}\r\n")
            x += step_x
        Fout.close ()
        
    def mlt (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            Fout.write (f"{x} {y}\r\n")
            if y == 0:
            	y += 1
            else:
            	y *= const
            x += step_x
        Fout.close ()
        y /= const
        
    def drv (self, init_x, fin_x, step_x, const):    
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            if x == init_x:
            	y = y             
            else:
                y /= const
            Fout.write (f"{x} {y}\r\n")
            x += step_x
        Fout.close ()

    def pwr (self, init_x, fin_x, step_x, const):    
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            Fout.write (f"{x} {y}\r\n")  
            if y == 0:
               y += 1.1
            else:
            	y = y ** const
            x += step_x
        Fout.close ()
        y **= (1/const)

    def rev_pwr (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            Fout.write (f"{x} {y}\r\n")
            if y == 0:
            	y += 1
            else:				
            	y = const ** y
            x += step_x
        Fout.close ()
        y  = math.log(y,const)
        
    def log (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            Fout.write (f"{x} {y}\r\n")
            if y == 0:
            	y += 1.1
            else:		
            	y = math.log (y, const)
            x += step_x
        Fout.close ()
        y = const ** y
        
    def rev_log (self, init_x, fin_x, step_x, const):
        global y
        Fout = open ("Initial_File.txt","a")
        x = init_x
        while x <= fin_x:
            Fout.write (f"{x} {y}\r\n")
            if y == 0:
            	y += 1.1
            else:
            	y = math.log (const, y)
            x += step_x
        Fout.close ()
        y = 10 ** (math.log(const, 10)/y)
