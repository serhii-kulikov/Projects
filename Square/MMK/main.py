import random as rand
import PyGnuplot as gp
import Class_MMKS

#Func_Init
Func = Class_MMKS.Func ()
xf = []
yf = []
Func.FtoL (xf, yf)
xd = []
yd = []
Func.DtoL (xd, yd)

#Func_Eq
a = min (xf)                                                    
b = max (xf)                                                    
H = max (yf)                                                    
S = (b - a) * H
trials = 500000
counts = Func.Counter (xf, yf, xd, yd) 
S_s = counts / trials * S
print ("The square under the function is:", S_s)

#Plot
Plot = Class_MMKS.Plot ()
Plot.Rectangle (a, b, H)
Plot.Dots (trials, a, b, H)
gp.c ('plot "Rectangel.txt" w l lw 3 lt rgb "black", "Dots.txt" w p lw 0.05 lt rgb "blue", "Initial_File.txt" w l lw 5 lt rgb "red"')
