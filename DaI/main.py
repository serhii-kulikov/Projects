import class_DaI
import PyGnuplot as gp

len_Fin = int (len (open ("Initial_File.txt", "r").readlines ())) 
Diffentiation = class_DaI.DaI ()

Fin = open ("Initial_File.txt", "r")
Fout = open ("Diff_File.txt", "w")

for i in range (1, len_Fin + 1):
    str = Fin.readline().split()
    x = float (str [0])
    y = float (str [1])
    diff_y = Diffentiation.Diff (x, y)
    Fout.write (f"{x} {diff_y}\r\n")
    
Fin.close ()
Fout.close ()

Integration = class_DaI.DaI ()

Fin = open ("Initial_File.txt", "r")
Fout = open ("Int_File.txt", "w")

for i in range (1, len_Fin):
    str = Fin.readline().split()
    x, y = float (str [0]), float (str [1])
    int_y = Integration.Int (x, y)
    Fout.write (f"{x} {int_y}\r\n")

Fin.close ()
Fout.close ()

gp.c ('plot "Initial_File.txt" w l lw 1 lt rgb "black", "Diff_File.txt" w l lw 1 lt rgb "red", "Int_File.txt" w l lw 1 lt rgb "blue" ')
