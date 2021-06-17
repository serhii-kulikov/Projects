Fout = open ("Initial_File.txt","w")

for x in range (1, 11):
    y = x ** 0.5 
    Fout.write (f"{x} {y}\r\n")

Fout.close ()

