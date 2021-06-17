import class_Sort_Alg
import time
import numpy.random as rand
import PyGnuplot as gp

len_mas = int (input ('Enter len:'))
min_val = int (input ('Enter min_val:'))
max_val = int (input ('Enter max_val:'))

Algorytm = class_Sort_Alg.Sort_Alg() 

Fout = open ("Counting_sort_File.txt", "w")
for c in range (len_mas // 10, len_mas + 1, len_mas // 10): 
    start_time = time.time ()
    Algorytm.Counting_sort (min_val, max_val, c)
    Fout.write (f"{c} {time.time () - start_time }\r\n")
Fout.close ()   
'''
Fout = open ("Insertion_sort_File.txt", "w")
for c in range (len_mas // 10, len_mas + 1, len_mas // 10):
    start_time = time.time ()
    Algorytm.Insertion_sort (min_val, max_val, c)
    Fout.write (f"{c} {time.time () - start_time }\r\n")
Fout.close ()  

Fout = open ("Heap_sort_File.txt", "w")
for c in range (len_mas // 10, len_mas + 1, len_mas // 10): 
    start_time = time.time ()
    Algorytm.Heap_sort (min_val, max_val, c)
    Fout.write (f"{c} {time.time () - start_time }\r\n")
Fout.close ()  

Fout = open ("Quick_sort_File.txt", "w")
for c in range (len_mas // 10, len_mas + 1, len_mas // 10): 
    start_time = time.time ()
    Algorytm.Quick_sort (min_val, max_val, c)
    Fout.write (f"{c} {time.time () - start_time }\r\n")
Fout.close ()

Fout = open ("Bubble_Sort_File.txt", "w")
for c in range (len_mas // 10, len_mas + 1, len_mas // 10): 
    start_time = time.time ()
    Algorytm.Bubble_Sort (min_val, max_val, c)
    Fout.write (f"{c} {time.time () - start_time }\r\n")
Fout.close ()
'''
gp.c ('plot "Counting_sort_File.txt" w l lw 1 lt rgb "red", "Insertion_sort_File.txt" w l lw 1 lt rgb "yellow", "Heap_sort_File.txt" w l lw 1 lt rgb "green","Quick_sort_File.txt" w l lw 1 lt rgb "blue", "Bubble_Sort_File.txt" w l lw 1 lt rgb "purple"')

#gp.c ('plot "Counting_sort_File.txt" w l lw 1 lt rgb "red"')
