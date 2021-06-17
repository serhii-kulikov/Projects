import class_OpAm
import PyGnuplot as gp

condition = 0
class_OpAm.y = 0

Fout = open ("Initial_File.txt",'w')
Fout.close ()

while condition == 0:
    Op_Am = class_OpAm.OpAm()
    operation = input ('Enter the arythmetic operation symbol:')
    start = float (input ('Enter the initial x:'))
    finish = float (input ('Enter the final x:'))
    step = float (input ('Enter the step of x:'))
    const = float (input ('Enter the constant of equation:'))
    if operation == '+':
        Op_Am.add(start,finish,step, const)
    elif operation == '-':
        Op_Am.sbt(start,finish,step, const)
    elif operation == '*':
        Op_Am.mlt(start,finish,step, const)
    elif operation == '/':
        Op_Am.drv(start, finish, step, const)
    elif operation == 'y**c':
        Op_Am.pwr(start,finish,step, const)
    elif operation == 'c**y':
        Op_Am.rev_pwr (start, finish, step, const)
    elif operation == 'log_y':
        Op_Am.log(start,finish,step, const)
    elif operation == 'log_c':
        Op_Am.rev_log(start,finish,step, const)
    else:
        print ('Please, enter the corect operation symbol!')
        continue
    while True:
        proposition = input ('Anything else?')
        if proposition == 'Yes':
            break
        elif proposition == 'No':
            gp.c ('plot "Initial_File.txt" w l lw 1 lt rgb "blue"')
            condition = 1
            break
        else:
            print ('Please, enter "Yes" or "No"')
            continue
