Mathematical Modeling of Operational Amplifier
====================================================================
In the repository `Projects/Operation amplifier/` there are files `main.py` and` class_OpAm.py` and a `Initial_File.txt`. The .py files are executed in Python3.8.5.
***
The file `class_OpAm.py` is a class file. Input parameters are `init_x` - initial time value, `fin_x` - final time value , `step_x` - the step of using OA, and `const` - the value to which the conversion takes place. The file stores mathematical operations like: 
`add` - addition (... + с), 
`sbt` - subtraction ( ... - с), 
`mlt`- multiplication (... * с), 
`div` - division (... / с), 
`pwr` - power ( ... ** с), 
`rev_pwr` - reverse power (с ** ...), 
`log`- logarithm (log(..., c), 
`rev_log`- reverse logarithm(log(c, ...)).
***
The `main.py` file performs several tasks. 
1. Generating a signal by calling one of the functions 
2. Graphic output of data that is written to .txt files.
