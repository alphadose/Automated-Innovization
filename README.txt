How to use:-

1. Install all dependencies using "pip install -r requirements.txt"

2. Scripts to generate results for TRUSS, WELD, CLUTCH, SPRING and GEAR design problems are present in the "examples" folder

3. For example to get the results for TRUSS problem do "python examples/truss.py"

4. The source code for our entire framework is present in the file "model.py"


Design problem to script mapping:-

TRUSS -> examples/truss.py
WELD -> examples/beam.py
CLUTCH -> examples/clutch.py
SPRING -> examples/spring.py
GEAR with smaller dataset (no duplicates) -> examples/gear_small.py
CLUTCH with smaller dataset (no duplicates) -> examples/clutch_small.py
SPRING with smaller dataset (no duplicates) -> examples/spring_small.py
