# tsp
Deep RL + Pointer Nets for the Travelling Salesman Problem

## Installation
_This has been tested on Python 3.7, but will likely work for Python>=3.6_

1. Clone this repository

2. Install tsp requirements

    ```
    pip install -r requirements.txt
    ```

3. Install tsp (probably want to do so in editable mode)

    ```
    pip install -e .
    ```


## Solver Installations [Optional]

4. Install non-learned solver submodules

    ```
    git submodule update --init --recursive
    ```
    
    Then repeat steps 1-3 for each submodule in solvers/, also following specific instructions below...

- pyconcorde 
    - Ensure there's a C compiler (e.g. gcc) and make on your base environment PATH for building concorde:

        ```
        sudo apt update
        sudo apt install gcc make
        ```

    - Overwrite setup.py with our custom version to fix an install bug (urllib, used to download QSOPT, doesn't support HTTP 308 redirect; so we swap urllib with requests)

        ```
        cd solvers
        cp custom_pyconcorde_setup.py pyconcorde/setup.py
        ```
    
    - Install pyconcorde in editable mode ("-e" with pip), otherwise the wrapped concorde binaries may not correctly install (even if pyconcorde does)

        ```
        cd pyconcorde
        pip install -r requirements.txt
        pip install -e .
        ```

    - IFF the pyconcorde install fails due to "ModuleNotFoundError: No module named 'requests'" then run setup.py directly

        ```
        python setup.py install
        ```



## Unit Testing
To run unit tests:
```
python test/run.py
```

Note pytest and other testing module scripts will not work because Logger must first be initialized.

## TODOs
Core
- Distributed GPU training
- Non-iterative decoder-only policy

Auxiliary
- Pretty stdout printing
- Naive external solver (e.g. nearest insertion)
- Plotting optimality gaps, running averages, and horizontal bounds with plot.py (Visdom)
- Undiscounted GAE for A2C

Bugs
- IO bug when using unittest with bridge.c_stderr_redirector (need PyConcorde to reproduce)
- Bugs with parallel ConcordeSolver use (redirectors appear to not be thread-safe)
    - Add parallel support in launch/gen_supervised_dataset.py after fixing this
- "inf" points in line plot result in graph not being plotted by Visdom (can replace with "nan" which displays as hole)

