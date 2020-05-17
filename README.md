# c19map.org scripts

This repository contains scripts to help run the [c19map.org](https://c19map.org/) project.

+ import.py - Pulls data from John's Hopkins University and from our spreadsheet with intervention and population data and outputs some files.
+ predict.py - Runs our model on that data and outputs some prediction CSV files, and graphs.

## Setup

In order to run predict.py we first need to compile the integrator for our differential equation, using the following
commands:

```
./compile_integrators.py
python3 setup.py build_ext --inplace
```

The first command generates the `integrators.pyx` file which can integrate the differential equation for our model (and can
also optionally calculate a sensitivity matrix for use with differential methods.)  The second command compiles
`integrators.pyx` into a fast library we can import.
