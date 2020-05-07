# c19map.org scripts

This repository contains scripts to help run the [c19map.org](https://c19map.org/) project.

+ import.py - Pulls data from John's Hopkins University and from our Intervention spreadsheet and outputs some files.
+ compartmental\_model.py - Runs our model on that data and outputs some prediction CSV files.

## Setup

In order to run compartmental\_model.py we need to compile the integrator for our differential equation, using the following
commands:

```
./generate_integrator.py
python3 setup.py build_ext --inplace
```

The first command generates the `.pyx` file which integrates the differential equation for our model (and also calculates a
sensitivity matrix.)  The second command compiles the `.pyx` file into a fast library we can import.
