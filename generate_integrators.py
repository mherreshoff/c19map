#!/usr/bin/env python3
from compiler.ode_cython import OdeCython

ode_cython = OdeCython();

ode_cython.declare_ode(
        name="compound",
        variables=["x"],
        parameters=["g"],
        derivatives={"x": "g*x"})

ode_cython.declare_ode(
        name="pendulum",
        variables=["theta", "omega"],
        parameters = ["b", "c"],
        derivatives={
            "theta": "omega",
            "omega": "-b*omega - c*sin(theta)"})

ode_cython.declare_ode(
        name="sir",
        variables=["S", "I", "R"],
        parameters=["gamma"],
        influences=["beta"],
        derivatives={
            "S": "-(S*I/(S+I+R))*beta",
            "I": "(S*I/(S+I+R))*beta - gamma * I",
            "R": "gamma * I"})

ode_cython.declare_ode(
        name="seir",
        variables=["S", "E", "I", "R"],
        parameters=[
            "exposed_leave_rate",
            "infectious_leave_rate"],
        influences=["beta"],
        derivatives={
            "S": "-I*beta*S/(S+E+I+H+R)",
            "E": "I*beta*S/(S+E+I+H+R) - E*exposed_leave_rate",
            "I": "E*exposed_leave_rate - I*infectious_leave_rate",
            "R": "I*infectious_leave_rate"})

ode_cython.declare_ode(
        name="augmented_seir",
        variables=["S", "E", "I", "H", "D", "R"],
        parameters=[
            "exposed_leave_rate",
            "infectious_leave_rate",
            "hospital_leave_rate",
            "hospital_p",
            "death_p"],
        influences=["beta"],
        derivatives={
            "S": "-I*beta*S/(S+E+I+H+R)",
            "E": "I*beta*S/(S+E+I+H+R) - E*exposed_leave_rate",
            "I": "E*exposed_leave_rate - I*infectious_leave_rate",
            "H": "I*infectious_leave_rate*hospital_p - H*hospital_leave_rate",
            "D": "H*hospital_leave_rate*death_p",
            "R": "I*infectious_leave_rate*(1-hospital_p) + H*hospital_leave_rate*(1-death_p)"
            })

ode_cython.save('integrators.pyx')
