#!/usr/bin/env python
import re

command_pat = re.compile('\s*\%\s*(.*)\:?')
def group_template(lines):
    result = []
    stack = [result]
    for i, line in enumerate(lines):

        m = command_pat.match(line)
        if not m:
            stack[-1].append(('output', line))
        else:
            command = m.group(1)
            if command == 'end':
                stack.pop()
                if not stack:
                    raise Exception(f"Parse Error at line {i}: too many %end's.")
            else:
                inner = []
                stack[-1].append((m.group(1), inner))
                stack.append(inner)
    return result

for_loop_pat = re.compile('for\s+(.*)\s+in\s+(.*)')
if_statment_pat = re.compile('if\s+(.*)')

def expand(code, global_env, local_env):
    result = []
    lines = code.split("\n")
    def assign(var, val): local_env[var] = val
    def run(s): return eval(s, global_env, local_env)
    def expand_grouped(g):
        for command, arg in g:
            if command == 'output':
                expansion = run("f"+repr(arg))
                    # Hijack the f-string mechanism to expand {...}s.
                result.append(expansion)
            else:
                for_m = for_loop_pat.match(command)
                if_m = if_statment_pat.match(command)
                if for_m:
                    variable = for_m.group(1)
                    iterator = for_m.group(2)
                    values = run(iterator)
                    if ',' in variable:
                        variables = variable.split(',')
                        for val in values:
                            if len(val) != len(variables):
                                raise Exception("Destructuring bind failed.")
                            for var,v in zip(variables, val): assign(var, v)
                            expand_grouped(arg)
                    else:
                        for val in values:
                            assign(variable, val)
                            expand_grouped(arg)
                elif if_m:
                    conditional = if_m.group(1)
                    if run(conditional): expand_grouped(arg)
                else:
                    raise Exception("Unrecognized control structure: " + command)
    expand_grouped(group_template(lines))
    return "\n".join(result) + "\n"
