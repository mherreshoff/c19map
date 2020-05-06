#!/usr/bin/env python

def group_template(lines):
    result = []
    stack = [result]
    for i, line in enumerate(lines):
        if line == '%end':
            stack.pop()
            if len(stack) == 0:
                raise ValueError, "Parse Error: too many %end's."
        elif line[0] == '%':
            inner = []
            stack[-1].append((line, inner))
            stack.append(inner)
        else:
            stack[-1].append(('output', line))
    return result []

for_loop = re.compile('\%for (.*) in (.*)')
if_statment = re.compile('\%if (.*)')

def expand_line(line, global_env, local_env):

def expand(code, global_env, local_env):
    result = []
    lines = code.split("\n")
    grouped_lines = group_template(lines)
    def handle(g):
        for command, arg in g:
            if command == 'output':
                expansion = eval("f"+repr(line), global_env, local_env)
                    # Hijack the f-string mechanism to expand {...}s.
                result.append(expand_line(line, global_env, local_env))
            elif command[0] == '%':
                for_m = for_loop.match(command)
                if_m = for_loop.match(command)
                if for_m:
                    variable = for_m.group(1)
                    iterator = for_m.group(2)
                    values = eval(iterator, global_env, local_env):
                    if ',' in variable:
                        variables = ','.split(variable)
                        for val in values:
                            for var,v in zip(variables, val):
                                local_env[var]=v
                            handle(arg)
                    else:
                        for val in values:
                            local_env[variable]=val
                            handle(arg)
                elif if_m:
                    conditional = if_m.group(1)
                    if eval(conditional, global_env, local_env):
                        handle(arg)
                else:
                    print("Unrecognized control structure: " + command)
    return "\n".join(result) + "\n"

