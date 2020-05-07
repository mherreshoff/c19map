# simple_templates is a really simple templating system that lets you
# embed for loops and if statements into an f"..." style string.
import re
import sys

command_pat = re.compile('\s*\%\s*(.*)')
def group_by_countrol_structure(lines):
    result = []
    stack = [result]
    for line_number, line in enumerate(lines, start=1):
        m = command_pat.match(line)
        if not m:
            stack[-1].append((line_number, 'output', line))
        else:
            command = m.group(1)
            if command == 'end':
                stack.pop()
                if not stack:
                    raise Exception(f"Parse Error at line {i}: too many %end's.")
            else:
                inner = []
                stack[-1].append((line_number, m.group(1), inner))
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
        for line_number, command, arg in g:
            if command == 'output':
                try:
                    expansion = run("f"+repr(arg))   # Punts to f"...{...}..." expansion.
                except Exception as e:
                    print(f"Expansion error on line {line_number}: {e}")
                    sys.exit(1)
                result.append(expansion)
            else:
                for_m = for_loop_pat.match(command)
                if_m = if_statment_pat.match(command)
                if for_m:
                    variable = for_m.group(1)
                    iterator = for_m.group(2)
                    try: values = run(iterator)
                    except Exception as e:
                        print(f"Line {line_number}: couldn't evaluate iterator \"{iterator}\": {e}")
                        sys.exit(1)
                    try: values = list(values)
                    except Exception as e:
                        print(f"Line {line_number}: couldn't listify iterator \"{iterator}\": {e}")
                        sys.exit(1)
                    if ',' in variable:
                        variables = [v.strip() for v in variable.split(',')]
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
                    try: conditional_val = run(conditional)
                    except Exception as e:
                        print(f"Line {line_number}: couldn't evaluate conditional \"{conditional}\": {e}")
                        sys.exit(1)
                    if conditional_val: expand_grouped(arg)
                else:
                    raise Exception("Unrecognized control structure: " + command)
    expand_grouped(group_by_countrol_structure(lines))
    return "\n".join(result) + "\n"
