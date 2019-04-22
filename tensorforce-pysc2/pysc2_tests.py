from pysc2.agents import base_agent
from pysc2.lib import actions


FUNCTIONS = actions.FUNCTIONS

def getActionSpecFromFunctionID(id):
    print(FUNCTIONS[id].name)
    # print("ID:")
    # print(FUNCTIONS[id].id)
    for i in range(len(FUNCTIONS[id].args)):
        print('    arg ' + str(i) + ':')
        print(FUNCTIONS[id].args[i].id)
        print('argument name:')
        print(FUNCTIONS[id].args[i].name)
        print('argument input shape:')
        print(FUNCTIONS[id].args[i].sizes)
        # print('avail_fn')
        # print(FUNCTIONS[id].avail_fn)


if __name__ == "__main__":
    for i in range(len(actions.FUNCTIONS)):
        print('*****************')
        getActionSpecFromFunctionID(i)
