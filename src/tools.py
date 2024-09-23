from typing import Any, List

def update_dict_of_list(key:Any,value:Any,dictionnary:dict[Any,List])->dict:
    if key not in list(dictionnary.keys()): 
        dictionnary[key] = [value]
    else: dictionnary[key].append(value)
    return dictionnary
