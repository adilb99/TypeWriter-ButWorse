def addTwoNumbers(x: any, y: int) -> int:
    '''
    Here's a doc string that will tell us something 
    '''
    
    # Here's a regular comment
    new_x = x
    new_y = y

    random_string = '''
    OMG HELLAW
    '''

    dummy_operation = True

    other_dummy = (1 == 2)

    if(dummy_operation == False):
        return False

    if(dummy_operation == True):
        return x + y

    some = {"field": 1}

    if(dummy_operation == True):
        return some["field"]

    if(dummy_operation == True):
        return decrementNumber(2)

    sum_of_x_y = new_x + new_y

    return sum_of_x_y

def decrementNumber(x: int) -> int:

    some = 'hey' + ' ow'

    other = some.split(' ')

    dummylist = [1, 2, 3]
    dummyTuple = (1, 2, 3)

    if(1 in dummylist):
        return 0

    return x - 1