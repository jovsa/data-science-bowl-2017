import os

def verify_location(loc):
    if os.path.isdir(loc) or os.path.isfile(loc) :
        print('Found and verified location: ' + loc)
    else:
        raise Exception('Failed to verify location: ' + loc)
    return loc

def folder_explorer(folder):
    folder_info = {}
    for d in os.listdir(folder):
        folder_info[d] = int(len(os.listdir(folder + d)))
    return folder_info

def temp():
    print("You are now in temp")
