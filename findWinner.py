
from random import randint

#return 0 if the vehicle is the winner, 1 if the pedestrian is the winner
def findWinner(seq_human_readable, descriptors_human_readable):
	#TODO FANTA 
    """
    This function is not using information from descriptors_human_readable, it is quite complex and sometimes empty
    I put some conditions on some actions (from my observations on the sequences), I think they determine quite precisely the winner
    If the conditions are not entered, I decide ramdomly the winner
    """

    if(seq_human_readable.find("Approaching Phase: Driver / Vehicle Analysis_Vehicle Movement_Passed the pedestrian") > 0 or \
       seq_human_readable.find("Crossing Phase: Driver / Vehicle Analysis_Vehicle Movement_Passed the pedestrian") > 0 \
       ):
        winner = 0
    
    elif(seq_human_readable.find("Crossing Phase: Pedestrian Analysis_Hand Movements_Waved Hand") > 0 or \
         seq_human_readable.find("Approaching Phase: Driver / Vehicle Analysis_Vehicle Movement_Decelerated due to other pedestrians") > 0 or \
         seq_human_readable.find("Approaching Phase: Driver / Vehicle Analysis_Vehicle Movement_Stopped due to other pedestrian") > 0):
        winner = 1
        
    elif(seq_human_readable.find("Pedestrian") > 0 and seq_human_readable.find("Vehicle") < 0):
        winner = 1 
    
    else:
        print("Random winner")
        winner = randint(0, 1)
    
    print("is " + str(winner))
    return winner