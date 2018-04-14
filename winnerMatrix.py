# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:59:40 2018

@author: fanta
"""


def winnerMatrix(size):
    winmatrix = np.zeros([size])
    winmatrix[0] = 0
    winmatrix[1] = 0
    winmatrix[2] = 0
    winmatrix[3] = 0
    winmatrix[4] = 1 # no vehicle mentioned
    winmatrix[5] = 0  # 1 female, young adult, distraction: headphone
    winmatrix[6] = 0 # 1 male, young adult, wait to cross with other pedestrians
    winmatrix[7] = 0 # 2 males, young adults, stopped for vehicle
    winmatrix[8] = 1  # 1 female, teenager, no vehicle mentioned
    winmatrix[9] = 1 # 1 male, 1 female, young adult and midage adult, no vehicle mentioned
    winmatrix[10] = 1 # 1 female, young adult, distraction: headphone, vehicle stopped due to traffic
    winmatrix[11] = 0 # no info
    winmatrix[12] = 0 # 1 male, midage, other vehicle is a cyclist
    winmatrix[13] = 0 # no info, pedestrian step back on pavement
    winmatrix[14] = 1 # 1 female, distraction: headphone, vehicle stooped due to traffic
    winmatrix[15] = 0 # vehicle decelerated, pedestrian slowed down/stopped
    winmatrix[16] = 0 # 1 male, midage adult, pedestrian stepped back on pavement
    winmatrix[17] = 0 # pedstrian looked at other RUsers, no info
    winmatrix[18] = 0 # 1 male, midage, distraction: mobile phone
    winmatrix[19] = 1 # 1 female, young adult, initiated crossing, vehicle decelerated
    winmatrix[20] = 1 # 1 female, midage adult, initiated crossing, vehicle decelerated
    winmatrix[21] = 1 # 2 females, young adults, distraction: talking to each other, vehicle stopped due to other pedestrian
    winmatrix[22] = 1 # 1 female, young adult, vehicle didn't turn in the intersection
    winmatrix[23] = 0 # 3 females, young adults, distraction: mobile phone
    winmatrix[24] = 1 # 1 female, young adult, distraction: headphones, no vehicle mentioned
    winmatrix[25] = 1 # 1 female, young adult, distraction: headphones, vehicle decelerated but 2 other vehicles entered in the intersection, she had to wait them 
    winmatrix[26] = 0 # 2 females, young adults, stepping out on road 
    winmatrix[27] = 0 # 2 males, young adults
    winmatrix[28] = 1 # 1 female, young adult, vehicle decelerated for observed pedestrian and a group in the other side of the crossing
    winmatrix[29] = 0 # 2 males, young adults
    winmatrix[30] = 0 # 1 female, young adult, distraction: heaphones
    winmatrix[31] = 1 # 2 females, young adults, distraction: talking to each other, car stopped for a cyclist
    winmatrix[32] = 1 # 1 female, young adult, no vehicle mentioned
    winmatrix[33] = 0 # 1 female, young audlt, distracion: headphones, looking at observers
    winmatrix[34] = 0 # 2 females, young adults
    winmatrix[35] = 0 # 1 male, 1 female, young adults, distraction: headphones, talking to each other
    winmatrix[36] = 0 # 1 female, young adult, distraction: headphones, overcast
    winmatrix[37] = 0 # 1 male, young adult, sunny
    winmatrix[38] = 1 # 1 female, young adult, hand gestures both sides, overcast
    winmatrix[39] = 0 # 1 male, young adult, overcast
    winmatrix[40] = 0 # 1 male, young adult, overcast
    winmatrix[41] = 0 # 1 male, young addult, overcast
    winmatrix[42] = 0 # 1 male, midage, sunny
    winmatrix[43] = 0 # 1 female, young adult, stepped in front of a group as a leader, sunny
    winmatrix[44] = 1 # 7 males, 7 females, young adults, midage adults, overcast
    winmatrix[45] = 0 # 3 males, young adults, overcast
    winmatrix[46] = 0 # 1 female, midage adult, distraction: mobile phone, overcast
    winmatrix[47] = 0 # 1 male, young adult, overcast
    winmatrix[48] = 0 # 1 female, young adult, sunny
    winmatrix[49] = 0 # 1 female, young adult, sunny, vehicle 2 stopped for vehicle 1 (V-V interaction, obs. 149)
    winmatrix[50] = 0 # 2 females, young adults, overcast
    winmatrix[51] = 0 # 1 female, young adult, overcast, raining
    winmatrix[52] = 1 # 2 females, young adults, hand gesture of drivers to pass, overcast
    winmatrix[53] = 0 # 1 female, young adult, crossing behind a group, overcast
    winmatrix[54] = 0 # 1 male, 1 female, young adults, overcast
    winmatrix[55] = 0 # 1 female, young adult, car parked on her crossing path, she turned around it, overcast
    winmatrix[56] = 0 # 1 male, young adult, overcast
    winmatrix[57] = 0 # 2 females, young adults, sunny
    winmatrix[58] = 1 # 1 female, young adult, vehicle was parking, overcast
    winmatrix[59] = 0 # 1 female, young adult, overcast
    winmatrix[60] = 0 # 1 female, young adult, overcast
    winmatrix[61] = 0 # 1 male, midage adult, overcast
    winmatrix[62] = 0 # 2 females, young adults, crosses with two other girls
    winmatrix[63] = 0 # 3 males, 5 females, vehicle passed in between the observed person and the rest of the group, overcast
    winmatrix[64] = 1 # no info
    winmatrix[65] = 0 # 1 female, young adult, overcast
    winmatrix[66] = 0 # 2 females, young adults, one held the back of the other from crossing, overcast
    winmatrix[67] = 0 # 1 female, young adult, overcast
    winmatrix[68] = 0 # 1 female, young adult, overcast
    winmatrix[69] = 0 # 1 male, young adult, distraction: mobile phone, went around the car as the car has parked, overcast
    winmatrix[70] = 0 # 1 female, young adult, overcast
    winmatrix[71] = 0 # 1 female, young adult, distraction: headphones, overcast
    winmatrix[72] = 0 # 1 female, young adult, distraction: headphones, overcast
    winmatrix[73] = 0 # 1 female, young adult, distraction: headphones, overcast
    winmatrix[74] = 0 # 1 male, young adult, distraction: headphones, overcast
    winmatrix[75] = 1 # 1 male, 2 females, midage adults, overcast
    winmatrix[76] = 1 # 1 female, young adult, had crossing initiated, overcast
    winmatrix[77] = 0 # 1 male, young adult, distraction: headphones, had to stop for several vehicles to pass, overcast
    winmatrix[78] = 1 # 1 male, midage adult, car went behind himwhen he was half-way on the road, overcast
    winmatrix[79] = 0 # 1 male, young adult, distraction: headphones, overcast
    winmatrix[80] = 0 # 1 female, young adult, distraction: headphones, overcast
    winmatrix[81] = 1 # 2 females, midage adults, overcast
    winmatrix[82] = 1 # 1 male, 1 female, young adult, midage adult, overcast 
    winmatrix[83] = 0 # 1 male, midage adult, overcast
    winmatrix[84] = 0 # 1 male, young adult, overcast
    winmatrix[85] = 0 # 1 female, young adult, distraction: headphones, overcast 
    winmatrix[86] = 0 # 1 male, midage adult, distraction: mobile phone
    winmatrix[87] = 1 # 1 female, older adult, overcast
    winmatrix[88] = 0 # 1 female, young adult, overcast
    winmatrix[89] = 1 # 1 male, midage adult, distraction: headphones, overcast
    winmatrix[90] = 0 # 2 females, young adults, overcast
   
    winmatrix[91] = 0 # 1 male, young adult, distraction: headphones, sunny
    winmatrix[92] = 1 # 1 female, young adult, distraction: headphones, sunny
    winmatrix[93] = 0 # 1 female, youn adult, distraction: headphones, sunny
    winmatrix[94] = 0 # 1 male, 1 female, midage adults, overcast
    winmatrix[95] = 0 # 1 male, 1 female, young adults, overcast
    winmatrix[96] = 0 # pedstrian stepped out on road behind a parked truck, no info
    winmatrix[97] = 0 # 2 females, young adults, overcast
    winmatrix[98] = 0 # 2 females, young adults, part of a bigger group, overcast
    winmatrix[99] = 0 # 1 male, young adult, distraction: mobile phone, headphones, overcast
    winmatrix[100] = 0 # 1 male, young adult, distraction: headphones, overcast
    
    
    return winmatrix
    
    
if __name__=="__main__":
    
    win = winnerMatrix(101)
    print(win)