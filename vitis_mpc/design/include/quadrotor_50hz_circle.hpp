#pragma once

#include "admm.hpp"

tinytype Xref_data[NTOTAL*NSTATES] = {
  -1.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0157905,	1.2565048,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9996842,	0.0251301,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0473616,	1.2557111,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9987370,	0.0502443,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0789028,	1.2541244,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9971589,	0.0753268,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1103942,	1.2517455,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9949510,	0.1003617,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1418158,	1.2485759,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9921147,	0.1253332,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1731478,	1.2446178,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9886517,	0.1502256,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2043705,	1.2398735,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9845643,	0.1750231,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2354641,	1.2343461,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9798551,	0.1997100,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2664090,	1.2280390,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9745269,	0.2242708,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2971856,	1.2209563,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9685832,	0.2486899,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3277745,	1.2131024,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9620277,	0.2729519,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3581563,	1.2044823,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9548645,	0.2970416,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3883120,	1.1951014,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9470983,	0.3209436,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4182224,	1.1849657,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9387339,	0.3446429,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4478686,	1.1740815,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9297765,	0.3681246,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4772319,	1.1624557,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9202318,	0.3913737,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5062938,	1.1500957,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9101060,	0.4143756,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5350360,	1.1370093,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8994053,	0.4371158,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5634401,	1.1232047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8881364,	0.4595799,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5914884,	1.1086907,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8763067,	0.4817537,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6191631,	1.0934764,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8639234,	0.5036232,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6464468,	1.0775714,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8509945,	0.5251746,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6733221,	1.0609858,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8375280,	0.5463943,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6997721,	1.0437301,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8235326,	0.5672689,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7257802,	1.0258152,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8090170,	0.5877853,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7513298,	1.0072523,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7939904,	0.6079303,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7764049,	0.9880532,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7784623,	0.6276914,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8009895,	0.9682300,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7624425,	0.6470560,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8250683,	0.9477953,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7459411,	0.6660119,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8486259,	0.9267619,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7289686,	0.6845471,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8716475,	0.9051432,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7115357,	0.7026500,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8941186,	0.8829528,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6936533,	0.7203090,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9160249,	0.8602046,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6753328,	0.7375131,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9373526,	0.8369132,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6565858,	0.7542514,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9580883,	0.8130931,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6374240,	0.7705132,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9782188,	0.7887595,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6178596,	0.7862884,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9977315,	0.7639276,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5979050,	0.8015670,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0166140,	0.7386133,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5775727,	0.8163393,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0348543,	0.7128324,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5568756,	0.8305959,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0524411,	0.6866013,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5358268,	0.8443279,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0693631,	0.6599365,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5144395,	0.8575267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0856096,	0.6328549,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4927273,	0.8701838,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1011705,	0.6053736,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4707039,	0.8822912,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1160358,	0.5775099,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4483832,	0.8938414,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1301962,	0.5492814,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4257793,	0.9048271,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1436428,	0.5207060,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4029064,	0.9152412,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1563670,	0.4918017,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3797791,	0.9250772,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1683608,	0.4625868,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3564119,	0.9343289,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1796167,	0.4330797,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3328195,	0.9429905,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1901275,	0.4032990,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3090170,	0.9510565,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1998866,	0.3732636,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2850193,	0.9585218,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2088878,	0.3429925,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2608415,	0.9653816,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2171255,	0.3125047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2364990,	0.9716317,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2245944,	0.2818195,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2120071,	0.9772681,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2312898,	0.2509564,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1873813,	0.9822873,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2372075,	0.2199347,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1626372,	0.9866859,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2423437,	0.1887741,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1377903,	0.9904614,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2466953,	0.1574942,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1128564,	0.9936113,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2502594,	0.1261149,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0878512,	0.9961336,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2530339,	0.0946560,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0627905,	0.9980267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2550168,	0.0631372,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0376902,	0.9992895,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2562071,	0.0315786,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0125660,	0.9999210,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2566040,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0125660,	0.9999210,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2562071,	-0.0315786,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0376902,	0.9992895,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2550168,	-0.0631372,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0627905,	0.9980267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2530339,	-0.0946560,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0878512,	0.9961336,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2502594,	-0.1261149,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1128564,	0.9936113,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2466953,	-0.1574942,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1377903,	0.9904614,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2423437,	-0.1887741,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1626372,	0.9866859,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2372075,	-0.2199347,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1873813,	0.9822873,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2312898,	-0.2509564,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2120071,	0.9772681,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2245944,	-0.2818195,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2364990,	0.9716317,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2171255,	-0.3125047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2608415,	0.9653816,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.2088878,	-0.3429925,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2850193,	0.9585218,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1998866,	-0.3732636,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3090170,	0.9510565,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1901275,	-0.4032990,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3328195,	0.9429905,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1796167,	-0.4330797,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3564119,	0.9343289,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1683608,	-0.4625868,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3797791,	0.9250772,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1563670,	-0.4918017,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4029064,	0.9152412,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1436428,	-0.5207060,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4257793,	0.9048271,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1301962,	-0.5492814,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4483832,	0.8938414,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1160358,	-0.5775099,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4707039,	0.8822912,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.1011705,	-0.6053736,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4927273,	0.8701838,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0856096,	-0.6328549,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5144395,	0.8575267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0693631,	-0.6599365,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5358268,	0.8443279,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0524411,	-0.6866013,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5568756,	0.8305959,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0348543,	-0.7128324,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5775727,	0.8163393,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0166140,	-0.7386133,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5979050,	0.8015670,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9977315,	-0.7639276,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6178596,	0.7862884,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9782188,	-0.7887595,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6374240,	0.7705132,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9580883,	-0.8130931,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6565858,	0.7542514,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9373526,	-0.8369132,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6753328,	0.7375131,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.9160249,	-0.8602046,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6936533,	0.7203090,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8941186,	-0.8829528,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7115357,	0.7026500,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8716475,	-0.9051432,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7289686,	0.6845471,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8486259,	-0.9267619,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7459411,	0.6660119,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8250683,	-0.9477953,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7624425,	0.6470560,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.8009895,	-0.9682300,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7784623,	0.6276914,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7764049,	-0.9880532,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7939904,	0.6079303,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7513298,	-1.0072523,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8090170,	0.5877853,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.7257802,	-1.0258152,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8235326,	0.5672689,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6997721,	-1.0437301,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8375280,	0.5463943,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6733221,	-1.0609858,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8509945,	0.5251746,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6464468,	-1.0775714,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8639234,	0.5036232,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.6191631,	-1.0934764,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8763067,	0.4817537,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5914884,	-1.1086907,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8881364,	0.4595799,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5634401,	-1.1232047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8994053,	0.4371158,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5350360,	-1.1370093,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9101060,	0.4143756,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.5062938,	-1.1500957,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9202318,	0.3913737,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4772319,	-1.1624557,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9297765,	0.3681246,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4478686,	-1.1740815,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9387339,	0.3446429,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.4182224,	-1.1849657,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9470983,	0.3209436,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3883120,	-1.1951014,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9548645,	0.2970416,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3581563,	-1.2044823,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9620277,	0.2729519,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3277745,	-1.2131024,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9685832,	0.2486899,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2971856,	-1.2209563,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9745269,	0.2242708,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2664090,	-1.2280390,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9798551,	0.1997100,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2354641,	-1.2343461,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9845643,	0.1750231,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.2043705,	-1.2398735,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9886517,	0.1502256,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1731478,	-1.2446178,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9921147,	0.1253332,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1418158,	-1.2485759,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9949510,	0.1003617,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1103942,	-1.2517455,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9971589,	0.0753268,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0789028,	-1.2541244,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9987370,	0.0502443,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0473616,	-1.2557111,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9996842,	0.0251301,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0157905,	-1.2565048,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  1.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0157905,	-1.2565048,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9996842,	-0.0251301,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0473616,	-1.2557111,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9987370,	-0.0502443,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0789028,	-1.2541244,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9971589,	-0.0753268,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1103942,	-1.2517455,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9949510,	-0.1003617,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1418158,	-1.2485759,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9921147,	-0.1253332,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1731478,	-1.2446178,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9886517,	-0.1502256,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2043705,	-1.2398735,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9845643,	-0.1750231,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2354641,	-1.2343461,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9798551,	-0.1997100,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2664090,	-1.2280390,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9745269,	-0.2242708,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2971856,	-1.2209563,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9685832,	-0.2486899,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3277745,	-1.2131024,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9620277,	-0.2729519,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3581563,	-1.2044823,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9548645,	-0.2970416,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3883120,	-1.1951014,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9470983,	-0.3209436,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4182224,	-1.1849657,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9387339,	-0.3446429,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4478686,	-1.1740815,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9297765,	-0.3681246,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4772319,	-1.1624557,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9202318,	-0.3913737,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5062938,	-1.1500957,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.9101060,	-0.4143756,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5350360,	-1.1370093,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8994053,	-0.4371158,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5634401,	-1.1232047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8881364,	-0.4595799,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5914884,	-1.1086907,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8763067,	-0.4817537,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6191631,	-1.0934764,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8639234,	-0.5036232,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6464468,	-1.0775714,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8509945,	-0.5251746,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6733221,	-1.0609858,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8375280,	-0.5463943,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6997721,	-1.0437301,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8235326,	-0.5672689,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7257802,	-1.0258152,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.8090170,	-0.5877853,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7513298,	-1.0072523,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7939904,	-0.6079303,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7764049,	-0.9880532,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7784623,	-0.6276914,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8009895,	-0.9682300,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7624425,	-0.6470560,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8250683,	-0.9477953,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7459411,	-0.6660119,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8486259,	-0.9267619,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7289686,	-0.6845471,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8716475,	-0.9051432,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.7115357,	-0.7026500,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8941186,	-0.8829528,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6936533,	-0.7203090,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9160249,	-0.8602046,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6753328,	-0.7375131,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9373526,	-0.8369132,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6565858,	-0.7542514,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9580883,	-0.8130931,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6374240,	-0.7705132,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9782188,	-0.7887595,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.6178596,	-0.7862884,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9977315,	-0.7639276,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5979050,	-0.8015670,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0166140,	-0.7386133,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5775727,	-0.8163393,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0348543,	-0.7128324,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5568756,	-0.8305959,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0524411,	-0.6866013,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5358268,	-0.8443279,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0693631,	-0.6599365,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.5144395,	-0.8575267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0856096,	-0.6328549,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4927273,	-0.8701838,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1011705,	-0.6053736,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4707039,	-0.8822912,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1160358,	-0.5775099,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4483832,	-0.8938414,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1301962,	-0.5492814,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4257793,	-0.9048271,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1436428,	-0.5207060,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.4029064,	-0.9152412,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1563670,	-0.4918017,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3797791,	-0.9250772,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1683608,	-0.4625868,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3564119,	-0.9343289,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1796167,	-0.4330797,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3328195,	-0.9429905,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1901275,	-0.4032990,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.3090170,	-0.9510565,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1998866,	-0.3732636,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2850193,	-0.9585218,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2088878,	-0.3429925,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2608415,	-0.9653816,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2171255,	-0.3125047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2364990,	-0.9716317,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2245944,	-0.2818195,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.2120071,	-0.9772681,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2312898,	-0.2509564,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1873813,	-0.9822873,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2372075,	-0.2199347,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1626372,	-0.9866859,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2423437,	-0.1887741,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1377903,	-0.9904614,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2466953,	-0.1574942,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.1128564,	-0.9936113,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2502594,	-0.1261149,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0878512,	-0.9961336,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2530339,	-0.0946560,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0627905,	-0.9980267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2550168,	-0.0631372,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0376902,	-0.9992895,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2562071,	-0.0315786,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0125660,	-0.9999210,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2566040,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0125660,	-0.9999210,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2562071,	0.0315786,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0376902,	-0.9992895,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2550168,	0.0631372,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0627905,	-0.9980267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2530339,	0.0946560,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.0878512,	-0.9961336,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2502594,	0.1261149,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1128564,	-0.9936113,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2466953,	0.1574942,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1377903,	-0.9904614,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2423437,	0.1887741,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1626372,	-0.9866859,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2372075,	0.2199347,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.1873813,	-0.9822873,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2312898,	0.2509564,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2120071,	-0.9772681,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2245944,	0.2818195,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2364990,	-0.9716317,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2171255,	0.3125047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2608415,	-0.9653816,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.2088878,	0.3429925,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.2850193,	-0.9585218,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1998866,	0.3732636,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3090170,	-0.9510565,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1901275,	0.4032990,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3328195,	-0.9429905,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1796167,	0.4330797,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3564119,	-0.9343289,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1683608,	0.4625868,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.3797791,	-0.9250772,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1563670,	0.4918017,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4029064,	-0.9152412,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1436428,	0.5207060,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4257793,	-0.9048271,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1301962,	0.5492814,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4483832,	-0.8938414,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1160358,	0.5775099,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4707039,	-0.8822912,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.1011705,	0.6053736,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.4927273,	-0.8701838,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0856096,	0.6328549,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5144395,	-0.8575267,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0693631,	0.6599365,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5358268,	-0.8443279,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0524411,	0.6866013,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5568756,	-0.8305959,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0348543,	0.7128324,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5775727,	-0.8163393,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-1.0166140,	0.7386133,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.5979050,	-0.8015670,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9977315,	0.7639276,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6178596,	-0.7862884,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9782188,	0.7887595,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6374240,	-0.7705132,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9580883,	0.8130931,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6565858,	-0.7542514,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9373526,	0.8369132,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6753328,	-0.7375131,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.9160249,	0.8602046,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.6936533,	-0.7203090,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8941186,	0.8829528,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7115357,	-0.7026500,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8716475,	0.9051432,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7289686,	-0.6845471,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8486259,	0.9267619,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7459411,	-0.6660119,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8250683,	0.9477953,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7624425,	-0.6470560,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.8009895,	0.9682300,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7784623,	-0.6276914,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7764049,	0.9880532,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.7939904,	-0.6079303,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7513298,	1.0072523,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8090170,	-0.5877853,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.7257802,	1.0258152,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8235326,	-0.5672689,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6997721,	1.0437301,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8375280,	-0.5463943,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6733221,	1.0609858,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8509945,	-0.5251746,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6464468,	1.0775714,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8639234,	-0.5036232,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.6191631,	1.0934764,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8763067,	-0.4817537,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5914884,	1.1086907,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8881364,	-0.4595799,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5634401,	1.1232047,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.8994053,	-0.4371158,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5350360,	1.1370093,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9101060,	-0.4143756,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.5062938,	1.1500957,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9202318,	-0.3913737,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4772319,	1.1624557,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9297765,	-0.3681246,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4478686,	1.1740815,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9387339,	-0.3446429,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.4182224,	1.1849657,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9470983,	-0.3209436,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3883120,	1.1951014,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9548645,	-0.2970416,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3581563,	1.2044823,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9620277,	-0.2729519,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.3277745,	1.2131024,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9685832,	-0.2486899,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2971856,	1.2209563,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9745269,	-0.2242708,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2664090,	1.2280390,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9798551,	-0.1997100,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2354641,	1.2343461,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9845643,	-0.1750231,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2043705,	1.2398735,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9886517,	-0.1502256,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1731478,	1.2446178,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9921147,	-0.1253332,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1418158,	1.2485759,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9949510,	-0.1003617,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.1103942,	1.2517455,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9971589,	-0.0753268,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0789028,	1.2541244,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9987370,	-0.0502443,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0473616,	1.2557111,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -0.9996842,	-0.0251301,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0157905,	1.2565048,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	
  -1.0000000,	-0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000	
};