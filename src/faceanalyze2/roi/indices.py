LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MOUTH_CORNERS = (61, 291)
LEFT_MOUTH_CORNER_GROUP = [61, 185, 40, 39, 37]
RIGHT_MOUTH_CORNER_GROUP = [291, 409, 270, 269, 267]

# M7 motion viewer region seeds (7 regions).
# Any landmark not listed below is assigned automatically by nearest region centroid.
REGION_LEFT_EYE = LEFT_EYE
REGION_RIGHT_EYE = RIGHT_EYE
REGION_LEFT_BROW = LEFT_BROW
REGION_RIGHT_BROW = RIGHT_BROW
REGION_MOUTH = sorted(set(INNER_LIP + LEFT_MOUTH_CORNER_GROUP + RIGHT_MOUTH_CORNER_GROUP + list(MOUTH_CORNERS)))
REGION_NOSE = [1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 326, 327, 331]
REGION_JAW = [
    10,
    21,
    54,
    58,
    67,
    93,
    103,
    109,
    127,
    132,
    136,
    148,
    149,
    150,
    152,
    162,
    172,
    176,
    234,
    251,
    284,
    288,
    297,
    323,
    332,
    338,
    356,
    361,
    365,
    377,
    378,
    379,
    389,
    397,
    400,
    454,
]
