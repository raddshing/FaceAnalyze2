from __future__ import annotations

from typing import Iterable

LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
INNER_LIP = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]
MOUTH_CORNERS = (61, 291)
LEFT_MOUTH_CORNER_GROUP = [61, 185, 40, 39, 37]
RIGHT_MOUTH_CORNER_GROUP = [291, 409, 270, 269, 267]


def dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        idx = int(value)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


# ==========================
# Legacy pair ROI mapping (M8)
# ==========================

# fmt: off
PAIR_L = [109, 108, 107, 67, 69, 66, 103, 104, 105, 54, 68, 63, 21, 71, 70, 162, 139, 156, 127, 34, 143, 35, 124, 46, 53, 52, 65, 55, 226, 113, 225, 224, 223, 222, 221, 193, 130, 247, 30, 29, 27, 28, 56, 189, 190, 33, 246, 161, 160, 159, 158, 157, 173, 133, 7, 163, 144, 145, 153, 154, 155, 25, 110, 24, 23, 22, 26, 112, 243, 244, 245, 122, 31, 228, 229, 230, 231, 232, 233, 111, 117, 118, 119, 120, 121, 128, 234, 227, 116, 100, 47, 114, 188, 142, 126, 217, 174, 196, 209, 198, 236, 3, 129, 49, 131, 134, 51, 102, 48, 115, 220, 45, 64, 219, 218, 237, 44, 235, 166, 79, 239, 98, 240, 59, 75, 60, 20, 238, 241, 125, 242, 141, 99, 97, 93, 137, 123, 50, 101, 36, 203, 205, 206, 132, 177, 147, 187, 207, 58, 215, 213, 172, 138, 192, 216, 92, 165, 167, 212, 186, 57, 185, 40, 39, 37, 61, 76, 62, 184, 74, 73, 72, 183, 42, 41, 38, 78, 191, 80, 81, 82, 95, 88, 178, 87, 96, 89, 179, 86, 77, 90, 180, 85, 146, 91, 181, 84, 202, 43, 106, 182, 83, 204, 194, 201, 214, 210, 211, 32, 208, 135, 169, 170, 140, 171, 136, 150, 149, 176, 148]  # noqa: E501
PAIR_R = [338, 337, 336, 297, 299, 296, 332, 333, 334, 284, 298, 293, 251, 301, 300, 389, 368, 383, 356, 264, 372, 265, 353, 276, 283, 282, 295, 285, 446, 342, 445, 444, 443, 442, 441, 417, 359, 467, 260, 259, 257, 258, 286, 413, 414, 263, 466, 388, 387, 386, 385, 384, 398, 362, 249, 390, 373, 374, 380, 381, 382, 255, 339, 254, 253, 252, 256, 341, 463, 464, 465, 351, 261, 448, 449, 450, 451, 452, 453, 340, 346, 347, 348, 349, 350, 357, 454, 447, 345, 329, 277, 343, 412, 371, 355, 437, 399, 419, 429, 420, 456, 248, 358, 279, 360, 363, 281, 331, 278, 344, 440, 275, 294, 439, 438, 457, 274, 455, 392, 309, 459, 327, 460, 289, 305, 290, 250, 458, 461, 354, 462, 370, 328, 326, 323, 366, 352, 280, 330, 266, 423, 425, 426, 361, 401, 376, 411, 427, 288, 435, 433, 397, 367, 416, 436, 322, 391, 393, 432, 410, 287, 409, 270, 269, 267, 291, 306, 292, 408, 304, 303, 302, 407, 272, 271, 268, 308, 415, 310, 311, 312, 324, 318, 402, 317, 325, 319, 403, 316, 307, 320, 404, 315, 375, 321, 405, 314, 422, 273, 335, 406, 313, 424, 418, 421, 434, 430, 431, 262, 428, 364, 394, 395, 369, 396, 365, 379, 378, 400, 377]  # noqa: E501

ROI_MOUTH = [142, 147, 154, 155, 156, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 172, 173, 174, 176, 177, 178, 179, 181, 182, 183, 185, 186, 187, 189, 190, 191, 193, 194, 195, 197, 198, 199, 200, 202, 203, 205, 206, 207]  # noqa: E501
ROI_EYE = [21, 22, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]  # noqa: E501
ROI_EYEBROW = [2, 5, 8, 11, 14, 17, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 43]

AREA0_GREEN = [161, 162, 163, 165, 166, 167, 168, 169, 170, 172, 173, 174, 176, 177, 178, 179, 181, 182, 183, 185, 186, 187, 189, 190, 191, 193, 194, 195]  # noqa: E501
AREA1_BLUE = dedupe_preserve_order([142, 156, 155, 154, 159, 160, 158, 197, 198, 199] + AREA0_GREEN)  # noqa: E501
AREA2_YELLOW = [102, 140, 121, 142, 156, 157, 155, 154, 159]
AREA3_RED = dedupe_preserve_order([164, 171, 175, 180, 184, 188, 192, 196] + AREA0_GREEN)  # noqa: E501
# fmt: on

# Backward-compatible aliases.
LIST_L = PAIR_L
LIST_R = PAIR_R

ROI_MOUTH_PAIR_IDX = ROI_MOUTH
ROI_EYE_PAIR_IDX = ROI_EYE
ROI_EYEBROW_PAIR_IDX = ROI_EYEBROW
AREA0_GREEN_PAIR_IDX = AREA0_GREEN
AREA1_BLUE_PAIR_IDX = AREA1_BLUE
AREA2_YELLOW_PAIR_IDX = AREA2_YELLOW
AREA3_RED_PAIR_IDX = AREA3_RED


def resolve_pair_indices(pair_idx_list: Iterable[int]) -> tuple[list[int], list[int]]:
    """Resolve pair-index ROI list into left/right MediaPipe landmark indices."""
    unique_pair_idx = dedupe_preserve_order(pair_idx_list)
    max_pair = len(PAIR_L) - 1
    left: list[int] = []
    right: list[int] = []
    for pair_idx in unique_pair_idx:
        idx = int(pair_idx)
        if idx < 0 or idx > max_pair:
            raise ValueError(f"Pair index out of range: {idx} (valid: 0..{max_pair})")
        left.append(PAIR_L[idx])
        right.append(PAIR_R[idx])
    return left, right


def _validate_pair_roi_tables() -> None:
    if len(PAIR_L) != 220 or len(PAIR_R) != 220:
        raise ValueError("PAIR_L and PAIR_R must each have length 220")
    if len(PAIR_L) != len(PAIR_R):
        raise ValueError("PAIR_L and PAIR_R must have the same length")
    if not all(isinstance(value, int) for value in PAIR_L + PAIR_R):
        raise TypeError("PAIR_L and PAIR_R must contain integers only")
    if not all(0 <= value <= 477 for value in PAIR_L + PAIR_R):
        raise ValueError("PAIR_L and PAIR_R values must be in range 0..477")

    pair_idx_lists = [
        ("ROI_MOUTH", ROI_MOUTH),
        ("ROI_EYE", ROI_EYE),
        ("ROI_EYEBROW", ROI_EYEBROW),
        ("AREA0_GREEN", AREA0_GREEN),
        ("AREA1_BLUE", AREA1_BLUE),
        ("AREA2_YELLOW", AREA2_YELLOW),
        ("AREA3_RED", AREA3_RED),
    ]
    max_pair_idx = len(PAIR_L) - 1
    for name, values in pair_idx_lists:
        if not all(isinstance(value, int) for value in values):
            raise TypeError(f"{name} must contain integers only")
        if not all(0 <= value <= max_pair_idx for value in values):
            raise ValueError(f"{name} must contain pair indices in range 0..{max_pair_idx}")


_validate_pair_roi_tables()


# M7 motion viewer region seeds (7 regions).
# Any landmark not listed below is assigned automatically by nearest region centroid.
REGION_LEFT_EYE = LEFT_EYE
REGION_RIGHT_EYE = RIGHT_EYE
REGION_LEFT_BROW = LEFT_BROW
REGION_RIGHT_BROW = RIGHT_BROW
REGION_MOUTH = sorted(
    set(INNER_LIP + LEFT_MOUTH_CORNER_GROUP + RIGHT_MOUTH_CORNER_GROUP + list(MOUTH_CORNERS))
)
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
