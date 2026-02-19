from __future__ import annotations

from faceanalyze2.roi.indices import (
    AREA0_GREEN,
    AREA0_GREEN_PAIR_IDX,
    AREA1_BLUE,
    AREA1_BLUE_PAIR_IDX,
    AREA2_YELLOW,
    AREA2_YELLOW_PAIR_IDX,
    AREA3_RED,
    AREA3_RED_PAIR_IDX,
    LIST_L,
    LIST_R,
    PAIR_L,
    PAIR_R,
    ROI_EYE,
    ROI_EYE_PAIR_IDX,
    ROI_EYEBROW,
    ROI_EYEBROW_PAIR_IDX,
    ROI_MOUTH,
    ROI_MOUTH_PAIR_IDX,
    resolve_pair_indices,
)


def test_pair_tables_have_expected_size_and_range() -> None:
    assert len(PAIR_L) == 220
    assert len(PAIR_R) == 220
    assert PAIR_L == LIST_L
    assert PAIR_R == LIST_R
    assert min(PAIR_L) >= 0
    assert min(PAIR_R) >= 0
    assert max(PAIR_L) < 478
    assert max(PAIR_R) < 478


def test_indirect_roi_indices_are_within_pair_table_range() -> None:
    assert ROI_MOUTH == ROI_MOUTH_PAIR_IDX
    assert ROI_EYE == ROI_EYE_PAIR_IDX
    assert ROI_EYEBROW == ROI_EYEBROW_PAIR_IDX
    assert AREA0_GREEN == AREA0_GREEN_PAIR_IDX
    assert AREA1_BLUE == AREA1_BLUE_PAIR_IDX
    assert AREA2_YELLOW == AREA2_YELLOW_PAIR_IDX
    assert AREA3_RED == AREA3_RED_PAIR_IDX

    all_indices = (
        ROI_MOUTH_PAIR_IDX
        + ROI_EYE_PAIR_IDX
        + ROI_EYEBROW_PAIR_IDX
        + AREA0_GREEN_PAIR_IDX
        + AREA1_BLUE_PAIR_IDX
        + AREA2_YELLOW_PAIR_IDX
        + AREA3_RED_PAIR_IDX
    )
    assert all(0 <= idx < 220 for idx in all_indices)


def test_area_lists_include_area0_without_duplicates() -> None:
    assert all(idx in AREA1_BLUE_PAIR_IDX for idx in AREA0_GREEN_PAIR_IDX)
    assert all(idx in AREA3_RED_PAIR_IDX for idx in AREA0_GREEN_PAIR_IDX)
    assert len(AREA1_BLUE_PAIR_IDX) == len(set(AREA1_BLUE_PAIR_IDX))
    assert len(AREA3_RED_PAIR_IDX) == len(set(AREA3_RED_PAIR_IDX))


def test_resolve_pair_indices_deduplicates_and_maps() -> None:
    left, right = resolve_pair_indices([5, 7, 5, 8])
    assert left == [PAIR_L[5], PAIR_L[7], PAIR_L[8]]
    assert right == [PAIR_R[5], PAIR_R[7], PAIR_R[8]]
    assert len(left) == len(right)
    assert all(0 <= idx < 478 for idx in left)
    assert all(0 <= idx < 478 for idx in right)
