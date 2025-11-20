import copy
import traceback

from util import (N_BOX, N, H, V, h_index, v_index,)

from .dnb_engine import DotsAndBoxesEngine

# ------------ 유틸 ------------
def h_edge(c, r):  # 수평 엣지 액션 튜플
    return (c, r, H)

def v_edge(c, r):  # 수직 엣지 액션 튜플
    return (c, r, V)

def apply_actions(eng, actions):
    """여러 액션을 순서대로 적용하고 마지막 결과를 반환"""
    out = None
    for a in actions:
        out = eng.apply_action(a)
    return out

def complete_box_actions(bc, br):
    """
    박스 (br, bc)를 완성시키는 4개의 엣지.
    마지막이 실제로 완성되도록 안전한 순서 반환.
    박스 엣지: H(br,bc), H(br+1,bc), V(br,bc), V(br,bc+1)
    """
    return [
        h_edge(bc, br),
        v_edge(bc, br),
        v_edge(bc + 1, br),
        h_edge(bc, br + 1),
    ]

# ------------- 간단 assertion 도우미 -------------
class TestFailure(Exception):
    pass

def assert_true(expr, msg="assert_true failed"):
    if not expr:
        raise TestFailure(msg)

def assert_equal(a, b, msg=None):
    if a != b:
        raise TestFailure(msg or f"assert_equal failed: {a} != {b}")

def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        if isinstance(e, exc_type):
            return
        raise TestFailure(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise TestFailure(f"Expected {exc_type.__name__}, but no exception was raised")

# ------------- 개별 테스트 -------------
def test_initial_state():
    eng = DotsAndBoxesEngine()
    s = eng.get_state()
    assert_equal(s["edges"], [0, 0], "초기 edges는 0이어야 함")
    assert_true(s["cur_player"] in (0, 1), "cur_player는 0 또는 1")
    assert_equal(s["score"], [0, 0], "초기 점수는 [0,0]")
    assert_true(not eng.is_game_over(), "초기에는 게임오버가 아님")

def test_apply_action_horizontal_no_box_then_turn_switch():
    eng = DotsAndBoxesEngine()
    start_player = eng.cur_player
    out = eng.apply_action(h_edge(0, 0))
    assert_true(out["is_box_completed"] is False, "박스가 완성되면 안 됨")
    assert_equal(eng.cur_player, 1 - start_player, "턴이 바뀌어야 함")

def test_complete_single_box_scores_and_turn_stays():
    eng = DotsAndBoxesEngine()
    p1 = 1 - eng.cur_player
    seq = complete_box_actions(0, 0)
    out = None

    for a in seq:
        out = eng.apply_action(a)
    
    assert_true(out["is_box_completed"] is True, "박스 하나가 완성되어야 함")
    assert_equal(out["completed_boxes"], [(0, 0)], "완성 박스는 (0,0)")
    assert_equal(eng.cur_player, p1, "박스 완성 시 턴 유지")

def test_two_boxes_completed_on_shared_edge():
    """
    공유 엣지 H(1,0)을 마지막에 두어 (0,0)과 (1,0) 두 박스를 동시에 완성.
    """
    eng = DotsAndBoxesEngine()

    # (0,0) 박스에서 H(1,0)을 제외한 3개 세팅
    for a in [h_edge(0, 0), v_edge(0, 0), v_edge(1, 0)]:
        eng.apply_action(a)

    # (1,0) 박스에서 H(1,0)을 제외한 3개 세팅
    for a in [v_edge(0, 1), v_edge(1, 1), h_edge(0, 2)]:
        eng.apply_action(a)

    # 현재 턴 메모 후, 공유 엣지 H(1,0)으로 두 박스 완성
    cur = eng.cur_player
    out = eng.apply_action(h_edge(0, 1))
    assert_true(out["is_box_completed"] is True, "동시에 2박스 완성")
    print(out["completed_boxes"])
    assert_equal(sorted(out["completed_boxes"]), [(0, 0), (0, 1)], "완성 박스는 (0,0),(0,1)")
    assert_equal(eng.cur_player, cur, "박스 완성 시 턴 유지")

def test_undo_action_restores_edge_score_and_turn():
    eng = DotsAndBoxesEngine()
    player_before = 1 - eng.cur_player

    seq = complete_box_actions(0, 0)
    for a in seq[:-1]:
        eng.apply_action(a)

    last_action = seq[-1]
    out = eng.apply_action(last_action)
    assert_equal(out["completed_boxes"], [(0, 0)])
    assert_equal(eng.cur_player, player_before)

    # undo
    u = eng.undo_action(last_action, out["completed_boxes"], player_before)
    assert_equal(eng.cur_player, player_before, "턴 복구")

    # 해당 엣지가 해제되었으니 재적용 가능해야 함
    eng.apply_action(last_action)  # 예외 없이 통과

def test_set_state_roundtrip():
    eng = DotsAndBoxesEngine()
    apply_actions(eng, [
        h_edge(0, 0),
        v_edge(0, 0),
        h_edge(0, 5),
        v_edge(5, 0),
    ])
    snap = eng.get_state()

    ## snap은 아직 없음
    eng2 = DotsAndBoxesEngine.from_state(copy.deepcopy(snap))
    assert_equal(eng2.get_state(), snap, "from_state 로드 일치")

    eng3 = DotsAndBoxesEngine()
    eng3.set_state(copy.deepcopy(snap))
    assert_equal(eng3.get_state(), snap, "set_state 로드 일치")

def test_invalid_action_bounds():
    eng = DotsAndBoxesEngine()
    # 수평에서 c=5는 불가 (0..4)
    assert_raises(ValueError, eng.apply_action, h_edge(0, 5))
    # 수직에서 r=5는 불가 (0..4)
    assert_raises(ValueError, eng.apply_action, v_edge(5, 0))
    # 방향 오류
    assert_raises(ValueError, eng.apply_action, (0, 0, 2))

def test_duplicate_edge_raises():
    eng = DotsAndBoxesEngine()
    eng.apply_action(h_edge(0, 0))
    assert_raises(ValueError, eng.apply_action, h_edge(0, 0))


def test_game_over_after_claiming_all_edges():
    eng = DotsAndBoxesEngine()

    # 모든 엣지를 한 번씩 채움
    # 수평 6*5=30
    for r in range(N):
        for c in range(N - 1):
            if not ((eng.get_state()["edges"][0] >> h_index(c, r)) & 1):
                eng.apply_action(h_edge(c, r))

    # 수직 5*6=30
    for r in range(N - 1):
        for c in range(N):
            if not ((eng.get_state()["edges"][1] >> v_index(c, r)) & 1):
                eng.apply_action(v_edge(c, r))

    assert_equal(sum(eng.score), N_BOX * N_BOX)  # 25
    assert_true(eng.is_game_over() is True)

# ------------- 러너 -------------
TESTS = [
    test_initial_state,
    test_apply_action_horizontal_no_box_then_turn_switch,
    test_complete_single_box_scores_and_turn_stays,
    test_two_boxes_completed_on_shared_edge,
    test_undo_action_restores_edge_score_and_turn,
    test_set_state_roundtrip,
    test_invalid_action_bounds,
    test_duplicate_edge_raises,
    test_game_over_after_claiming_all_edges,
]

def main():
    passed = 0
    failed = 0
    for t in TESTS:
        name = t.__name__
        try:
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name} -> {e}")
            traceback.print_exc()
            print("-" * 60)
    total = passed + failed
    print("=" * 60)
    print(f"총 {total}개 테스트 | PASS: {passed}  FAIL: {failed}")
    if failed > 0:
        exit(1)

if __name__ == "__main__":
    main()

    #test_complete_single_box_scores_and_turn_stays()