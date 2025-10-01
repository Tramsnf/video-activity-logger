import numpy as np

from vap.track import BotSortTracker
from vap.detect import Detection


def make_det(xyxy, feat):
    feature = np.array(feat, dtype=np.float32)
    feature = feature / np.linalg.norm(feature)
    return Detection(xyxy=xyxy, conf=0.9, cls_name="forklift", feature=feature)


def test_botsort_preserves_identity_with_features():
    tracker = BotSortTracker(iou_thresh=0.1, appearance_lambda=0.4, feature_momentum=0.0)

    frame1 = [
        make_det((0.0, 0.0, 10.0, 10.0), [1.0, 0.0, 0.0]),
        make_det((50.0, 0.0, 60.0, 10.0), [0.0, 1.0, 0.0]),
    ]
    out1 = tracker.update(frame1)
    assert {t.track_id for t in out1} == {1, 2}

    # Cross paths: bounding boxes now overlap closer to opposite positions, but features stay aligned
    frame2 = [
        make_det((48.0, 0.0, 58.0, 10.0), [1.0, 0.0, 0.0]),  # feature matches track 1
        make_det((2.0, 0.0, 12.0, 10.0), [0.0, 1.0, 0.0]),  # feature matches track 2
    ]
    out2 = tracker.update(frame2)
    ids_by_pos = {(round(t.xyxy[0]), round(t.xyxy[2])): t.track_id for t in out2}
    assert ids_by_pos[(48, 58)] == 1
    assert ids_by_pos[(2, 12)] == 2
