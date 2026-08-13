"""One-off: 60 s 1080p benchmark corpus + measurement (docs/BENCHMARKS.md)."""
import json
import time

import av
import cv2

from rr_vfiqa.testing.synth import _iter_frames, _mux

N, W, H, FPS = 7200, 1920, 1080, 120


def odd_defect(t, frame, last_even):
    ph = t / N
    if 0.30 < ph < 0.36:          # blur segment
        return cv2.GaussianBlur(frame, (0, 0), 3.0)
    if 0.60 < ph < 0.66:          # freeze segment
        return last_even.copy()
    return frame


t0 = time.perf_counter()
with av.open("_b1080/source_60.mp4", "w") as osrc, \
     av.open("_b1080/candidate_good_120.mp4", "w") as ogood, \
     av.open("_b1080/candidate_bad_120.mp4", "w") as obad:
    ss = osrc.add_stream("h264", rate=FPS // 2)
    sg = ogood.add_stream("h264", rate=FPS)
    sb = obad.add_stream("h264", rate=FPS)
    for st in (ss, sg, sb):
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "18", "preset": "medium"}
    last_even = None
    for t, frame, *_ in _iter_frames(N, W, H, 7):
        if t % 2 == 0:
            last_even = frame.copy()
            _mux(osrc, ss, frame)
            _mux(ogood, sg, frame)
            _mux(obad, sb, frame)
        else:
            _mux(ogood, sg, frame)
            _mux(obad, sb, odd_defect(t, frame, last_even))
        if t % 720 == 0:
            print(f"render {t}/{N} ({time.perf_counter()-t0:.0f}s)", flush=True)
    for out, st in ((osrc, ss), (ogood, sg), (obad, sb)):
        for packet in st.encode():
            out.mux(packet)
t_render = time.perf_counter() - t0
print(f"render done in {t_render:.0f}s", flush=True)

from rr_vfiqa.benchmark import benchmark

res = benchmark("_b1080/source_60.mp4",
                ["_b1080/candidate_good_120.mp4", "_b1080/candidate_bad_120.mp4"],
                preset="fast", cache_dir="_b1080/cache", device="cuda",
                flow_backend="raft", labels=["good", "bad"])
res["render_seconds"] = round(t_render, 1)
res["video"] = {"duration_s": N / FPS, "resolution": f"{W}x{H}",
                "source_frames": N // 2, "candidate_frames": N}
with open("_b1080/result.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2), flush=True)
