"""
viewer.py — interactive viser viewer for threshold-pruning a trained 3DGS scene.

Drag the threshold sliders and watch Gaussians disappear, with live kept count, estimated size,
render time and PSNR of the current view vs the unpruned render. Doubles as a sanity check of the
mask logic and as a way to eyeball what the EA's thresholds do before running it.

    python -m easplatting.viewer --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
        --cameras data/360_v2/bonsai --images images_2          # --cameras is optional
    # then open http://localhost:8080
    # Headless / cluster: binds to 127.0.0.1 only; from your laptop run
    #   ssh -L 8080:<gpu-node>:8080 <login-host>     (or -L 8080:localhost:8080 <host> if you are on the node)

Sliders threshold ACTIVATED values (sigmoid opacity, exp scale), in log10 space; the top/left end
of each slider is "off" (keeps everything). A Gaussian is kept when ALL conditions pass:
    opacity >= min_opacity  and  max(scale) <= max_scale  and  max(scale)/min(scale) <= max_anisotropy

View modes: pruned | unpruned | removed only (what the thresholds delete) | error x5 (|pruned - unpruned|).

Pass --cameras to get a "go to camera" dropdown (and a sensible start pose / up direction); a
gsplat-trainer scene (.pt) needs the exported cameras.json, as everywhere else.

Requires: viser and nerfview (installed with gsplat's trainer deps), a CUDA GPU.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from easplatting.cameras import Cameras, load_cameras, split_indices
from easplatting.gs_loader import activate, bytes_per_gaussian, load_scene, prune, save_ply
from easplatting.metrics import psnr_fast
from easplatting.render import render_views
from easplatting.thresholds import gene_stats, keep_mask, slider_bounds

VIEW_MODES = ("pruned", "unpruned", "removed only", "error x5")


# ------------------------------------------------------------------------ rendering

class ViewerState:
    """Scene + current thresholds; shared between the GUI callbacks and the render thread."""

    def __init__(self, raw, act, device):
        self.raw, self.act, self.device = raw, act, device
        self.stats = gene_stats(act)
        self.bounds = slider_bounds(self.stats)
        self.n_total = act["means"].shape[0]
        self.mask = torch.ones(self.n_total, dtype=torch.bool, device=device)
        self.thresholds = dict(min_opacity=0.0, max_scale=float("inf"), max_anisotropy=float("inf"))
        self.view_mode = "pruned"
        self.show_psnr = True
        self._cache = (None, None, None)          # (mask it was built for, kept, removed)

    def set_log_thresholds(self, op_log, scale_log, aniso_log):
        lo_op, hi_op = self.bounds["min_opacity"]
        _, hi_s = self.bounds["max_scale"]
        _, hi_a = self.bounds["max_anisotropy"]
        # slider ends mean "off": store those as no-op thresholds so saved JSONs read cleanly
        th = dict(min_opacity=0.0 if op_log <= lo_op + 1e-6 else 10 ** op_log,
                  max_scale=float("inf") if scale_log >= hi_s - 1e-6 else 10 ** scale_log,
                  max_anisotropy=float("inf") if aniso_log >= hi_a - 1e-6 else 10 ** aniso_log)
        self.thresholds = th
        self.mask = keep_mask(self.stats, **th)

    def splats(self):
        """(mask, kept, removed) dicts for the current mask; rebuilt only when the mask changed."""
        mask = self.mask
        if self._cache[0] is not mask:
            self._cache = (mask, prune(self.act, mask), prune(self.act, ~mask))
        return self._cache


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def render_view(state, viewmat, K, W, H):
    """One frame in the current view mode -> (H, W, 3) float in [0, 1], plus stats."""
    cams = Cameras(names=["v"], viewmats=viewmat[None], Ks=K[None], widths=[W], heights=[H])
    mask, kept, removed = state.splats()
    mode, n_kept = state.view_mode, int(mask.sum())

    _sync(state.device)
    t0 = time.perf_counter()
    if mode == "unpruned":
        img = render_views(state.act, cams)
    elif mode == "removed only":
        img = render_views(removed, cams) if n_kept < state.n_total else torch.zeros(1, H, W, 3, device=state.device)
    else:
        img = render_views(kept, cams)
    _sync(state.device)
    ms = (time.perf_counter() - t0) * 1e3          # time of the image shown, not of the PSNR reference

    db = None
    if mode in ("pruned", "error x5") and (state.show_psnr or mode == "error x5"):
        full = render_views(state.act, cams)
        db = float(psnr_fast(img, full, max_db=99.0)[0])
        if mode == "error x5":
            img = ((img - full).abs().mean(-1, keepdim=True) * 5).clamp(0, 1).expand_as(img)
    return img[0], dict(ms=ms, n_kept=n_kept, psnr=db)


# ------------------------------------------------------------------------------ app

def camera_to_viser(cams, i):
    """Dataset camera i -> (position, wxyz, fov) for a viser client camera (OpenCV frame)."""
    import viser.transforms as vt
    c2w = torch.linalg.inv(cams.viewmats[i].double()).numpy()
    fy = float(cams.Ks[i][1, 1])
    return c2w[:3, 3], vt.SO3.from_matrix(c2w[:3, :3]).wxyz, 2 * math.atan(cams.heights[i] / (2 * fy))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="point_cloud.ply or gsplat ckpt .pt")
    ap.add_argument("--cameras", default=None, help="optional: COLMAP dir / cameras.json (camera jump + start pose)")
    ap.add_argument("--images", default="images", help="COLMAP image folder (only used to size the cameras)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; localhost by default so a shared cluster node doesn't expose the "
                         "viewer to the network (use SSH port forwarding); 0.0.0.0 to expose it")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default="results/viewer", help="where the save buttons write")
    args = ap.parse_args()

    import nerfview
    import viser

    device = "cuda"
    raw = load_scene(args.scene, device=device)
    act = activate(raw)
    state = ViewerState(raw, act, device)
    n = state.n_total
    print(f"Loaded {n:,} Gaussians (SH degree {act['sh_degree']}); serving on {args.host}:{args.port}")

    cams = names = None
    if args.cameras:
        cams = load_cameras(args.cameras, images=args.images, split="all")
        _, test = split_indices(len(cams), 8)
        names = [f"{nm}{' (test)' if i in set(test) else ''}" for i, nm in enumerate(cams.names)]

    def render_fn(camera_state, render_tab_state):
        if render_tab_state.preview_render:
            W, H = render_tab_state.render_width, render_tab_state.render_height
        else:
            W, H = render_tab_state.viewer_width, render_tab_state.viewer_height
        viewmat = torch.from_numpy(camera_state.c2w).float().inverse().to(device)
        K = torch.from_numpy(camera_state.get_K((W, H))).float().to(device)
        img, info = render_view(state, viewmat, K, W, H)
        if not hasattr(h, "psnr"):                 # GUI still being built
            return img.cpu().numpy()
        h.kept.value = f"{info['n_kept']:,} / {n:,} ({100 * info['n_kept'] / n:.1f}%)"
        h.size.value = f"{info['n_kept'] * bytes_per_gaussian(raw) / 1e6:.0f} MB of {n * bytes_per_gaussian(raw) / 1e6:.0f} MB"
        h.ms.value = f"{info['ms']:.1f} ms ({1e3 / max(info['ms'], 1e-3):.0f} FPS)"
        h.psnr.value = "-" if info["psnr"] is None else f"{info['psnr']:.2f} dB"
        return img.cpu().numpy()

    server = viser.ViserServer(host=args.host, port=args.port)
    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")
    gui, b = server.gui, state.bounds

    class h:                       # GUI handles used by the render thread
        pass

    with gui.add_folder("Pruning thresholds"):
        s_op = gui.add_slider("min opacity (log10)", min=b["min_opacity"][0], max=b["min_opacity"][1],
                              step=0.01, initial_value=b["min_opacity"][0])
        s_sc = gui.add_slider("max scale (log10)", min=b["max_scale"][0], max=b["max_scale"][1],
                              step=0.01, initial_value=b["max_scale"][1])
        s_an = gui.add_slider("max anisotropy (log10)", min=b["max_anisotropy"][0], max=b["max_anisotropy"][1],
                              step=0.01, initial_value=b["max_anisotropy"][1])
        t_vals = gui.add_text("thresholds", "off / off / off", disabled=True)
        btn_3dgs = gui.add_button("preset: opacity 0.005 (3DGS cull)")
        btn_reset = gui.add_button("reset (keep all)")
    with gui.add_folder("View"):
        mode = gui.add_dropdown("mode", VIEW_MODES, initial_value="pruned")
        psnr_box = gui.add_checkbox("PSNR vs unpruned", True)
    with gui.add_folder("Live stats"):
        h.kept = gui.add_text("kept", "", disabled=True)
        h.size = gui.add_text("est. size", "", disabled=True)
        h.ms = gui.add_text("render", "", disabled=True)
        h.psnr = gui.add_text("view PSNR vs unpruned", "", disabled=True)
    with gui.add_folder("Save"):
        btn_json = gui.add_button("save thresholds (json)")
        btn_ply = gui.add_button("save pruned .ply")

    def on_thresholds(_=None):
        state.set_log_thresholds(s_op.value, s_sc.value, s_an.value)
        th = state.thresholds
        fmt = lambda v, off: "off" if v == off else f"{v:.4g}"
        t_vals.value = " / ".join([fmt(th["min_opacity"], 0.0), fmt(th["max_scale"], float("inf")),
                                   fmt(th["max_anisotropy"], float("inf"))])
        viewer.rerender(_)

    for s in (s_op, s_sc, s_an):
        s.on_update(on_thresholds)

    @btn_3dgs.on_click
    def _(_):
        s_op.value = max(b["min_opacity"][0], math.log10(0.005))       # fires on_thresholds via on_update

    @btn_reset.on_click
    def _(_):
        s_op.value, s_sc.value, s_an.value = b["min_opacity"][0], b["max_scale"][1], b["max_anisotropy"][1]

    @mode.on_update
    def _(_):
        state.view_mode = mode.value
        viewer.rerender(_)

    @psnr_box.on_update
    def _(_):
        state.show_psnr = psnr_box.value
        viewer.rerender(_)

    out = Path(args.out)

    @btn_json.on_click
    def _(_):
        out.mkdir(parents=True, exist_ok=True)
        th = {k: (None if math.isinf(v) or v == 0.0 else v) for k, v in state.thresholds.items()}
        kept = int(state.mask.sum())
        rec = dict(scene=args.scene, thresholds=th, n_total=n, n_kept=kept, kept_fraction=kept / n,
                   est_mb=kept * bytes_per_gaussian(raw) / 1e6, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        p = out / "thresholds.json"
        p.write_text(json.dumps(rec, indent=1))
        print("saved", p, rec)

    @btn_ply.on_click
    def _(_):
        out.mkdir(parents=True, exist_ok=True)
        p = out / "pruned.ply"
        save_ply(prune(raw, state.mask), p)
        print(f"saved {p} ({int(state.mask.sum()):,} Gaussians)")

    if cams is not None:
        with gui.add_folder("Dataset cameras"):
            cam_dd = gui.add_dropdown("camera", names, initial_value=names[0])
            btn_go = gui.add_button("go to camera")

        def go(client, i):
            pos, wxyz, fov = camera_to_viser(cams, i)
            client.camera.position, client.camera.wxyz, client.camera.fov = pos, wxyz, fov

        @btn_go.on_click
        def _(_):
            for client in server.get_clients().values():
                go(client, names.index(cam_dd.value))

        # orbit around a sensible 'up': the mean of the cameras' -y axes (OpenCV: y points down)
        up = -torch.linalg.inv(cams.viewmats.double())[:, :3, 1].mean(0).numpy()
        server.scene.set_up_direction(tuple(up / np.linalg.norm(up)))

        @server.on_client_connect
        def _(client):
            go(client, 0)

    on_thresholds()
    print("Viewer running. Ctrl+C to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
