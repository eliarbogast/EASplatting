#!/usr/bin/env bash
# get_data.sh — datasets and pretrained scenes.
#
#   bash scripts/get_data.sh                       # default local set: tandt + truck, mipnerf360 bonsai + its NB checkpoint
#   bash scripts/get_data.sh mipnerf360 bonsai garden      # Mip-NeRF 360 scenes (official downscaled images + COLMAP)
#   bash scripts/get_data.sh mipnerf360 all                # all 9 scenes
#   bash scripts/get_data.sh nb-checkpoints bonsai garden  # NerfBaselines "gaussian-splatting" checkpoints + metrics
#   bash scripts/get_data.sh tandt                         # Inria Tanks&Temples + Deep Blending inputs (~650 MB)
#   bash scripts/get_data.sh inria-pretrained truck garden # Inria's released 30k models (may include test views!)
#
#   FULLRES=1  also fetch full-resolution images/ for mipnerf360 (needed only to TRAIN with gsplat at
#              data_factor > 1: its parser re-downscales from full-res into images_<f>_png)
#
# Layout produced (all git-ignored)
#   data/360_v2/<scene>/{sparse/0, images_4 | images_2 [, images]}
#   data/tandt/{truck,train}/..., data/db/{playroom,drjohnson}/...
#   pretrained/nb/gaussian-splatting/mipnerf360/<scene>/point_cloud.ply   (+ <scene>.json: NB's metrics)
#   pretrained/inria/<scene>/point_cloud.ply
#
# Needs only python (+ coreutils); downloads and unzipping go through scripts/download.py, so there is
# no dependency on wget / curl / unzip. Run it where the internet is (login node on a cluster).
#
# Protocol (NerfBaselines, matches the 3DGS paper's split): test = every 8th image (name-sorted);
# outdoor scenes at images_4, indoor at images_2 — the official downscaled JPEGs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p data pretrained ext

M360_MAIN="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
M360_EXTRA="https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"   # flowers, treehill
NB_RESULTS="https://huggingface.co/nerfbaselines/nerfbaselines/resolve/main"
M360_ALL="bicycle flowers garden stump treehill bonsai counter kitchen room"

res_of() { case "$1" in bonsai|counter|kitchen|room) echo 2 ;; *) echo 4 ;; esac; }
zip_of() { case "$1" in flowers|treehill) echo "$M360_EXTRA" ;; *) echo "$M360_MAIN" ;; esac; }

mipnerf360() {
  local scenes="$*"; [ "$scenes" = "all" ] && scenes="$M360_ALL"
  for s in $scenes; do
    local r; r=$(res_of "$s")
    local prefixes=(--prefix "$s/sparse/" --prefix "$s/images_$r/")
    [ "${FULLRES:-0}" = "1" ] && prefixes+=(--prefix "$s/images/")
    echo "== Mip-NeRF 360 $s (images_$r$([ "${FULLRES:-0}" = 1 ] && echo ' + full-res'))"
    python scripts/fetch_zip_members.py "$(zip_of "$s")" "${prefixes[@]}" --out data/360_v2
    # the extra-scenes zip may nest scenes differently; fail loudly rather than silently
    [ -d "data/360_v2/$s/sparse" ] || { echo "!! data/360_v2/$s/sparse missing — check the zip layout with --list"; exit 1; }
  done
}

nb_checkpoints() {
  local scenes="$*"; [ "$scenes" = "all" ] && scenes="$M360_ALL"
  for s in $scenes; do
    local d="pretrained/nb/gaussian-splatting/mipnerf360/$s"
    mkdir -p "$d"
    echo "== NerfBaselines gaussian-splatting / mipnerf360 / $s"
    python scripts/download.py -q "$NB_RESULTS/gaussian-splatting/mipnerf360/$s.json" "$d/$s.json" || echo "  (no metrics json)"
    if [ ! -f "$d/point_cloud.ply" ]; then
      python scripts/download.py "$NB_RESULTS/gaussian-splatting/mipnerf360/$s.zip" "$d/$s.zip"
      python scripts/download.py -q --unzip "$d/$s.zip" "$d/unzipped"
      # newest iteration's point_cloud.ply, wherever the archive put it
      local ply; ply=$(find "$d/unzipped" -name point_cloud.ply -path "*iteration_*" | sort -V | tail -1)
      [ -n "$ply" ] || ply=$(find "$d/unzipped" -name "*.ply" | head -1)
      [ -n "$ply" ] || { echo "!! no .ply found in $d/$s.zip — inspect $d/unzipped"; exit 1; }
      ln -sf "$(realpath "$ply")" "$d/point_cloud.ply"
      echo "  -> $d/point_cloud.ply  ($(du -h "$ply" | cut -f1))"
    fi
  done
}

tandt() {
  if [ ! -d data/tandt/truck ]; then
    echo "== Tanks&Temples + Deep Blending (Inria preprocessed, undistorted)"
    python scripts/download.py https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip data/tandt_db.zip
    python scripts/download.py -q --unzip --no-overwrite data/tandt_db.zip data
  fi
}

inria_pretrained() {
  # Standalone script from gsplat's main branch: range-extracts single scenes from the 14.7 GB models.zip.
  [ -f ext/download_3dgs_paper_scenes.py ] || python scripts/download.py -q \
    https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/examples/download_3dgs_paper_scenes.py \
    ext/download_3dgs_paper_scenes.py
  python ext/download_3dgs_paper_scenes.py --scenes "$@" --output-dir pretrained/inria
}

cmd=${1:-default}; shift || true
case "$cmd" in
  default)          tandt; inria_pretrained truck; mipnerf360 bonsai; nb_checkpoints bonsai ;;
  mipnerf360)       mipnerf360 "${@:-bonsai}" ;;
  nb-checkpoints)   nb_checkpoints "${@:-bonsai}" ;;
  tandt)            tandt ;;
  inria-pretrained) inria_pretrained "${@:-truck}" ;;
  *) echo "unknown command: $cmd"; sed -n 2,12p "$0"; exit 1 ;;
esac
echo "Done."
