#!/usr/bin/env python3
"""
MSK X-ray SSL Phase 1 — CLI 학습 실행기

사용 예
-------
  ./venv/bin/python3 -u train.py --experiment E0
  ./venv/bin/python3 -u train.py --experiment E1 --batch-size 64 --steps-per-epoch 2000
  ./venv/bin/python3 -u train.py --experiment E0 --smoke-test-only
  ./venv/bin/python3 -u train.py --experiment E2 --run-tag indep_part --part-source local_crop
  ./venv/bin/python3 -u train.py --experiment E2 --run-tag indep_part_gram --part-source local_crop \
      --local-preservation gram --gram-teacher-ckpt outputs/msk_ssl/E0/best.pt

한 번에 하나의 experiment 만 실행한다 (E0~E3 자동 연속 실행 없음).
-u 옵션(unbuffered)을 반드시 붙여야 tee/tail 로 실시간 로그가 보인다.
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser(
        description="MSK X-ray DINOv2-style SSL (E0~E3) — 단일 experiment 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--experiment", required=True, choices=["E0", "E1", "E2", "E3"])
    ap.add_argument("--gpu", default="7", help="physical GPU id (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--run-tag", default="", help="출력 디렉토리 suffix (예: pilot)")

    # 경로
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--cache-list", default=None, help="PNG 경로 캐시 파일")
    ap.add_argument("--rebuild-cache", action="store_true", help="캐시 무시하고 재스캔")
    ap.add_argument("--annotation-csv", default=None)

    # 데이터 / 학습 규모
    ap.add_argument("--max-train-images", type=int, default=None,
                    help="0 또는 음수면 전체 사용")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="0 이면 실제 1 epoch = 전체 dataset")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--pre-resize", type=int, default=None, help="0 이면 사전 축소 비활성화")

    # 최적화
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-foreground-crop", action="store_true",
                    help="foreground-aware crop 비활성화 (명세 그대로의 순수 RRC)")

    # part-whole (E2 / E2b / E2c)
    ap.add_argument("--part-source", default=None, choices=["global_grid", "local_crop"],
                    help="E2 계열 Part 정의. global_grid=E2/E2b, local_crop=E2c")
    ap.add_argument("--local-part-feature", default=None, choices=["patch_mean", "cls"],
                    help="E2c local Part pooling (기본 patch_mean)")
    ap.add_argument("--containment-threshold", type=float, default=None,
                    help="E2c valid Part-Whole relation 임계값 (기본 0.8)")
    ap.add_argument("--no-partwhole-centering", action="store_true",
                    help="teacher part-whole target centering 비활성화 (E2 원본 재현용)")

    # local preservation (E2d)
    ap.add_argument("--local-preservation", default=None, choices=["none", "gram"],
                    help="gram = frozen Gram teacher 에 patch Gram 구조 고정 (E2d)")
    ap.add_argument("--gram-teacher-ckpt", default=None,
                    help="Gram teacher checkpoint 경로 (예: outputs/msk_ssl/E0/best.pt)")
    ap.add_argument("--lambda-gram", type=float, default=None)

    # 실행 제어
    ap.add_argument("--smoke-test", action="store_true", help="학습 전에 smoke test 수행")
    ap.add_argument("--smoke-test-only", action="store_true", help="smoke test 만 하고 종료")
    ap.add_argument("--no-resume", action="store_true", help="latest.pt 가 있어도 처음부터 학습")
    return ap.parse_args()


def main():
    args = parse_args()

    # torch import 이전에 GPU 를 고정해야 한다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import msk_ssl_phase1 as lib   # noqa: E402  (env 설정 이후 import)

    # 서버의 msk_ssl_phase1.py 가 요청한 기능을 지원하는지 확인 (구버전 파일 import 방지)
    if args.part_source == "local_crop" and not hasattr(lib, "LocalPartComposer"):
        raise RuntimeError("msk_ssl_phase1.py 가 E2c(local_crop) 를 지원하지 않습니다. 최신 파일로 교체하세요.")
    if args.local_preservation == "gram" and not hasattr(lib, "gram_anchor_loss"):
        raise RuntimeError("msk_ssl_phase1.py 가 E2d(gram) 를 지원하지 않습니다. 최신 파일로 교체하세요.")

    overrides = {"experiment": args.experiment, "run_tag": args.run_tag}
    if args.data_root: overrides["data_root"] = args.data_root
    if args.output_root: overrides["output_root"] = args.output_root
    if args.cache_list: overrides["dataset_cache_list"] = args.cache_list
    if args.annotation_csv: overrides["anatomical_annotation_csv"] = args.annotation_csv
    if args.max_train_images is not None:
        overrides["max_train_images"] = args.max_train_images if args.max_train_images > 0 else None
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.epochs: overrides["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        overrides["steps_per_epoch"] = args.steps_per_epoch if args.steps_per_epoch > 0 else None
    if args.num_workers is not None: overrides["num_workers"] = args.num_workers
    if args.pre_resize is not None:
        overrides["pre_resize"] = args.pre_resize if args.pre_resize > 0 else None
    if args.lr: overrides["lr"] = args.lr
    if args.seed is not None: overrides["seed"] = args.seed
    if args.no_foreground_crop: overrides["foreground_aware_crop"] = False
    if args.part_source: overrides["partwhole_part_source"] = args.part_source
    if args.local_part_feature: overrides["partwhole_local_part_feature"] = args.local_part_feature
    if args.containment_threshold is not None:
        overrides["partwhole_containment_threshold"] = args.containment_threshold
    if args.no_partwhole_centering: overrides["partwhole_center_whole"] = False
    if args.local_preservation: overrides["local_preservation"] = args.local_preservation
    if args.gram_teacher_ckpt:
        overrides["gram_teacher_ckpt"] = os.path.abspath(args.gram_teacher_ckpt)
    if args.lambda_gram is not None: overrides["lambda_gram"] = args.lambda_gram
    if args.no_resume: overrides["auto_resume"] = False

    cfg = lib.Config(**overrides)

    print("=" * 74)
    print(f"EXPERIMENT {cfg.experiment}  |  {lib.MODEL_DESCRIPTION.get(cfg.experiment, '')}")
    print("=" * 74)
    print(f"  data_root        : {cfg.data_root}")
    print(f"  run_dir          : {cfg.run_dir}")
    print(f"  batch_size       : {cfg.batch_size}")
    print(f"  epochs           : {cfg.epochs}  (steps_per_epoch={cfg.steps_per_epoch})")
    print(f"  max_train_images : {cfg.max_train_images}")
    print(f"  num_workers      : {cfg.num_workers}   pre_resize={cfg.pre_resize}")
    print(f"  foreground crop  : {cfg.foreground_aware_crop}")
    print(f"  aux losses       : position={cfg.use_position}  part-whole={cfg.use_partwhole}")
    if cfg.use_partwhole:
        print(f"  part source      : {cfg.partwhole_part_source}"
              f"  (local feature={cfg.partwhole_local_part_feature},"
              f" containment>={cfg.partwhole_containment_threshold},"
              f" centering={cfg.partwhole_center_whole})")
    print(f"  local preserve   : {cfg.local_preservation}"
          + (f"  (lambda={cfg.lambda_gram}, teacher={cfg.gram_teacher_ckpt})"
             if cfg.local_preservation != "none" else ""))
    print("=" * 74, flush=True)

    if args.smoke_test or args.smoke_test_only:
        lib.run_smoke_test(cfg, experiment=cfg.experiment)
        if args.smoke_test_only:
            return

    # 캐시 재생성이 필요하면 학습 시작 전에 한 번만 수행
    file_paths = lib.get_image_paths(cfg, rebuild_cache=args.rebuild_cache)

    trainer = lib.Trainer(cfg, file_paths=file_paths)
    trainer.fit()


if __name__ == "__main__":
    main()
