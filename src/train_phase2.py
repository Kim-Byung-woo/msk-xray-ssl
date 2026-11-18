#!/usr/bin/env python3
"""
MSK X-ray SSL Phase 2 — CLI 학습 실행기
(D3-E0 과 같은 조건: --batch-size 256 --max-train-images 0 을 항상 포함)

  ./venv/bin/python3 -u train_phase2.py --experiment D3-E0 --smoke-test-only
  ./venv/bin/python3 -u train_phase2.py --experiment D3-E0 --batch-size 256 --max-train-images 0
  ./venv/bin/python3 -u train_phase2.py --experiment D3-E1 --run-tag e1c        --pos-n-cross 64 --batch-size 256 --max-train-images 0
  ./venv/bin/python3 -u train_phase2.py --experiment D3-E1 --run-tag e1c_within --pos-n-cross 0  --batch-size 256 --max-train-images 0
  ./venv/bin/python3 -u train_phase2.py --experiment D3-E2 --partwhole-method adam --run-tag adam --smoke-test-only
  ./venv/bin/python3 -u train_phase2.py --experiment D3-E2 --partwhole-method adam --run-tag adam \\
      --batch-size 256 --max-train-images 0 --monitor epoch_mean_base_loss
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser(
        description="MSK X-ray DINOv3 SSL Phase 2 — D3-E0 / D3-E1 / D3-E2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--experiment", required=True, choices=["D3-E0", "D3-E1", "D3-E2"])
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--cache-list", default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--annotation-csv", default=None)
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pos-n-cross", type=int, default=None,
                    help="position InfoNCE cross-image negatives (D3-E1c=64, D3-E1c-within=0)")
    ap.add_argument("--pos-chunk", type=int, default=None,
                    help="position_pair_chunk_size (메모리용, 수식 무관)")
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--monitor", default=None, choices=["epoch_mean_total_loss", "epoch_mean_base_loss"],
                    help="best.pt 기준. D3-E2 는 epoch_mean_base_loss 권장 (D3-E0 과 같은 기준)")
    # ---- [D3-E2] Part–Whole ----
    ap.add_argument("--partwhole-method", default=None, choices=["none", "adam"])
    ap.add_argument("--partwhole-start-epoch", type=int, default=None)
    ap.add_argument("--partwhole-part-size", type=int, default=None, help="128=native(B안), 256=2배 확대(E안)")
    ap.add_argument("--partwhole-n-crops", type=int, default=None)
    ap.add_argument("--partwhole-batch-frac", type=float, default=None)
    ap.add_argument("--partwhole-embed", default=None, choices=["patch_mean", "cls"])
    ap.add_argument("--partwhole-student-whole", default=None, choices=["unmasked", "clean"])
    ap.add_argument("--partwhole-loss", default=None, choices=["nmse", "cos"])
    ap.add_argument("--lambda-comp", type=float, default=None)
    ap.add_argument("--lambda-decomp", type=float, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--smoke-test-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import msk_ssl_phase2 as lib

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
    if args.lr: overrides["lr"] = args.lr
    if args.seed is not None: overrides["seed"] = args.seed
    if args.no_resume: overrides["auto_resume"] = False
    if args.pos_n_cross is not None: overrides["position_n_cross_negatives"] = args.pos_n_cross
    if args.pos_chunk is not None: overrides["position_pair_chunk_size"] = args.pos_chunk
    for key in ("warmup_epochs", "monitor", "partwhole_method", "partwhole_start_epoch",
                "partwhole_part_size", "partwhole_n_crops", "partwhole_batch_frac",
                "partwhole_embed", "partwhole_student_whole", "partwhole_loss",
                "lambda_comp", "lambda_decomp"):
        v = getattr(args, key)
        if v is not None:
            overrides[key] = v

    cfg = lib.Config(**overrides)

    print("=" * 74)
    print(f"EXPERIMENT {cfg.experiment}  |  {lib.MODEL_DESCRIPTION.get(cfg.experiment, '')}")
    print("=" * 74)
    print(f"  data_root        : {cfg.data_root}")
    print(f"  run_dir          : {cfg.run_dir}")
    print(f"  batch_size       : {cfg.batch_size}")
    print(f"  epochs           : {cfg.epochs}  (steps_per_epoch={cfg.steps_per_epoch})")
    print(f"  max_train_images : {cfg.max_train_images}")
    print(f"  centering        : {cfg.centering}")
    print(f"  layerwise_decay  : {cfg.layerwise_decay}")
    print(f"  patch_embed_lr   : {cfg.patch_embed_lr_mult}")
    print(f"  teacher_momentum : {cfg.teacher_momentum} -> {cfg.teacher_momentum_final}")
    print(f"  monitor          : {cfg.monitor}")
    if cfg.use_partwhole:
        print(f"  part-whole       : {cfg.partwhole_method}  start_epoch={cfg.partwhole_start_epoch}"
              f"  lambda_comp={cfg.lambda_comp}  lambda_decomp={cfg.lambda_decomp}")
        print(f"  pw parts         : size={cfg.partwhole_part_size}  n_crops={cfg.partwhole_n_crops}"
              f"  batch_frac={cfg.partwhole_batch_frac}")
        print(f"  pw embed/loss    : {cfg.partwhole_embed} / {cfg.partwhole_loss}"
              f"  student_whole={cfg.partwhole_student_whole}")
    if cfg.use_position:
        print(f"  position         : {cfg.position_loss_type}  lambda={cfg.lambda_pos}  K_pos={cfg.K_pos}")
        print(f"  pos InfoNCE      : temp={cfg.position_infonce_temp}  within={cfg.position_n_within_negatives}"
              f"  cross={cfg.position_n_cross_negatives}  neg_ios<={cfg.position_negative_ios_threshold}"
              f"  pos_ios>={cfg.pos_overlap_threshold}  topk={cfg.position_topk}")
        print(f"  pos chunk        : {cfg.position_pair_chunk_size}")
    print("=" * 74, flush=True)

    if args.smoke_test or args.smoke_test_only:
        lib.run_smoke_test(cfg, experiment=cfg.experiment)
        if args.smoke_test_only:
            return

    file_paths = lib.get_image_paths(cfg, rebuild_cache=args.rebuild_cache)
    trainer = lib.Trainer(cfg, file_paths=file_paths)
    trainer.fit()


if __name__ == "__main__":
    main()
