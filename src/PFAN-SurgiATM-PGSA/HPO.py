import optuna
import os
import subprocess
import re
import json
import sys

BASE_PATH = "/workspace/PFAN-SurgiATM"
CHECKPOINT_DIR = os.path.join(BASE_PATH, "Checkpoint")
DATA_ROOT = os.path.join(BASE_PATH, "HPO_dataset/combined")

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    beta1 = trial.suggest_float("beta1", 0.5, 0.9)
    surgiatm_wz = trial.suggest_categorical('surgiatm_wz', [5, 11, 15, 21])
    lambda_smooth = trial.suggest_float("lambda_smooth", 0.01, 0.5, log=True)
    smooth_alpha = trial.suggest_float("smooth_alpha", 5.0, 50.0, log=True)

    exp_name = f"pfan_hpo_{trial.number}"

    cmd = (
        f"python train.py "
        f"--dataroot {DATA_ROOT} "
        f"--model pix2pix "
        f"--netG pfan "
        f"--netD basic "
        f"--direction AtoB "
        f"--checkpoints_dir {CHECKPOINT_DIR} "
        f"--name {exp_name} "
        f"--lr {lr} "
        f"--batch_size 16 "
        f"--beta1 {beta1} "
        f"--n_epochs 20 "
        f"--dataset_mode aligned "
        f"--gpu_ids 0 "
        
        f"--surgiatm_wz {surgiatm_wz} "
        f"--lambda_smooth {lambda_smooth} " 
        f"--smooth_alpha {smooth_alpha} "
    )

    final_score = 0.0
    try:
        # Popen을 통한 실시간 로그 스트리밍 및 Pruning 개입
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            # 1. 가로챈 로그를 다시 터미널(nohup 로그)로 출력
            print(line, end="")
            sys.stdout.flush()  # 버퍼링을 방지하여 tail -f에서 즉각 보이도록 강제 출력# 정규표현식을 통해 실시간으로 출력되는 검증 에포크와 지표를 포착
            
            match = re.search(r"\(VAL\s+epoch:\s*(\d+).*?SSIM:\s*([0-9.]+).*?PSNR:\s*([0-9.]+)", line)
            if match:
                epoch = int(match.group(1))
                ssim = float(match.group(2))
                psnr = float(match.group(3))
               
                # 가중치 기반 통합 점수 계산
                current_score = (ssim * 0.5) + ((psnr / 40.0) * 0.5)
                final_score = current_score

                # Optuna에 현재 에포크 성능 보고
                trial.report(current_score, epoch)

                # 하위 50% 성능 도달 시 프로세스 즉각 사살 (MedianPruner 작동)
                if trial.should_prune():
                    process.kill()
                    process.wait()
                    raise optuna.exceptions.TrialPruned()
                
        process.wait()

        # 정상 종료가 아니며(0) Pruning에 의한 강제 종료(-9)도 아닌 경우의 에러 처리
        if process.returncode != 0 and process.returncode != -9:
            print(f"--- Trial {trial.number} Failed with return code {process.returncode} ---")
            return 0.0
        
    except optuna.exceptions.TrialPruned:
        raise # Pruning 신호는 Optuna 엔진으로 패스

    except Exception as e:
        print(f"--- Trial {trial.number} Execution Error: {e} ---")
        return 0.0
    
    return final_score

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0)
    )

    study.optimize(objective, n_trials=50)

    results = {
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "best_score": study.best_value
    }

    result_file = os.path.join(BASE_PATH, "hpo_best_results.json")
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[HPO Completed] Best results saved to {result_file}")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")