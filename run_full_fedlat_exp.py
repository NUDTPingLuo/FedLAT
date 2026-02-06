import subprocess
import os
import sys
import time

# ==========================================
# Configuration
# ==========================================
PYTHON_EXEC = "python"  # Or specific path like "/root/anaconda3/envs/ntk/bin/python"
SCRIPT_NAME = "run_fed_lat.py"

# Fixed Parameters
DATASET = "cifar100"
MODEL = "cifar_cnn"
SEED = "0"

# Loop Variables
IID_SETTINGS = [True, False]
ATTACK_METHODS = ["pgd", "mim", "fgsm"]


# ==========================================
# Helper Function
# ==========================================
def run_command(cmd, log_message):
    print(f"\n{'=' * 60}")
    print(f"Running: {log_message}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    try:
        # Run command and wait for it to finish
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f">>> Success! Time taken: {elapsed:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f">>> Error running command. Exit code: {e.returncode}")
        sys.exit(1)


# ==========================================
# Main Execution Flow
# ==========================================
def main():
    # --- Stage 1: Standard Benign Training (Save Checkpoints) ---
    print(">>> STARTING STAGE 1: Standard Benign Training (100 Epochs)")

    for is_iid in IID_SETTINGS:
        for attack in ATTACK_METHODS:
            dist_tag = "IID" if is_iid else "Non_IID"
            save_path = f"sgd_benign100"

            cmd = [
                PYTHON_EXEC, SCRIPT_NAME,
                "--standard_epochs", "101",
                "--linear_epochs", "0",
                "--loaders", "CC",
                "--attack_method", attack,
                "--dataset", DATASET,
                "--model", MODEL,
                "--save_path", save_path,
                "--constant_save",
                "--random_seed", SEED,
                "--skip_second_test",
                "--is_iid", str(is_iid)
            ]

            run_command(cmd, f"Stage 1 | {dist_tag} | {attack}")

    # --- Stage 2: Linearized Adversarial Training (Load Checkpoint -> Linear Adv) ---
    print("\n\n>>> STARTING STAGE 2: Linearized Adversarial Training (Load Ep50 -> Train 50)")

    for is_iid in IID_SETTINGS:
        for attack in ATTACK_METHODS:
            dist_tag = "IID" if is_iid else "Non_IID"

            # Construct paths dynamically based on Stage 1
            # Note: run_exp.py prepends "attack_distribute_" to save_path automatically.
            # We must match the folder structure created in Stage 1.
            # Based on your code: args.save_path = args.attack_method + '_' + distribute + '_' + args.save_path

            base_folder_name = f"{dist_tag}_sgd_benign100"
            checkpoint_path = f"{base_folder_name}/phase1_epoch_50.pkl"

            # Output path for Stage 2
            save_path_stage2 = f"sgd_benign50_to_linear_adv50"

            cmd = [
                PYTHON_EXEC, SCRIPT_NAME,
                "--base_model_path", checkpoint_path,
                "--loaders", "CA",
                "--attack_method", attack,
                "--dataset", DATASET,
                "--model", MODEL,
                "--standard_epochs", "0",
                "--linear_epochs", "51",
                "--save_path", save_path_stage2,
                "--random_seed", SEED,
                "--skip_first_test",
                "--constant_save_linear",
                "--is_iid", str(is_iid)
            ]

            run_command(cmd, f"Stage 2 | {dist_tag} | {attack} | Load: {checkpoint_path}")

    print("\n>>> All experiments completed successfully!")


if __name__ == "__main__":
    main()