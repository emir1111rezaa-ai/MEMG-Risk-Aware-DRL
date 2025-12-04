"""Quick start script to run full pipeline."""
import os
import sys
import time
from datetime import datetime

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def main():
    """Run complete MEMG pipeline."""
    start_time = time.time()
    
    print_header("MULTI-ENERGY MICROGRID - COMPLETE PIPELINE")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Step 1: Training
    print_header("STEP 1: TRAINING PHASE")
    print("This will take approximately 2-3 hours with GPU, 6-8 hours with CPU.\n")
    
    try:
        import train
        train.main()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)
    
    # Step 2: Evaluation
    print_header("STEP 2: EVALUATION PHASE")
    print("This will take approximately 30-60 minutes.\n")
    
    try:
        import evaluate
        evaluate.main()
        print("\n✓ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        sys.exit(1)
    
    # Summary
    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total Execution Time: {hours}h {minutes}m {seconds}s")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n✅ Generated Files:")
    files = [
        'forecaster.pkl',
        'ppo_agent.pth',
        'simple_drl_agent.pth',
        'method_comparison.csv',
        'cost_comparison.png',
        'cvar_comparison.png',
        'violations_comparison.png',
        'battery_operation.png',
        'timeseries_detailed.png',
        'training_curves_ppo.png'
    ]
    
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024  # KB
            print(f"  ✓ {f:<30} ({size:.1f} KB)")
        else:
            print(f"  ✗ {f:<30} (NOT FOUND)")
    
    print("\n🎉 All done! Check the generated plots and CSV for results.")
    print("\nNext steps:")
    print("  1. Review method_comparison.csv for quantitative results")
    print("  2. Examine plots for visual analysis")
    print("  3. Adjust config.py and re-run for different scenarios")
    print("  4. Use the trained models for further experiments\n")

if __name__ == '__main__':
    main()
