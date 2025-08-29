#!/usr/bin/env python3
"""
Run StableToolBench evaluation using Python with proper environment loading
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

def run_stabletoolbench_eval():
    """Run StableToolBench evaluation with proper environment setup."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key and model from environment
    api_key = os.getenv('OPENAI_API_KEY')
    judge_model = os.getenv('OPENAI_JUDGE_MODEL')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment!")
        return False
    
    print(f"🔑 Using API key: {api_key[:10]}...")
    print(f"🤖 Using judge model: {judge_model}")
    
    # Set up paths (using absolute paths)
    current_dir = Path.cwd()
    stabletoolbench_path = current_dir / "../../stabletoolbench_lib/libs/stabletoolbench"
    converted_answer_path = current_dir / "synthetic_evaluation_results/converted_answers"
    save_path = current_dir / "synthetic_evaluation_results/evaluation"
    test_ids_path = current_dir / "synthetic_dataset/test_query_ids"
    
    # Create API pool file
    api_pool = [
        {
            "api_key": api_key,
            "api_base": None
        }
    ]
    
    api_key_file = stabletoolbench_path / "openai_key.json"
    with open(api_key_file, 'w') as f:
        json.dump(api_pool, f, indent=2)
    
    print(f"✅ Created API pool file: {api_key_file}")
    
    # Change to StableToolBench directory
    original_cwd = os.getcwd()
    os.chdir(stabletoolbench_path)
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['API_POOL_FILE'] = str(api_key_file)
        env['EVAL_MODEL'] = judge_model
        
        print("🚀 Running StableToolBench Pass Rate Evaluation")
        print(f"📁 Converted Answer Path: {converted_answer_path}")
        print(f"📁 Save Path: {save_path}")
        print(f"📁 Test IDs Path: {test_ids_path}")
        
        # Run Pass Rate (SoPR) evaluation
        print("\n📊 Running Pass Rate (SoPR) evaluation...")
        result = subprocess.run([
            sys.executable, "toolbench/tooleval/eval_pass_rate.py",
            "--converted_answer_path", str(converted_answer_path),
            "--save_path", str(save_path),
            "--reference_model", "langgraph_react",
            "--test_ids", str(test_ids_path),
            "--max_eval_threads", "1",
            "--evaluate_times", "1",
            "--test_set", "G1_instruction",
            "--overwrite"
        ], env=env, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Pass Rate evaluation completed successfully!")
            print(result.stdout)
        else:
            print("❌ Pass Rate evaluation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Run FAC (Final Answer Correctness) evaluation using OpenAI
        print("\n📊 Running FAC evaluation using OpenAI...")
        result = subprocess.run([
            sys.executable, str(current_dir / "custom_fac_eval.py"),
            "--converted_answer_path", str(converted_answer_path),
            "--save_path", str(save_path),
            "--test_ids", str(test_ids_path),
            "--test_set", "G1_instruction"
        ], env=env, capture_output=True, text=True, timeout=600)  # 10 minutes timeout for FAC
        
        if result.returncode == 0:
            print("✅ FAC evaluation completed successfully!")
            print(result.stdout)
        else:
            print("❌ FAC evaluation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            # Don't return False here, continue with SoWR
        
        # Note: SoWR evaluation requires two models to compare
        # Since we only have one model (langgraph_react), we skip SoWR
        print("\n📊 SoWR evaluation skipped - requires two models to compare")
        print("   To run SoWR, you need results from two different models")
        
        return True
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = run_stabletoolbench_eval()
    if not success:
        sys.exit(1)
