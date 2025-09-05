#!/usr/bin/env python3
"""
StableToolBench FAC-Only Evaluation Runner
Uses original StableToolBench code but only runs FAC evaluation.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_stabletoolbench_fac_eval():
    """Run StableToolBench FAC evaluation using original code."""
    
    # Paths - go up to project root
    project_root = Path(__file__).parent.parent.parent
    stabletoolbench_dir = project_root / "stabletoolbench_lib/libs/stabletoolbench"
    converted_answers_path = project_root / "fac_evaluation_results/converted_answers/langgraph_react/G1_instruction.json"
    test_ids_path = project_root / "src/max_tool_experiment/synthetic_dataset/test_query_ids/G1_instruction.json"
    output_path = project_root / "fac_evaluation_results/evaluation/fac_results.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Running StableToolBench FAC evaluation...")
    print(f"Converted answers: {converted_answers_path}")
    print(f"Test IDs: {test_ids_path}")
    print(f"Output: {output_path}")
    
    # Check if files exist
    if not converted_answers_path.exists():
        print(f"❌ Converted answers file not found: {converted_answers_path}")
        return False
    
    if not test_ids_path.exists():
        print(f"❌ Test IDs file not found: {test_ids_path}")
        return False
    
    # Run modified StableToolBench FAC evaluation with OpenShift
    try:
        # Get OpenShift URL from environment
        openshift_url = os.getenv('OPENSHIFT_EVALUATOR_URL')
        
        if openshift_url:
            print(f"🔍 Using OpenShift hosted model: {openshift_url}")
            cmd = [
                sys.executable,
                str(stabletoolbench_dir / "toolbench/tooleval/fac_eval.py"),
                "--openshift_url", openshift_url,
                "--evaluation_path", str(converted_answers_path),
                "--output_path", str(output_path),
                "--ids", str(test_ids_path)
            ]
        else:
            print("⚠️ No OpenShift URL provided, falling back to local vLLM (if available)")
            cmd = [
                sys.executable,
                str(stabletoolbench_dir / "toolbench/tooleval/fac_eval.py"),
                "--evaluation_path", str(converted_answers_path),
                "--output_path", str(output_path),
                "--ids", str(test_ids_path)
            ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        # Set environment variables for StableToolBench
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{stabletoolbench_dir}:{env.get('PYTHONPATH', '')}"
        
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(project_root),  # Run from project root
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ FAC evaluation completed successfully!")
        print(f"📊 Results saved to: {output_path}")
        
        # Parse and display results
        if output_path.exists():
            parse_fac_results(output_path)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FAC evaluation failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Error running FAC evaluation: {e}")
        return False

def parse_fac_results(results_file: Path):
    """Parse and display FAC evaluation results."""
    try:
        import pandas as pd
        
        df = pd.read_csv(results_file)
        
        # Count solved vs unsolved
        total_queries = len(df)
        solved_count = sum(1 for eval_text in df['evaluation'] if 'solved' in eval_text.lower())
        solve_rate = (solved_count / total_queries) * 100 if total_queries > 0 else 0
        
        print(f"\n📊 FAC Evaluation Results:")
        print(f"   Total Queries: {total_queries}")
        print(f"   Solved: {solved_count}")
        print(f"   Unsolved: {total_queries - solved_count}")
        print(f"   Solve Rate: {solve_rate:.1f}%")
        
        # Show sample evaluations
        print(f"\n📝 Sample Evaluations:")
        for i, (query, answer, evaluation) in enumerate(zip(df['query'][:3], df['final_answer'][:3], df['evaluation'][:3])):
            print(f"\n   Query {i+1}: {query[:60]}...")
            print(f"   Answer: {answer[:60]}...")
            print(f"   Evaluation: {evaluation[:100]}...")
        
    except ImportError:
        print("⚠️ pandas not available, cannot parse results")
    except Exception as e:
        print(f"⚠️ Could not parse results: {e}")

def main():
    """Main function."""
    print("🚀 Starting StableToolBench FAC-Only Evaluation...")
    print("=" * 60)
    
    success = run_stabletoolbench_fac_eval()
    
    if success:
        print("\n✅ FAC evaluation completed successfully!")
    else:
        print("\n❌ FAC evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
