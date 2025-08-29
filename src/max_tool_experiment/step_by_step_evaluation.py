#!/usr/bin/env python3
"""
Step-by-Step Evaluation Pipeline
A streamlined approach that eliminates overlap between synthetic_evaluation.py and run_local_evaluation.py

Steps:
1. Check manual dataset exists
2. Run agent on queries (if needed)
3. Run StableToolBench evaluation metrics
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add StableToolBench to path
stabletoolbench_path = Path("../../stabletoolbench_lib/libs/stabletoolbench")
sys.path.append(str(stabletoolbench_path))

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools import tool

class StepByStepEvaluation:
    """Streamlined evaluation pipeline without overlap."""
    
    def __init__(self):
        self.output_dir = Path("synthetic_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup directories
        self.raw_dir = self.output_dir / "raw_answers" / "langgraph_react"
        self.converted_dir = self.output_dir / "converted_answers" / "langgraph_react"
        self.eval_dir = self.output_dir / "evaluation"
        
        for dir_path in [self.raw_dir, self.converted_dir, self.eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def step_1_check_dataset(self) -> bool:
        """Step 1: Check if manual dataset exists."""
        print("📊 Step 1: Checking manual dataset...")
        
        dataset_file = Path("synthetic_dataset/test_instruction/G1_instruction.json")
        
        if dataset_file.exists():
            print("✅ Found manual dataset")
            return True
        
        print("❌ Manual dataset not found!")
        print("   Please ensure synthetic_dataset/test_instruction/G1_instruction.json exists")
        return False
    
    def step_2_run_agent(self, force_rerun: bool = False) -> bool:
        """Step 2: Run agent on queries if needed."""
        print("\n🤖 Step 2: Checking/Running agent on queries...")
        
        converted_file = self.converted_dir / "synthetic_results.json"
        
        # Check if we already have agent results
        if not force_rerun and converted_file.exists():
            print("✅ Found existing agent results, skipping agent execution")
            return True
        
        # Run agent if needed
        print("🔄 Running agent on synthetic queries...")
        try:
            result = subprocess.run([
                sys.executable, "synthetic_evaluation.py"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Agent execution completed successfully")
                return True
            else:
                print(f"❌ Agent execution failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Agent execution error: {e}")
            return False
    
    def step_3_run_evaluation(self) -> bool:
        """Step 3: Run StableToolBench evaluation metrics."""
        print("\n📈 Step 3: Running StableToolBench evaluation metrics...")
        
        # Check if we have results to evaluate (look in both possible locations)
        converted_file = self.converted_dir / "langgraph_react" / "G1_instruction.json"
        
        if not converted_file.exists():
            print("❌ No agent results found. Run steps 1-2 first.")
            return False
        
        # Run evaluation using the Python-based runner
        try:
            result = subprocess.run([
                sys.executable, "run_stabletoolbench_eval.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Evaluation completed successfully")
                return True
            else:
                print(f"❌ Evaluation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return False
    
    def run_complete_pipeline(self, force_rerun: bool = False):
        """Run the complete step-by-step pipeline."""
        print("🚀 Starting Step-by-Step Evaluation Pipeline")
        print("=" * 60)
        
        # Step 1: Check dataset
        if not self.step_1_check_dataset():
            print("❌ Pipeline failed at Step 1")
            return False
        
        # Step 2: Run agent
        if not self.step_2_run_agent(force_rerun):
            print("❌ Pipeline failed at Step 2")
            return False
        
        # Step 3: Run evaluation
        if not self.step_3_run_evaluation():
            print("❌ Pipeline failed at Step 3")
            return False
        
        print("\n🎉 Pipeline completed successfully!")
        self.print_results_summary()
        return True
    
    def run_specific_step(self, step: int, **kwargs):
        """Run a specific step of the pipeline."""
        print(f"🎯 Running Step {step} only...")
        
        if step == 1:
            return self.step_1_check_dataset()
        elif step == 2:
            return self.step_2_run_agent(**kwargs)
        elif step == 3:
            return self.step_3_run_evaluation()
        else:
            print(f"❌ Invalid step: {step}. Use 1, 2, or 3.")
            return False
    
    def print_results_summary(self):
        """Print summary of results."""
        print("\n📊 RESULTS SUMMARY")
        print("=" * 40)
        
        # Check for evaluation results
        eval_files = list(self.eval_dir.glob("*.json")) + list(self.eval_dir.glob("*.csv"))
        
        if eval_files:
            print("✅ Evaluation results found:")
            for file in eval_files:
                print(f"   - {file.name}")
            
            # Try to read and display metrics
            self.display_metrics()
        else:
            print("⚠️ No evaluation results found")
    
    def display_metrics(self):
        """Display evaluation metrics if available."""
        try:
            # Check for Pass Rate results
            pass_rate_file = self.eval_dir / "G1_instruction_langgraph_react.json"
            if pass_rate_file.exists():
                with open(pass_rate_file, 'r') as f:
                    data = json.load(f)
                
                # Calculate solve rate
                solved_count = 0
                total_count = 0
                for query_id, result in data.items():
                    total_count += 1
                    if 'is_solved' in result:
                        for eval_time, status in result['is_solved'].items():
                            if 'Solved' in str(status):
                                solved_count += 1
                                break
                
                if total_count > 0:
                    solve_rate = (solved_count / total_count) * 100
                    print(f"📈 Pass Rate (SoPR): {solve_rate:.1f}% ({solved_count}/{total_count})")
            
            # Check for FAC results
            fac_file = self.eval_dir / "fac_evaluation_results.csv"
            if fac_file.exists():
                print("📊 FAC evaluation results available")
                
        except Exception as e:
            print(f"⚠️ Could not display metrics: {e}")

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step-by-Step Evaluation Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], 
                       help="Run specific step only (1=dataset, 2=agent, 3=evaluation)")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force rerun agent execution")
    
    args = parser.parse_args()
    
    evaluator = StepByStepEvaluation()
    
    if args.step:
        # Run specific step
        success = evaluator.run_specific_step(args.step, 
                                            force_rerun=args.force_rerun)
    else:
        # Run complete pipeline
        success = evaluator.run_complete_pipeline(
            force_rerun=args.force_rerun
        )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
