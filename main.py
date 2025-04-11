import asyncio
import json
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.action import ActionManager
from src.config import ConfigManager
from src.interpreter import QueryInterpreterModule
from src.model import GoogleLLMModel


class IntentEvaluator:
    """
    Evaluates the accuracy of intent recognition by generating test prompts
    and analyzing whether the QueryInterpreterModule correctly identifies the intended action.
    """

    def __init__(self, llm_model, action_manager, query_interpreter):
        self.llm_model = llm_model
        self.action_manager = action_manager
        self.query_interpreter = query_interpreter
        self.results = []
        self.used_prompts = set()

        # Define difficulty classes
        self.difficulty_classes = [
            {
                "name": "Simple",
                "description": "Direct requests with explicit action and parameters",
                "system_prompt": "Create a clear, straightforward user request that directly asks for the specified action. Use explicit parameters and be very specific in the request. The request should be simple and directly map to the action."
            },
            {
                "name": "Moderate",
                "description": "Indirect requests with implicit parameters",
                "system_prompt": "Create a moderately complex user request that indirectly asks for the specified action. Some parameters should be implicit or need to be inferred. Use everyday language rather than system terminology."
            },
            {
                "name": "Complex",
                "description": "Ambiguous requests with contextual dependencies",
                "system_prompt": "Create a complex user request that is somewhat ambiguous about the specified action. Make the request conversational, with some distracting information. Parameters should be mentioned indirectly or require contextual understanding. The user might not know exactly what they want or use non-standard terminology."
            }
        ]

    async def generate_prompt(self, action_name, difficulty_class):
        """
        Generates a user prompt for a specific action with the given difficulty level.
        """
        action = self.action_manager.get_action(action_name)
        if not action:
            raise ValueError(f"Unknown action: {action_name}")

        difficulty = self.difficulty_classes[difficulty_class]

        # Build a prompt for the LLM to generate a user query
        system_prompt = f"""You are helping create test data for evaluating an AI assistant. 
            {difficulty["description"]}

            The action to generate a prompt for is: "{action.name}" - {action.description}

            Required parameters: {[field for field, info in action.params_model.model_fields.items() if info.is_required()]}
            Optional parameters: {[field for field, info in action.params_model.model_fields.items() if not info.is_required()]}

            {difficulty["system_prompt"]}

            IMPORTANT: 
            1. Generate ONLY the user's request text, nothing else. 
            2. Do not include any explanations or metadata.
            3. Make sure each request is unique and varied from previous ones.
            4. Use realistic values for parameters (dates, times, event names, etc.)
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a user request for the action '{action_name}'"}
        ]

        # Generate the prompt
        response = await asyncio.to_thread(self.llm_model.process, messages)
        if not response.text:
            raise Exception("Failed to generate prompt")

        generated_prompt = response.text.strip()

        # Check if this prompt was used before (simple check to ensure variance)
        if generated_prompt in self.used_prompts:
            # Try once more with an explicit request for uniqueness
            messages.append({"role": "assistant", "content": generated_prompt})
            messages.append({"role": "user", "content": "Please create a different unique variation of this request"})
            response = await asyncio.to_thread(self.llm_model.process, messages)
            generated_prompt = response.text.strip()

        self.used_prompts.add(generated_prompt)
        return generated_prompt

    async def evaluate_action(self, target_action_name):
        """
        Evaluates how well the system identifies a specific action across different difficulty levels.
        """
        for difficulty_idx, difficulty in enumerate(self.difficulty_classes):
            try:
                # Generate a prompt for this action at this difficulty level
                prompt = await self.generate_prompt(target_action_name, difficulty_idx)

                # Try to interpret the prompt using the query interpreter
                messages = [{"role": "user", "content": prompt}]
                interpretation = await self.query_interpreter.interpret(messages)

                # Record the result
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "target_action": target_action_name,
                    "difficulty_class": difficulty["name"],
                    "prompt": prompt,
                    "interpreted_action": interpretation.get("action_name", "unknown"),
                    "success": interpretation.get("action_name") == target_action_name,
                    "interpretation": interpretation
                }

                self.results.append(result)

            except Exception as e:
                # Record the failure
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "target_action": target_action_name,
                    "difficulty_class": difficulty["name"],
                    "prompt": "Error generating prompt",
                    "interpreted_action": "error",
                    "success": False,
                    "error": str(e)
                }
                self.results.append(result)

        return self.results

    async def run_continuous_evaluation(self, duration_seconds=None, max_evaluations=None):
        """
        Runs a continuous evaluation, randomly selecting actions to test.

        Args:
            duration_seconds: If set, run for this many seconds
            max_evaluations: If set, run this many evaluations
        """
        start_time = time.time()
        evaluation_count = 0

        all_actions = list(self.action_manager.get_all_actions().keys())
        # Filter out utility actions that aren't real user intents
        excluded_actions = ["ask_for_missing_info", "no_suiting_action", "system_capabilities"]
        actions_to_evaluate = [action for action in all_actions if action not in excluded_actions]

        while True:
            # Check if we should stop
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break
            if max_evaluations and evaluation_count >= max_evaluations:
                break

            # Select a random action to evaluate
            target_action = random.choice(actions_to_evaluate)

            await self.evaluate_action(target_action)
            evaluation_count += 1

            # Wait 20 seconds before the next evaluation
            await asyncio.sleep(20)

        return self.results

    def save_results(self, filename="intent_evaluation_results.json"):
        """Save the evaluation results to a JSON file"""
        output_path = f"{filename}"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Also generate a simple summary
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])

        summary = {
            "total_evaluations": total,
            "successful_evaluations": successful,
            "accuracy": successful / total if total > 0 else 0,
            "by_difficulty": {},
            "by_action": {}
        }

        # Breakdown by difficulty
        for difficulty in self.difficulty_classes:
            difficulty_name = difficulty["name"]
            difficulty_results = [r for r in self.results if r["difficulty_class"] == difficulty_name]
            total_difficulty = len(difficulty_results)
            successful_difficulty = sum(1 for r in difficulty_results if r["success"])

            summary["by_difficulty"][difficulty_name] = {
                "total": total_difficulty,
                "successful": successful_difficulty,
                "accuracy": successful_difficulty / total_difficulty if total_difficulty > 0 else 0
            }

        # Breakdown by action
        action_names = set(r["target_action"] for r in self.results)
        for action_name in action_names:
            action_results = [r for r in action_results if r["target_action"] == action_name]
            total_action = len(action_results)
            successful_action = sum(1 for r in action_results if r["success"])

            summary["by_action"][action_name] = {
                "total": total_action,
                "successful": successful_action,
                "accuracy": successful_action / total_action if total_action > 0 else 0
            }

        # Save summary
        summary_path = f"intent_evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
    

class IntentEvaluationAnalyzer:
    """Class for analyzing intent recognition evaluation results."""

    def __init__(self, file_path: str):
        """
        Initialize the analyzer with data from the specified JSON file.

        Args:
            file_path: Path to the intent_evaluation_results.json file
        """
        self.file_path = file_path
        self.data = None
        self.df = None
        self.output_dir = "intent_analysis_results"

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self) -> None:
        """Load and parse the JSON data file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            print(f"Successfully loaded {len(self.data)} evaluation entries")

            # Convert to DataFrame for easier analysis
            self.df = pd.DataFrame(self.data)

            # Parse timestamps
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

            # Extract parameter count for each entry
            self.df['param_count'] = self.df['interpretation'].apply(
                lambda x: len(x.get('parameters', {})) if isinstance(x, dict) else 0
            )

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def overall_accuracy(self) -> Dict[str, float]:
        """
        Calculate overall accuracy metrics.

        Returns:
            Dictionary containing accuracy metrics
        """
        if self.df is None:
            self.load_data()

        total = len(self.df)
        correct = self.df['success'].sum()
        accuracy = correct / total if total > 0 else 0

        return {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'error_rate': 1 - accuracy
        }

    def accuracy_by_difficulty(self) -> pd.DataFrame:
        """
        Calculate accuracy metrics grouped by difficulty class.

        Returns:
            DataFrame with accuracy metrics by difficulty class
        """
        if self.df is None:
            self.load_data()

        result = self.df.groupby('difficulty_class').agg(
            total=('success', 'count'),
            correct=('success', 'sum')
        )

        result['accuracy'] = result['correct'] / result['total']
        return result.sort_values('accuracy', ascending=False)

    def accuracy_by_action(self) -> pd.DataFrame:
        """
        Calculate accuracy metrics grouped by target action.

        Returns:
            DataFrame with accuracy metrics by target action
        """
        if self.df is None:
            self.load_data()

        result = self.df.groupby('target_action').agg(
            total=('success', 'count'),
            correct=('success', 'sum')
        )

        result['accuracy'] = result['correct'] / result['total']
        return result.sort_values('accuracy', ascending=False)

    def generate_confusion_matrix(self) -> pd.DataFrame:
        """
        Generate a confusion matrix for intent recognition.

        Returns:
            DataFrame representing the confusion matrix
        """
        if self.df is None:
            self.load_data()

        # Create confusion matrix
        confusion = pd.crosstab(
            self.df['target_action'],
            self.df['interpreted_action'],
            normalize='index'
        )

        return confusion

    def time_trends(self) -> Optional[pd.DataFrame]:
        """
        Analyze accuracy trends over time.

        Returns:
            DataFrame with accuracy metrics by time period or None if insufficient data
        """
        if self.df is None:
            self.load_data()

        # Check if we have enough timestamp variation
        time_range = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
        if time_range < 3600:  # Less than an hour of data
            return None

        # Group by day and calculate accuracy
        self.df['date'] = self.df['timestamp'].dt.date
        time_trend = self.df.groupby('date').agg(
            total=('success', 'count'),
            correct=('success', 'sum')
        )

        time_trend['accuracy'] = time_trend['correct'] / time_trend['total']
        return time_trend

    def plot_overall_accuracy(self) -> None:
        """Generate and save overall accuracy pie chart."""
        metrics = self.overall_accuracy()

        labels = ['Correct', 'Incorrect']
        sizes = [metrics['accuracy'], metrics['error_rate']]
        colors = ['#4CAF50', '#F44336']  # Green for correct, red for incorrect

        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.1, 0))
        plt.title('Overall Intent Recognition Accuracy', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/overall_accuracy.png", dpi=300)
        plt.close()

    def plot_accuracy_by_difficulty(self) -> None:
        """Generate and save accuracy by difficulty bar chart."""
        df_diff = self.accuracy_by_difficulty()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_diff.index, df_diff['accuracy'], color=sns.color_palette("viridis", len(df_diff)))

        plt.title('Intent Recognition Accuracy by Difficulty Class', fontsize=16)
        plt.xlabel('Difficulty Class')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)

        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=11)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/accuracy_by_difficulty.png", dpi=300)
        plt.close()

    def plot_accuracy_by_action(self) -> None:
        """Generate and save accuracy by action bar chart."""
        df_action = self.accuracy_by_action()

        plt.figure(figsize=(14, 8))
        bars = plt.bar(df_action.index, df_action['accuracy'], color=sns.color_palette("mako", len(df_action)))

        plt.title('Intent Recognition Accuracy by Target Action', fontsize=16)
        plt.xlabel('Target Action')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha='right')

        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=10)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/accuracy_by_action.png", dpi=300)
        plt.close()

    def plot_confusion_matrix(self) -> None:
        """Generate and save confusion matrix heatmap."""
        confusion = self.generate_confusion_matrix()

        plt.figure(figsize=(16, 12))
        sns.heatmap(confusion, annot=True, fmt='.1%', cmap='Blues', linewidths=.5)
        plt.title('Intent Recognition Confusion Matrix', fontsize=16)
        plt.ylabel('Target Action')
        plt.xlabel('Interpreted Action')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300)
        plt.close()

    def plot_parameter_distribution(self) -> None:
        """Generate and save parameter distribution chart."""
        if self.df is None:
            self.load_data()

        param_counts = self.df[self.df['success'] == True]['param_count'].value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(param_counts.index.astype(str), param_counts.values,
                       color=sns.color_palette("YlGnBu", len(param_counts)))

        plt.title('Distribution of Parameter Count in Successfully Recognized Intents', fontsize=16)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Frequency')

        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     str(int(height)), ha='center', va='bottom')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_distribution.png", dpi=300)
        plt.close()

    def generate_summary_report(self) -> str:
        """
        Generate a textual summary report of the analysis.

        Returns:
            String containing the summary report
        """
        # Get metrics
        overall = self.overall_accuracy()
        by_difficulty = self.accuracy_by_difficulty()
        by_action = self.accuracy_by_action()

        # Best and worst performing actions
        best_action = by_action.sort_values('accuracy', ascending=False).iloc[0]
        worst_action = by_action.sort_values('accuracy', ascending=True).iloc[0]

        # Generate report
        report = [
            "=== Intent Recognition Evaluation Summary ===",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total samples analyzed: {overall['total_samples']}",
            "",
            "--- Overall Performance ---",
            f"Overall accuracy: {overall['accuracy']:.1%}",
            f"Total correct: {overall['correct']} out of {overall['total_samples']}",
            "",
            "--- Performance by Difficulty Class ---"
        ]

        # Add difficulty class breakdown
        for idx, row in by_difficulty.iterrows():
            report.append(f"{idx}: {row['accuracy']:.1%} ({int(row['correct'])}/{int(row['total'])})")

        report.extend([
            "",
            "--- Best and Worst Performing Intents ---",
            f"Best: {best_action.name} - {best_action['accuracy']:.1%} accuracy",
            f"Worst: {worst_action.name} - {worst_action['accuracy']:.1%} accuracy",
        ])

        return "\n".join(report)

    def run_analysis(self) -> None:
        """Run all analyses and generate all visualizations."""
        print("Starting intent evaluation analysis...")

        # Load data if not already loaded
        if self.df is None:
            self.load_data()

        # Generate all visualizations
        print("Generating visualizations...")
        self.plot_overall_accuracy()
        self.plot_accuracy_by_difficulty()
        self.plot_accuracy_by_action()
        self.plot_confusion_matrix()
        self.plot_parameter_distribution()

        # Generate summary report
        print("Generating summary report...")
        report = self.generate_summary_report()
        report_path = f"{self.output_dir}/summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # Print completion message
        print(f"Analysis complete! Results saved to {self.output_dir}/ directory")
        print(f"Summary report saved to {report_path}")


# Main function to run the module
if __name__ == "__main__":
    import time
    import random
    import json

    # Create LLM model, action manager, and query interpreter
    llm_model = GoogleLLMModel(ConfigManager.get_config()["system_prompt"])
    action_manager = ActionManager()
    query_interpreter = QueryInterpreterModule(llm_model, action_manager)

    # Create the evaluator
    evaluator = IntentEvaluator(llm_model, action_manager, query_interpreter)

    file_path = "intent_evaluation_results.json"

    # Run the evaluation
    print("Starting intent recognition evaluation...")
    try:
        # Run for a specified number of evaluations (each evaluation tests one action with all difficulty levels)
        asyncio.run(evaluator.run_continuous_evaluation(max_evaluations=100))

        # Save the results
        summary = evaluator.save_results(filename=file_path)
        print(f"Evaluation complete! Overall accuracy: {summary['accuracy']:.2%}")
        print("Results and summary have been saved to JSON files.")

    except Exception as e:
        print(f"Error during evaluation: {e}")


    try:
        analyzer = IntentEvaluationAnalyzer(file_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"An error occurred during analysis: {e}")