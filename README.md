# Intent Recognition and Evaluation Project

## Overview

This project evaluates the performance of an intent recognition system for a room booking application. It uses a large language model (LLM) to interpret user queries and determine the intended action, then compares the interpreted action against a predefined target action to measure accuracy.

## Project Structure

```
intent_evaluation_results.json  # JSON file containing the evaluation results
main.py                         # Main script to run the evaluation and analysis
Readme.md                       # This file
intent_analysis_results/        # Directory containing analysis reports and visualizations
    accuracy_by_action.png      # Accuracy by action bar chart
    accuracy_by_difficulty.png  # Accuracy by difficulty bar chart
    confusion_matrix.png        # Confusion matrix heatmap
    overall_accuracy.png        # Overall accuracy pie chart
    summary_report.txt          # Textual summary report
src/                            # Source code directory
    action.py                   # Manages available actions
    config.py                   # Configuration settings
    interpreter.py              # Interprets user queries
    model.py                    # LLM model interface
```

## Key Components

*   **`intent_evaluation_results.json`**:  This file stores the results of evaluating the intent recognition system. Each entry includes the original user prompt, the target action, the interpreted action, and a success flag.  Example entries:

    ```json
    {
        "timestamp": "2025-04-11T11:53:40.038305",
        "target_action": "list_available_rooms",
        "difficulty_class": "Complex",
        "prompt": "Hey, I'm planning a small workshop...",
        "interpreted_action": "list_available_rooms",
        "success": true,
        "interpretation": { ... }
    }
    ```

*   **`main.py`**: This script orchestrates the intent recognition evaluation process. It initializes the LLM model, action manager, and query interpreter.  It then runs the evaluation and generates analysis reports and visualizations.

*   **`src/` Directory**:
    *   [`action.py`](src/action.py): Defines and manages the available actions that the intent recognition system can perform (e.g., `book_event_timeslot`, `cancel_booking`, `list_available_rooms`).
    *   [`config.py`](src/config.py): Handles configuration settings for the project, such as API keys and model names.
    *   [`interpreter.py`](src/interpreter.py): Contains the [`QueryInterpreterModule`](src/interpreter.py) class, which uses the LLM model to interpret user queries and determine the intended action.
    *   [`model.py`](src/model.py): Defines the interface for interacting with the LLM. The [`GoogleLLMModel`](src/model.py) class implements this interface for the Google LLM.

## Evaluation Process

The `main.py` script performs the following steps:

1.  **Initialization**: Creates instances of the LLM model ([`GoogleLLMModel`](src/model.py)), action manager ([`ActionManager`](src/action.py)), and query interpreter ([`QueryInterpreterModule`](src/interpreter.py)).
2.  **Evaluation**:  The [`IntentEvaluator`](main.py) class reads user prompts and target actions from `intent_evaluation_results.json`, uses the query interpreter to determine the interpreted action, and compares it to the target action.
3.  **Analysis**: The [`IntentEvaluationAnalyzer`](main.py) class analyzes the evaluation results, calculates accuracy metrics, generates a confusion matrix, and creates visualizations.
4.  **Reporting**: The analyzer generates a summary report (summary_report.txt) and saves it along with the visualizations in the `intent_analysis_results/` directory.

## Analysis and Reporting

The [`IntentEvaluationAnalyzer`](main.py) class provides the following analysis and reporting capabilities:

*   **Overall Accuracy**: Calculates the overall accuracy of the intent recognition system.
*   **Accuracy by Difficulty**:  Calculates accuracy for different difficulty classes (e.g., Simple, Moderate, Complex).
*   **Accuracy by Action**:  Calculates accuracy for each action type (e.g., `book_event_timeslot`, `cancel_booking`).
*   **Confusion Matrix**: Generates a confusion matrix to visualize the types of errors the system is making.
*   **Summary Report**: Creates a textual summary report that includes overall performance metrics, accuracy by difficulty class, and the best and worst-performing intents.  See [`IntentEvaluationAnalyzer.generate_summary_report`](main.py).
*   **Visualizations**: Generates various charts and plots to visualize the analysis results, including:
    *   Overall Accuracy Pie Chart ([`IntentEvaluationAnalyzer.plot_overall_accuracy`](main.py))
    *   Accuracy by Difficulty Bar Chart ([`IntentEvaluationAnalyzer.plot_accuracy_by_difficulty`](main.py))
    *   Accuracy by Action Bar Chart ([`IntentEvaluationAnalyzer.plot_accuracy_by_action`](main.py))
    *   Confusion Matrix Heatmap ([`IntentEvaluationAnalyzer.plot_confusion_matrix`](main.py))
    *   Parameter Distribution Chart ([`IntentEvaluationAnalyzer.plot_parameter_distribution`](main.py))

## How to Run the Project

1.  **Install Dependencies**: Make sure you have the necessary Python packages installed. You can install them using pip:

    ```bash
    pip install pandas matplotlib seaborn
    ```

2.  **Configure API Key**:  Set up the Google LLM API key in `src/config.py`.  See [`GoogleLLMModel.__init__`](src/model.py) for how the API key is used.

3.  **Run the Evaluation**: Execute the `main.py` script:

    ```bash
    python main.py
    ```

4.  **View Results**:  The analysis reports and visualizations will be generated in the `intent_analysis_results/` directory.

## Example Usage

An example of how the system interprets a user prompt is shown below:

**Prompt**: "Book event timeslot for room\_name: \"Meeting Room A\", room\_capacity: 20, event\_name: \"Project Kickoff\", start: \"2024-03-15T10:00\", end: \"2024-03-15T11:00\", duration: 60"

**Interpretation**:

```json
{
    "action_name": "book_event_timeslot",
    "parameters": {
        "room_name": "Meeting Room A",
        "room_capacity": 20,
        "event_name": "Project Kickoff",
        "start": "2024-03-15T10:00:00Z",
        "end": "2024-03-15T11:00:00Z",
        "duration": 60
    },
    "explanation": "Event 'Project Kickoff' booked in 'Room A' from 2024-03-15 10:00 to 2024-03-15 11:00. Duration set to 60 minutes."
}
```