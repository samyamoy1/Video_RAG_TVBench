import json
import re

with open("evaluation_results.json", "r", encoding="utf-8") as file:
    results = json.load(file)


def get_direction(text):
    text = text.lower()

    vertical = None
    horizontal = None

    if "down" in text or "bottom" in text:
        vertical = "down"
    elif "up" in text or "top" in text:
        vertical = "up"

    if "right" in text:
        horizontal = "right"
    elif "left" in text:
        horizontal = "left"

    return vertical, horizontal


def get_number(text):
    text = text.lower()

    word_numbers = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twice": 2,
        "thrice": 3,
    }

    match = re.search(r"\b\d+\b", text)

    if match:
        return int(match.group())

    for word, number in word_numbers.items():
        if re.search(r"\b" + word + r"\b", text):
            return number

    return None


def get_time_range(text):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds?",
        text.lower()
    )

    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return start, end

    return None, None


def get_action(text):
    text = text.lower()

    if "put down" in text:
        return "put down"

    if "closed" in text or "close" in text:
        return "close"

    if "opened" in text or "open" in text:
        return "open"

    if "drank" in text or "drink" in text:
        return "drink"

    if "took" in text or "take" in text:
        return "take"

    return None


print("\nTVBench Baseline Evaluation")
print("-" * 35)


# Moving Direction

direction_results = []

for result in results:
    if result["category"] != "moving_direction":
        continue

    gt_vertical, gt_horizontal = get_direction(result["ground_truth"])
    pred_vertical, pred_horizontal = get_direction(result["qwen_answer"])

    vertical_correct = gt_vertical == pred_vertical
    horizontal_correct = gt_horizontal == pred_horizontal

    if vertical_correct and horizontal_correct:
        score = "Correct"
    elif vertical_correct or horizontal_correct:
        score = "Partially Correct"
    else:
        score = "Wrong"

    direction_results.append(score)

print("\nMoving Direction")
print("Correct:", direction_results.count("Correct"))
print("Partially Correct:", direction_results.count("Partially Correct"))
print("Wrong:", direction_results.count("Wrong"))


# Action Count

count_results = []

for result in results:
    if result["category"] != "action_count":
        continue

    ground_truth = int(result["ground_truth"])
    predicted = get_number(result["qwen_answer"])

    if predicted == ground_truth:
        score = "Correct"
    else:
        score = "Wrong"

    count_results.append(score)

    print(f"\n{result['video']}")
    print("Ground truth:", ground_truth)
    print("Qwen:", result["qwen_answer"])
    print("Extracted number:", predicted)
    print("Result:", score)

print("\nAction Count Results")
print("Correct:", count_results.count("Correct"))
print("Wrong:", count_results.count("Wrong"))


# Action Localization

print("\nAction Localization")
print("-" * 35)

for result in results:
    if result["category"] != "action_localization":
        continue

    predicted_start, predicted_end = get_time_range(
        result["qwen_answer"]
    )

    print(f"\n{result['video']}")
    print("Ground truth:", result["ground_truth"])
    print(
        "Ground truth range:",
        result.get("start"),
        "-",
        result.get("end")
    )
    print("Qwen:", result["qwen_answer"])
    print(
        "Predicted range:",
        predicted_start,
        "-",
        predicted_end
    )


# Action Sequence

sequence_results = []

for result in results:
    if result["category"] != "action_sequence":
        continue

    ground_truth_action = get_action(result["ground_truth"])
    predicted_action = get_action(result["qwen_answer"])

    if ground_truth_action == predicted_action:
        score = "Correct"
    else:
        score = "Wrong"

    sequence_results.append(score)

    print(f"\n{result['video']}")
    print("Ground truth:", result["ground_truth"])
    print("Qwen:", result["qwen_answer"])
    print("Ground truth action:", ground_truth_action)
    print("Predicted action:", predicted_action)
    print("Result:", score)

print("\nAction Sequence Results")
print("Correct:", sequence_results.count("Correct"))
print("Wrong:", sequence_results.count("Wrong"))