# nlp/preProcess.py
from collections import Counter
import inflect  # to deal with plurals

# Sources that helped, but I had to do a lot of tweaking
# https://pypi.org/project/inflect/
# https://stackoverflow.com/questions/12206276/how-to-check-if-given-word-is-in-plural-or-singular-form
# https://realpython.com/python-counter/

def preprocess_labels(labels):
    p = inflect.engine()

    # Combine action labels
    combined_labels = []
    for label in labels:
        label_parts = label.split()
        if len(label_parts) > 1:
            action = label_parts[1]
            object = label_parts[0]

            # If action already exists in combined_labels, add to object
            if any(action in s for s in combined_labels):
                for i, s in enumerate(combined_labels):
                    if action in s:
                        combined_labels[i] = f"{s} and {object}"
            else:
                combined_labels.append(label)
        else:
            combined_labels.append(label)

    # Count occurrences of each label
    counter = Counter(combined_labels)

    processed_labels = []
    for label, count in counter.items():
        # If count is greater than 1, make label plural
        if count > 1:
            label_parts = label.split()
            # Checks if action exists in the label
            if len(label_parts) > 1:
                action = label_parts[1]
                object = label_parts[0]
                object = p.plural(object)
                label = f"{object} {action}"
            else:
                label = p.plural(label)

        # Add label (or its plural) and its count to processed_labels
        processed_labels.append(f"{count} {label}")

    return processed_labels
