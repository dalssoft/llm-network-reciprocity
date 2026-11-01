from datetime import datetime
from typing import Any, Dict, List


def remove_indentation(text: str, empty_lines: bool = True) -> str:
    """
    Remove indentation from the text.
    Empty lines are lines at the beginning or end of the text that are empty.
    """
    lines = text.split("\n")
    if empty_lines:
        lines = [line for line in lines if line.strip()]
    return "".join([line.lstrip() + "\n" for line in lines])


def to_bullet_points(itens: Any, bullet_point: str = "-") -> str:
    """
    Convert a list or dictionary to a string of bullet points.
    Ex:
    to_bullet_points(["item1", "item2", "item3"])
    to_bullet_points({"item1": "value1", "item2": "value2", "item3": "value3"})

    Output:
    - item1
    - item2
    - item3
    """
    if isinstance(itens, Dict):
        return dict_to_bullet_points(itens, bullet_point)
    elif isinstance(itens, List):
        return list_to_bullet_points(itens, bullet_point)
    else:
        raise ValueError(f"Unsupported type: {type(itens)}")


def dict_to_bullet_points(d: Dict, bullet_point: str = "-") -> str:
    return trim_new_lines("".join([f"{bullet_point}{k}: {v}\n" for k, v in d.items()]))


def list_to_bullet_points(l: List, bullet_point: str = "-") -> str:
    return trim_new_lines("".join([f"{bullet_point}{item}\n" for item in l]))


def to_topics(itens: Dict, topic_tag: str = "*") -> str:
    """
    Convert a dictionary to a string of topics.

    Ex:
    to_topics({"topic1": "value1", "topic2": "value2", "topic3": "value3"})

    Output:
    * topic1
    value1
    * topic2
    value2
    """
    return trim_new_lines(
        "".join([f"{topic_tag}{k}{topic_tag}\n{v}\n" for k, v in itens.items()])
    )


def add_tags(tag: str, text: str) -> str:
    """
    Add tags to the text.

    Ex:
    add_tags("topic", "value")

    Output:
    <topic>
    value
    </topic>
    """
    return f"<{tag}>\n{text}\n</{tag}>"


def trim_new_lines(text: str) -> str:
    """
    Remove new lines from the text.
    """
    return text.rstrip("\n")


def time_ago(current_time: datetime, timestamp: datetime) -> str:
    """
    Return the time ago from the timestamp in days.
    """
    delta = current_time - timestamp
    if delta.days == 0:
        return "today"
    elif delta.days == 1:
        return "yesterday"
    else:
        return f"{delta.days} days ago"


def replace_new_lines(text: str, to: str = " ") -> str:
    """
    Replace new lines with a string.
    """
    text = text.replace("\n\r", to)
    text = text.replace("\r\n", to)
    text = text.replace("\n", to)
    text = text.replace("\r", to)
    return text
