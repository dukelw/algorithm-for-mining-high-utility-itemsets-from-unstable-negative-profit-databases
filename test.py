def contains_same_characters(E: list, target: str) -> bool:
    """
    Checks if a list of strings (E) contains a specific string (target) with the same characters.

    Parameters:
    E (list): A list of strings.
    target (str): The string to check for in the list.

    Returns:
    bool: True if the list contains the target string with the same characters, False otherwise.
    """
    target_set = set(target)

    for item in E:
        if set(item) == target_set:
            return True

    return False


# Example usage
order = ["a", "abe", "c", "d", "b", "g", "e", "h"]
print(contains_same_characters(order, "ba"))
