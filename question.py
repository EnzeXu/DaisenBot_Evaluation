def question_formatting(question_type: str, question: str, options: list = None):
    # Display question type
    formatted_type = question_type.replace("_", " ").title()
    output = f"Question Type: {formatted_type}\n\n"
    
    # Display question
    output += f"Q: {question}\n"
    
    # Display options if the question type is multiple choice
    if question_type == "multiple_choice" and options:
        output += "\nOptions:\n"
        for i, option in enumerate(options, start=1):
            output += f"  {chr(64 + i)}) {option}\n"
    
    return output