def question_formatting(question_type: str, question: str, options: list = None):
    output = ""
    
    # Add instruction for AI system first
    if question_type == "multiple_choice" and options:
        output += "You are a grading assistant. Answer with ONLY the letter of your chosen option in the format [A], [B], [C], or [D]. Do not include any explanation, reasoning, or additional text. Your response must be exactly one of: [A], [B], [C], or [D].\n\n"
    elif question_type == "explanation":
        # For explanation type, instruct AI to repeat the question only
        output += "You are a grading assistant. Your task is to repeat the following question exactly as written, with no additional text, explanation, or commentary. Only output the question itself.\n\n"
        output += question
        return output
    else:
        # Generic instruction for other question types
        output += "Instruction: Provide a direct and concise answer to the question.\n\n"
    
    # Display question type
    formatted_type = question_type.replace("_", " ").title()
    output += f"Question Type: {formatted_type}\n\n"
    
    # Display question
    output += f"Q: {question}\n"
    
    # Display options if the question type is multiple choice
    if question_type == "multiple_choice" and options:
        output += "\nOptions:\n"
        for i, option in enumerate(options, start=1):
            output += f"  {chr(64 + i)}) {option}\n"
    
    return output