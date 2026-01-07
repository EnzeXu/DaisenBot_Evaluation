import json
import requests
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from eval import sentence_similarity
from agent import gpt_baseline

def question_formatting(question_type: str, question: str, options: list = None):
    output = ""
    
    # Add instruction for AI system first
    if question_type == "multiple_choice" and options:
        output += "Answer with ONLY the letter of your chosen option in the format [A], [B], [C], or [D]. Do not include any explanation, reasoning, or additional text. Your response must be exactly one of: [A], [B], [C], or [D].\n\n"
    elif question_type == "open_ended":
        output += "Try your best to provide a clear response to the question below (within 100 words)\n\n"
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


def process_one_question(benchmark: str,
                         question_id: str,
                         view_record_dir: str,
                         question_dir: str,
                         screenshot_dir: str,
                         debug: bool = False):
    """
    Build and send the frontend-like request to the local API on port 3001 for a single question.

    Returns:
        (status_code:int, usage_total_tokens:int, response_text:str)
    """
    try:
        vr_path = Path(view_record_dir) / f"{benchmark}.json"
        q_path = Path(question_dir) / f"{benchmark}.json"
        ss_path = Path(screenshot_dir) / f"{benchmark}_screenshots.json"
        logs_path = Path("logs.txt")

        # load question file and find question entry
        with open(q_path, "r") as f:
            questions = json.load(f)
        q_entry = next((q for q in questions if q.get("id") == question_id), None)
        if q_entry is None:
            raise FileNotFoundError(f"Question id {question_id} not found in {q_path}")

        view_id = q_entry.get("view_id")
        # load view_record and find matching view
        with open(vr_path, "r") as f:
            views = json.load(f)
        view_entry = next((v for v in views if v.get("id") == view_id), None)
        if view_entry is None:
            raise FileNotFoundError(f"View id {view_id} not found in {vr_path}")

        view_url = view_entry.get("view_url")
        # parse start/end times from view_url query
        parsed = urlparse(view_url)
        qs = parse_qs(parsed.query)
        start_time = float(qs.get("starttime", ["0"])[0])
        end_time = float(qs.get("endtime", ["0"])[0])

        # load screenshot file and locate a data URL for this view (fallback to first found)
        image_data_url = None
        if ss_path.exists():
            with open(ss_path, "r") as f:
                screenshots = json.load(f)
            # try to find an entry that mentions view_id or view_url
            def find_data_url(obj):
                if isinstance(obj, str):
                    if obj.startswith("data:image/"):
                        return obj
                    return None
                if isinstance(obj, dict):
                    for v in obj.values():
                        res = find_data_url(v)
                        if res:
                            return res
                if isinstance(obj, list):
                    for item in obj:
                        res = find_data_url(item)
                        if res:
                            return res
                return None

            # try entries that match view id first
            for e in screenshots:
                if isinstance(e, dict) and (e.get("view_id") == view_id or e.get("id") == view_id or e.get("view_url") == view_url):
                    image_data_url = find_data_url(e)
                    if image_data_url:
                        break
            if image_data_url is None:
                # fallback: find first data url anywhere
                image_data_url = find_data_url(screenshots)

        if not image_data_url:
            raise FileNotFoundError(f"Could not find image data url in {ss_path}")

        # prepare messages per spec
        assistant_msg = {
            "content": [
                {"text": "Hello! What can I help you with today?", "type": "text"}
            ],
            "role": "assistant"
        }
        system_msg = {
            "content": [
                {"text": f"[Current URL (Remember it, but there is no need to mention it unless the user asks for it)] {view_url}\n", "type": "text"}
            ],
            "role": "system"
        }

        # produce user text using existing question_formatting
        q_text = question_formatting(q_entry.get("question_type", "open_ended"), q_entry.get("question", ""), q_entry.get("options"))

        if debug:
            print("========== Formatted Question Text ==========")
            print(q_text)
            print()
        user_msg = {
            "content": [
                {"text": q_text, "type": "text"},
                {"image_url": {"url": image_data_url}, "type": "image_url"}
            ],
            "files": [
                {
                    "content": image_data_url,
                    "id": 1,
                    "name": "Screenshot #1",
                    "size": "100.0 KB",
                    "type": "image-screenshot"
                }
            ],
            "role": "user"
        }

        payload = {
            "messages": [assistant_msg, system_msg, user_msg],
            "traceInfo": {
                "endTime": end_time,
                "selected": 1,
                "selectedComponentNameList": [],
                "startTime": start_time
            },
            "selectedGitHubRoutineKeys": ["GPU.SA.L1VTLB"] # GPU.SA.L1VTLB, Driver
        }

        # log request
        with open(logs_path, "a") as lf:
            lf.write("REQUEST:\n")
            json.dump(payload, lf, indent=2)
            lf.write("\n\n")

        # send to local API on port 3001 (POST to root)
        resp = requests.post("http://localhost:3001/api/gpt", json=payload, timeout=100)
        status_code = resp.status_code

        # prefer to parse JSON only when the server returned JSON
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            resp_text = resp.text
            with open(logs_path, "a") as lf:
                lf.write("RESPONSE (non-json or error):\n")
                lf.write(resp_text + "\n\n")
            return status_code, 0, resp_text, q_text, image_data_url

        resp_json = resp.json()

        # log response
        with open(logs_path, "a") as lf:
            lf.write("RESPONSE:\n")
            json.dump(resp_json, lf, indent=2)
            lf.write("\n\n")

        usage_tokens = int(resp_json.get("usage", {}).get("total_tokens", 0))
        # content may be nested under choices[0].message.content (string)
        choices = resp_json.get("choices", [])
        content_str = ""
        if choices and isinstance(choices, list):
            first = choices[0].get("message", {}).get("content")
            if isinstance(first, str):
                content_str = first
            else:
                # if content is structured, convert to string
                content_str = json.dumps(first)
        if debug:
            print(f"========== DaisenBot (status: {status_code}, usage: {usage_tokens} tokens) ==========")
            print(content_str)
            print()

        return status_code, usage_tokens, content_str, q_text, image_data_url

    except Exception as e:
        # log exception
        with open("logs.txt", "a") as lf:
            lf.write(f"ERROR: {str(e)}\n")
        return 500, 0, str(e), q_text, image_data_url

def load_ground_truth(benchmark: str, question_id: str, ground_truth_dir: str, debug: bool=False) -> str:
    """
    Load the ground-truth answer for a given benchmark and question_id.
    Returns the answer string or an empty string if not found / on error.
    """
    try:
        gt_path = Path(f"{ground_truth_dir}/{benchmark}.json")
        if not gt_path.exists():
            return ""
        with open(gt_path, "r") as f:
            labels = json.load(f)
        entry = next((e for e in labels if e.get("question_id") == question_id or e.get("id") == question_id), None)
        if not entry:
            return ""
        answer = str(entry.get("answer", ""))
        if debug:
            print(f"========== Ground Truth Answer Loaded ==========")
            print(answer)
            print()
        return answer
    except Exception:
        return ""

if __name__ == "__main__":
    # basic test case
    benchmark = "bicg"
    question_id = "Q10000000102" # "Q10000000002"
    debug = True

    default_view_record_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "view_record")
    default_ground_truth_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "label")
    default_question_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "question")
    default_screenshot_dir = str(Path(__file__).parent / "screenshot")

    status, usage, resp_text, q_text, image_data_url = process_one_question(
        benchmark=benchmark,
        question_id=question_id,
        view_record_dir=default_view_record_dir,
        question_dir=default_question_dir,
        screenshot_dir=default_screenshot_dir,
        debug=debug
    )

    gpt_baseline_resp_with_image = gpt_baseline(q_text, image_data_url, debug=debug)
    gpt_baseline_resp_without_image = gpt_baseline(q_text, "", debug=debug)
    
    gt_text = load_ground_truth(benchmark, question_id, default_ground_truth_dir, debug)

    score_daisenbot = sentence_similarity(resp_text, gt_text)
    score_baseline_with_image = sentence_similarity(gpt_baseline_resp_with_image, gt_text)
    score_baseline_without_image = sentence_similarity(gpt_baseline_resp_without_image, gt_text)

    print(f"Similarity Score (DaisenBot vs. ground truth): {score_daisenbot:.4f}")
    print(f"Similarity Score (GPT Baseline with image vs. ground truth): {score_baseline_with_image:.4f}")
    print(f"Similarity Score (GPT Baseline without image vs. ground truth): {score_baseline_without_image:.4f}")