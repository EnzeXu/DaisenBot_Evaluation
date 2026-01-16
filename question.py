import json
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import eval
from agent import gpt_baseline
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ExperimentWriter:
    def __init__(self, record_dir: str = None):
        self.record_dir = Path(record_dir) if record_dir else Path(__file__).parent / "record"
        self.record_dir.mkdir(parents=True, exist_ok=True)

    def _now_ny(self) -> str:
        return datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%dT%H:%M:%S")

    def _sanitize(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        # normalize newlines then escape them for file safety
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.replace("\n", "\\n")
        return s

    def _strip_prompt_prefix(self, s: str) -> str:
        """
        Remove any leading prompting text up to and including the marker '\n\nQ: '
        so returned text starts with the actual question (e.g. 'What', 'How', ...).
        If the marker is not found, return original string.
        """
        if s is None:
            return ""
        marker = "\n\nQ: "
        if marker in s:
            return s.split(marker, 1)[1]
        # also tolerate '\n\nQ:' without trailing space
        marker2 = "\n\nQ:"
        if marker2 in s:
            return s.split(marker2, 1)[1].lstrip()
        return s

    def _file_path(self, question_type: str) -> Path:
        return self.record_dir / f"question_{question_type}.csv"

    def _ensure_header(self, question_type: str):
        path = self._file_path(question_type)
        if path.exists():
            return
        if question_type == "multiple_choice":
            # time,benchmark,question_id,question_type,n_times_idx,q_text,gt_text,method,response,score
            header = ["time","benchmark","question_id","question_type","n_times_idx","q_text","gt_text","method","response","score"]
        else:  # open_ended
            # time,benchmark,question_id,question_type,n_times_idx,q_text,gt_text,method,response,score_sbert,score_bertscore,score_rougel,score_bleu
            header = ["time","benchmark","question_id","question_type","n_times_idx","q_text","gt_text","method","response","score_sbert","score_bertscore","score_rougel","score_bleu"]
        with path.open("w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(header)

    def write_multiple_choice(self,
                              benchmark: str,
                              question_id: str,
                              question_type: str,
                              n_times_idx: int,
                              q_text: str,
                              gt_text: str,
                              method: str,
                              response: str,
                              score: float):
        self._ensure_header("multiple_choice")
        path = self._file_path("multiple_choice")
        time_str = self._now_ny()
        # strip prompting before writing q_text
        q_text_clean = self._strip_prompt_prefix(q_text)
        row = [
            time_str,
            self._sanitize(benchmark),
            self._sanitize(question_id),
            self._sanitize(question_type),
            str(n_times_idx),
            self._sanitize(q_text_clean),
            self._sanitize(gt_text),
            self._sanitize(method),
            self._sanitize(response),
            format(float(score), ".6f")
        ]
        with path.open("a", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(row)

    def write_open_ended(self,
                         benchmark: str,
                         question_id: str,
                         question_type: str,
                         n_times_idx: int,
                         q_text: str,
                         gt_text: str,
                         method: str,
                         response: str,
                         score_sbert: float,
                         score_bertscore: float,
                         score_rougel: float,
                         score_bleu: float):
        self._ensure_header("open_ended")
        path = self._file_path("open_ended")
        time_str = self._now_ny()
        q_text_clean = self._strip_prompt_prefix(q_text)
        row = [
            time_str,
            self._sanitize(benchmark),
            self._sanitize(question_id),
            self._sanitize(question_type),
            str(n_times_idx),
            self._sanitize(q_text_clean),
            self._sanitize(gt_text),
            self._sanitize(method),
            self._sanitize(response),
            format(float(score_sbert), ".6f"),
            format(float(score_bertscore), ".6f"),
            format(float(score_rougel), ".6f"),
            format(float(score_bleu), ".6f"),
        ]
        with path.open("a", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(row)

    def generate_summary(self, question_type: str, n_times: int):
        """
        Read question_{question_type}.csv and produce question_{question_type}_summary.csv.
        - For multiple_choice: aggregate per (benchmark,question_id,question_type,q_text,gt_text,method)
          compute score_avg and score_most (1 if count_of_1s >= ceil(n_times/2) else 0).
          response in summary is the most frequent response across runs; time uses the latest time string.
        - For open_ended: aggregate per (benchmark,question_id,question_type,q_text,gt_text,method)
          compute averages for sbert, bertscore, rougel, bleu. Response is the most frequent response; time uses latest.
        Existing summary file will be replaced.
        """
        import math
        from collections import Counter, defaultdict

        src_path = self._file_path(question_type)
        summary_path = self.record_dir / f"question_{question_type}_summary.csv"

        if not src_path.exists():
            # nothing to summarize; remove old summary if exists
            if summary_path.exists():
                summary_path.unlink()
            return

        # read rows
        with src_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        groups = {}  # key -> aggregation dict

        if question_type == "multiple_choice":
            # expected columns: time,benchmark,question_id,question_type,n_times_idx,q_text,gt_text,method,response,score
            for r in rows:
                key = (
                    r["benchmark"],
                    r["question_id"],
                    r["question_type"],
                    r["q_text"],
                    r["gt_text"],
                    r["method"],
                )
                g = groups.get(key)
                if g is None:
                    g = {
                        "times": [],
                        "responses": Counter(),
                        "scores": [],
                    }
                    groups[key] = g
                g["times"].append(r.get("time", ""))
                resp = r.get("response", "")
                g["responses"][resp] += 1
                try:
                    g["scores"].append(float(r.get("score", "0")))
                except Exception:
                    g["scores"].append(0.0)

            # write summary (replace existing)
            with summary_path.open("w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                header = ["time","benchmark","question_id","question_type","q_text","gt_text","method","response","score_most","score_avg"]
                writer.writerow(header)
                threshold = math.ceil(n_times / 2)
                for key, g in groups.items():
                    benchmark, question_id, question_type_k, q_text, gt_text, method = key
                    # choose latest time (ISO format string)
                    time_val = max(g["times"]) if g["times"] else ""
                    # most common response
                    response_most, _ = g["responses"].most_common(1)[0] if g["responses"] else ("", 0)
                    scores = g["scores"]
                    score_avg = sum(scores) / len(scores) if scores else 0.0
                    ones = sum(1 for s in scores if float(s) >= 0.5)  # treat >=0.5 as 1
                    score_most = 1 if ones >= threshold else 0
                    writer.writerow([
                        time_val,
                        benchmark,
                        question_id,
                        question_type_k,
                        q_text,
                        gt_text,
                        method,
                        response_most,
                        str(score_most),
                        format(score_avg, ".6f")
                    ])

        elif question_type == "open_ended":
            # expected columns:
            # time,benchmark,question_id,question_type,n_times_idx,q_text,gt_text,method,response,score_sbert,score_bertscore,score_rougel,score_bleu
            for r in rows:
                key = (
                    r["benchmark"],
                    r["question_id"],
                    r["question_type"],
                    r["q_text"],
                    r["gt_text"],
                    r["method"],
                )
                g = groups.get(key)
                if g is None:
                    g = {
                        "times": [],
                        "responses": Counter(),
                        "sbert": [],
                        "bertscore": [],
                        "rougel": [],
                        "bleu": []
                    }
                    groups[key] = g
                g["times"].append(r.get("time", ""))
                resp = r.get("response", "")
                g["responses"][resp] += 1
                def safe_float(val):
                    try:
                        return float(val)
                    except Exception:
                        return 0.0
                g["sbert"].append(safe_float(r.get("score_sbert", "0")))
                g["bertscore"].append(safe_float(r.get("score_bertscore", "0")))
                g["rougel"].append(safe_float(r.get("score_rougel", "0")))
                g["bleu"].append(safe_float(r.get("score_bleu", "0")))

            # write summary
            with summary_path.open("w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                header = ["time","benchmark","question_id","question_type","q_text","gt_text","method","response","score_sbert_avg","score_bertscore_avg","score_rougel_avg","score_bleu_avg"]
                writer.writerow(header)
                for key, g in groups.items():
                    benchmark, question_id, question_type_k, q_text, gt_text, method = key
                    time_val = max(g["times"]) if g["times"] else ""
                    response_most, _ = g["responses"].most_common(1)[0] if g["responses"] else ("", 0)
                    def avg(lst):
                        return sum(lst) / len(lst) if lst else 0.0
                    sbert_avg = avg(g["sbert"])
                    bertscore_avg = avg(g["bertscore"])
                    rougel_avg = avg(g["rougel"])
                    bleu_avg = avg(g["bleu"])
                    writer.writerow([
                        time_val,
                        benchmark,
                        question_id,
                        question_type_k,
                        q_text,
                        gt_text,
                        method,
                        response_most,
                        format(sbert_avg, ".6f"),
                        format(bertscore_avg, ".6f"),
                        format(rougel_avg, ".6f"),
                        format(bleu_avg, ".6f"),
                    ])
        else:
            raise ValueError(f"Unsupported question type for summary: {question_type}")


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

        # Build traceInfo and selectedGitHubRoutineKeys from question attachment per rules:
        # (1) if attachment is null -> use start_time/end_time, selected=0, empty selectedComponentNameList, empty selectedGitHubRoutineKeys
        # (2) if attachment.traceInfo is not null -> copy its fields (use defaults for any missing fields)
        # (3) if attachment.selectedGitHubRoutineKeys is present -> copy it (even if empty list)
        # (4) if either subfield is null -> replace with defaults from (1)
        attachment = q_entry.get("attachment")

        default_trace = {
            "endTime": end_time,
            "selected": 0,
            "selectedComponentNameList": [],
            "startTime": start_time
        }

        if attachment:
            trace_info_raw = attachment.get("traceInfo")
            if trace_info_raw is not None:
                # copy traceInfo but ensure required keys exist and fall back to defaults
                trace_info_payload = {
                    "endTime": trace_info_raw.get("endTime", end_time),
                    "selected": trace_info_raw.get("selected", 0),
                    "selectedComponentNameList": trace_info_raw.get("selectedComponentNameList", []) or [],
                    "startTime": trace_info_raw.get("startTime", start_time)
                }
            else:
                trace_info_payload = default_trace

            selected_keys_payload = attachment.get("selectedGitHubRoutineKeys")
            if selected_keys_payload is None:
                selected_keys_payload = []
        else:
            trace_info_payload = default_trace
            selected_keys_payload = []

        payload = {
            "messages": [assistant_msg, system_msg, user_msg],
            "traceInfo": trace_info_payload,
            "selectedGitHubRoutineKeys": selected_keys_payload
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
    
def process_questions(benchmark: str,
                      question_id_list: list,
                      question_dir: str,
                      view_record_dir: str,
                      screenshot_dir: str,
                      ground_truth_dir: str,
                      question_type: str,
                      n_times: int = 3,
                      debug: bool = False):
    
    writer = ExperimentWriter()
    
    for question_id in tqdm(question_id_list):
        for n_times_idx in range(n_times):

            status, usage, daisenbot_text, q_text, image_data_url = process_one_question(
                benchmark=benchmark,
                question_id=question_id,
                view_record_dir=view_record_dir,
                question_dir=question_dir,
                screenshot_dir=screenshot_dir,
                debug=debug
            )

            gpt_baseline_resp_with_image = gpt_baseline(q_text, image_data_url, debug=debug)
            gpt_baseline_resp_without_image = gpt_baseline(q_text, "", debug=debug)
            
            gt_text = load_ground_truth(benchmark, question_id, ground_truth_dir, debug)

            if question_type == "multiple_choice":
                score_daisenbot = eval.multiple_choice_eval(daisenbot_text, gt_text)
                score_baseline_with_image = eval.multiple_choice_eval(gpt_baseline_resp_with_image, gt_text)
                score_baseline_without_image = eval.multiple_choice_eval(gpt_baseline_resp_without_image, gt_text)
                if debug:
                    print(f"Similarity Score (DaisenBot vs. ground truth): {score_daisenbot:.4f}")
                    print(f"Similarity Score (GPT Baseline with image vs. ground truth): {score_baseline_with_image:.4f}")
                    print(f"Similarity Score (GPT Baseline without image vs. ground truth): {score_baseline_without_image:.4f}")

                writer.write_multiple_choice(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "daisenbot_base", daisenbot_text, score_daisenbot
                )
                writer.write_multiple_choice(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "gpt-5.2_with_image", gpt_baseline_resp_with_image, score_baseline_with_image
                )
                writer.write_multiple_choice(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "gpt-5.2_without_image", gpt_baseline_resp_without_image, score_baseline_without_image
                )
            elif question_type == "open_ended":
                # sbert
                sbert_score_daisenbot = eval.sbert_cosine_similarity(daisenbot_text, gt_text)
                sbert_score_baseline_with_image = eval.sbert_cosine_similarity(gpt_baseline_resp_with_image, gt_text)
                sbert_score_baseline_without_image = eval.sbert_cosine_similarity(gpt_baseline_resp_without_image, gt_text)
                # bertscore
                bertscore_daisenbot = eval.bertscore_f1(daisenbot_text, gt_text)
                bertscore_baseline_with_image = eval.bertscore_f1(gpt_baseline_resp_with_image, gt_text)
                bertscore_baseline_without_image = eval.bertscore_f1(gpt_baseline_resp_without_image, gt_text)
                # rouge-l
                rouge_l_daisenbot = eval.rouge_l_f1(daisenbot_text, gt_text)
                rouge_l_baseline_with_image = eval.rouge_l_f1(gpt_baseline_resp_with_image, gt_text)
                rouge_l_baseline_without_image = eval.rouge_l_f1(gpt_baseline_resp_without_image, gt_text)
                # bleu
                bleu_daisenbot = eval.bleu_score(daisenbot_text, gt_text)
                bleu_baseline_with_image = eval.bleu_score(gpt_baseline_resp_with_image, gt_text)
                bleu_baseline_without_image = eval.bleu_score(gpt_baseline_resp_without_image, gt_text)

                if debug:
                    print(f"Similarity Score (DaisenBot vs. ground truth): {sbert_score_daisenbot:.6f} / {bertscore_daisenbot:.6f} / {rouge_l_daisenbot:.4f} / {bleu_daisenbot:.4f}")
                    print(f"Similarity Score (GPT Baseline with image vs. ground truth): {sbert_score_baseline_with_image:.6f} / {bertscore_baseline_with_image:.6f} / {rouge_l_baseline_with_image:.4f} / {bleu_baseline_with_image:.4f}")
                    print(f"Similarity Score (GPT Baseline without image vs. ground truth): {sbert_score_baseline_without_image:.6f} / {bertscore_baseline_without_image:.6f} / {rouge_l_baseline_without_image:.4f} / {bleu_baseline_without_image:.4f}")
                    
                writer.write_open_ended(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "daisenbot_base", daisenbot_text,
                    sbert_score_daisenbot, bertscore_daisenbot, rouge_l_daisenbot, bleu_daisenbot
                )
                writer.write_open_ended(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "gpt-5.2_with_image", gpt_baseline_resp_with_image,
                    sbert_score_baseline_with_image, bertscore_baseline_with_image, rouge_l_baseline_with_image, bleu_baseline_with_image
                )
                writer.write_open_ended(
                    benchmark, question_id, question_type,
                    n_times_idx,
                    q_text, gt_text,
                    "gpt-5.2_without_image", gpt_baseline_resp_without_image,
                    sbert_score_baseline_without_image, bertscore_baseline_without_image, rouge_l_baseline_without_image, bleu_baseline_without_image
                )
            
            else:
                raise ValueError(f"Unsupported question type: {question_type}")

    writer.generate_summary(question_type, n_times)

if __name__ == "__main__":
    # basic test case
    benchmark = "bicg" # matrixmultiplication
    benchmark_id = "10"
    question_id_tail = "02" # 01 for multiple choice, 02 for open ended
    question_id_list = [f"Q{benchmark_id}000000{i}{question_id_tail}" for i in range(5)] # "Q10000000001""Q10000000101","Q10000000201","Q10000000301","Q10000000401"
    question_type = "open_ended" # "open_ended" or "multiple_choice"
    debug = False

    default_view_record_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "view_record")
    default_ground_truth_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "label")
    default_question_dir = str(Path(__file__).parent.parent / "daisenbot_dataset" / "question")
    default_screenshot_dir = str(Path(__file__).parent / "screenshot")

    process_questions(
        benchmark=benchmark,
        question_id_list=question_id_list,
        question_dir=default_question_dir,
        view_record_dir=default_view_record_dir,
        screenshot_dir=default_screenshot_dir,
        ground_truth_dir=default_ground_truth_dir,
        question_type=question_type,
        n_times=3,
        debug=debug
    )

    