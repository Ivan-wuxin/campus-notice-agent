import argparse
import csv
import re
from datetime import date, timedelta
from pathlib import Path

MODEL_PATH = Path(__file__).with_name('notice_classifier.joblib')

KEYWORDS = {
    "考试": ("考试", "补考", "准考证", "笔试"),
    "竞赛": ("竞赛", "比赛", "参赛", "选拔赛", "大赛"),
    "课程": ("上课", "调课", "作业"),
    "活动": ("讲座", "志愿", "活动", "分享会"),
    "事务": ("缴费", "登记", "办证", "宿舍", "检修"),
}
DAY = r"(?:(?P<year>\d{4})\s*年\s*)?(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]"
CLOCK = r"(?:\s*(?:上午|下午|中午|晚间|晚上|晚)?\s*\d{1,2}(?:[:：]\d{2}(?:[–—-]\d{1,2}[:：]\d{2})?|\s*[点时](?:\d{1,2}分)?))?"
WHEN = re.compile(rf"(?:{DAY}|(?P<week>本|下)周(?P<weekday>[一二三四五六日天])){CLOCK}")
ACTION = re.compile(r"打印|完成|提交|填写|上传|缴费|报到|领取|报名|参加|检修|上课|举行|举办")
MANUAL_TIME = re.compile(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?")


def classify(notice):
    scores = {label: sum(word in notice for word in words)
              for label, words in KEYWORDS.items()}
    best = max(scores.values())
    matches = [label for label, score in scores.items() if score == best]
    return matches[0] if best and len(matches) == 1 else "待确认"


def normalize(match, published):
    if not match:
        return ""
    if match["week"]:
        if published is None:
            return ""
        weekday = "一二三四五六日".index(match["weekday"].replace("天", "日"))
        offset = weekday - published.weekday() + (7 if match["week"] == "下" else 0)
        day = published + timedelta(days=offset)
        end = match.end("weekday")
    else:
        if not match["year"] and published is None:
            return ""
        try:
            year = int(match["year"]) if match["year"] else published.year
            day = date(year, int(match["month"]), int(match["day"]))
        except ValueError:
            return ""
        if not match["year"] and published and day < published:
            return ""
        end = match.end("day")
    clock = re.search(r"(\d{1,2})(?:[:：](\d{2})|\s*[点时](?:(\d{1,2})分)?)",
                      match.string[end:match.end()])
    if not clock:
        return day.isoformat()
    hour = int(clock[1])
    minute = int(clock[2] or clock[3] or 0)
    if re.search(r"下午|中午|晚", clock.string[:clock.start()]) and 1 <= hour < 12:
        hour += 12
    return f"{day} {hour:02d}:{minute:02d}" if hour < 24 and minute < 60 else ""


def cancelled(text):
    # Negated cancellation and cancellation deadlines are not actual cancellations.
    text = re.sub(r"未取消|不取消|不会取消|无需取消|取消报名|取消资格|取消截止", "", text)
    return bool(re.search(r"取消|不再举办|不再举行|停止举办", text))


def extract(notice, published):
    notice = re.sub(
        r"(材料提交截止时间为[^，。；\n]+)[，,]\s*([^，。；\n]*(?:上传|提交)[^，。；\n]*)",
        r"\1 \2",
        notice,
    )
    # Resolve explicit cancellation within a sentence before comma splitting.
    sentences = re.split(r"[。；\n]", notice)
    active = []
    for sentence in sentences:
        if cancelled(sentence):
            print("检测到取消或停止举办的表述，已跳过该句；请人工核对已有待办：" + sentence)
        else:
            active.append(sentence)
    notice = "；".join(active)
    items = []
    for clause in re.split(r"[，。；！？\n①②③④⑤⑥⑦⑧⑨]", notice):
        clause = clause.strip()
        if not clause or re.fullmatch(r"关于[^：:]*通知", clause):
            continue
        if clause.startswith(("提交后", "现场可", "分享发言", "迟到")):
            continue
        dates = list(WHEN.finditer(clause))
        moved = len(dates) == 2 and "顺延至" in clause
        selected = dates[-1] if len(dates) == 1 or moved else None
        if not ACTION.search(clause.replace("截止时间", "")):
            continue
        kind = "待确认"
        if re.search(r"截止|(?:日|号|点|分|\d)\s*(?:之前|以前|前)|上课前", clause):
            kind = "截止"
        elif selected and re.match(r"\s*(?:起|开始)", clause[selected.end():]):
            kind = "开始"
        elif moved or re.search(r"举行|举办|上课", clause):
            kind = "日程"
        elif "检修时段" in clause:
            kind = "提醒"
        task = clause
        if selected and kind == "截止":
            action = ACTION.search(clause, selected.end())
            if action:
                task = clause[action.start():]
        if moved:
            task = "按调整后的时间上课（" + clause[selected.end():].strip() + "）"
        time = "" if re.search(r"考前\s*\d+\s*天", clause) else normalize(selected, published)
        items.append((kind, task, time))
    return items


def read_time(prompt, default="", with_clock=False):
    while True:
        value = input(prompt).strip()
        if not value:
            return default
        if value == "-":
            return ""
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return date.fromisoformat(value).isoformat()
            if with_clock and MANUAL_TIME.fullmatch(value) and " " in value:
                day, clock = value.split()
                date.fromisoformat(day)
                hour, minute = map(int, clock.split(":"))
                if hour < 24 and minute < 60:
                    return value
        except ValueError:
            pass
        print("日期无效。请输入 YYYY-MM-DD" + (" 或 YYYY-MM-DD HH:MM" if with_clock else "") + "。")


def suggest_category(notice, use_model=False):
    if use_model and MODEL_PATH.is_file():
        try:
            from joblib import load
            return str(load(MODEL_PATH).predict([notice])[0]), "机器学习模型"
        except (ImportError, OSError, ValueError, EOFError) as error:
            print(f"模型无法读取，改用关键词规则（{error}）。")
    return classify(notice), "关键词规则"


def main(use_model=False):
    print("请粘贴通知，输入空行结束。")
    lines = []
    while line := input().strip():
        lines.append(line)
    if not lines:
        print("通知为空。")
        return
    notice = "\n".join(lines)
    published = read_time("发布日期（YYYY-MM-DD，未知可回车）：")
    category, method = suggest_category(notice, use_model)
    category = input(f"建议类别为 {category}（{method}），回车保留或输入修改值：").strip() or category

    items = extract(notice, date.fromisoformat(published) if published else None)
    if not items and cancelled(notice):
        print("未创建新事项；已有CSV记录不会自动删除，请人工核对。")
        return
    rows = []
    for kind, task, time in items or [("待确认", "", "")]:
        print(f"\n{kind}｜{task or '待补充'}｜{time or '时间待确认'}")
        edited = input("修改事项，回车保留，输入 - 跳过：").strip()
        if edited == "-" or not (edited or task):
            continue
        time = read_time("修改时间，回车保留，输入 - 清空：", time, with_clock=True)
        rows.append((category, kind, edited or task, time, published, date.today().isoformat()))
    if not rows or input("\n确认保存以上事项？输入 y 保存：").strip().lower() != "y":
        return

    output = Path(__file__).with_name("todos_v2.csv")
    header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(["类别", "类型", "事项", "时间", "发布日期", "保存日期"])
        writer.writerows(rows)
    print(f"已保存 {len(rows)} 项至 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="校园通知待办 Agent")
    parser.add_argument("--model", action="store_true", help="使用已训练的分类模型")
    main(parser.parse_args().model)
