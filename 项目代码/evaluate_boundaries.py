import json
from pathlib import Path
from datetime import date
from notice_agent import extract
CASES = [
 ("明确截止", "请于11月28日前提交申请表。", "2026-11-24", ["2026-11-28"]),
 ("报名与举行", "请于11月26日20点前填写报名表；活动于11月29日19点举办。", "2026-11-24", ["2026-11-26 20:00", "2026-11-29 19:00"]),
 ("相对星期", "下周二上课。", "2026-11-24", ["2026-12-01"]),
 ("缺少发布依据", "下周二上课。", None, [""]),
 ("非法日期", "请于2月30日前提交材料。", "2026-02-01", [""]),
 ("非法钟点", "请于11月28日25点前提交材料。", "2026-11-24", [""]),
 ("显式跨年", "请于2027年1月5日前提交材料。", "2026-12-24", ["2027-01-05"]),
 ("隐式跨年", "请于1月5日前提交材料。", "2026-12-24", [""]),
 ("相对星期钟点", "下周二下午3点上课。", "2026-11-24", ["2026-12-01 15:00"]),
 ("取消活动", "原定11月28日举办的活动取消。", "2026-11-24", []),
]
def main():
 results=[]
 for name,text,published,expected in CASES:
  actual=extract(text,date.fromisoformat(published) if published else None)
  times=[r[2] for r in actual]
  results.append(dict(name=name,text=text,published=published,expected_times=expected,actual=actual,passed=times==expected))
 report=dict(description="人工构造边界回归测试，修复后复测；时间序列完全匹配为通过，不代表真实场景准确率。隐式跨年预期转人工确认，取消通知预期不创建事项。",passed=sum(r["passed"] for r in results),total=len(results),cases=results)
 Path(__file__).with_name("boundary_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
