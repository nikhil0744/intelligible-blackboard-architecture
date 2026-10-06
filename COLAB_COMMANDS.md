# 📋 Google Colab Copy-Paste Commands Cheat-Sheet

Use this reference to run any demo or clinical conflict scenario in your Google Colab notebook.

---

## 🔄 0. Pull Latest Code in Colab (Run First in Cell 2)
```bash
%cd /content/project_blackboard
!git pull origin prototype-live-demo
```

---

## ⚡ 1. Clinical Dilemma Prompts (Designed to Trigger Specialist Disagreement)

### Option A: Acute Atrial Fibrillation vs Active Bleeding Gastric Ulcer
*Conflict: High-risk stroke (CHA2DS2-VASc = 5) demanding immediate full-dose anticoagulation vs. active bleeding gastric ulcer where anticoagulants cause fatal hemorrhagic shock.*

```bash
!python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 74-year-old male with acute atrial fibrillation (heart rate 142 bpm, CHA2DS2-VASc = 5) presents simultaneously with massive hematemesis and melena from an active bleeding gastric ulcer (hemoglobin 6.8 g/dL). The clinician proposes immediate therapeutic anticoagulation with full-dose IV Heparin to prevent imminent embolic stroke. The pharmacy and critical care team must evaluate anticoagulation vs hemostasis."
```

---

### Option B: Severe Renal Failure (eGFR 22) vs Metformin Escalation
*Conflict: Uncontrolled hyperglycemia (HbA1c 10.2%) demanding aggressive glycemic control vs. FDA black-box contraindication of Metformin in eGFR < 30 due to fatal lactic acidosis.*

```bash
!python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 66-year-old with poorly controlled Type 2 Diabetes (HbA1c 10.2%) is admitted with acute kidney injury; serum creatinine is 3.4 mg/dL and eGFR is 22 mL/min/1.73m2. The primary clinician insists on continuing and escalating Metformin to 1000 mg BID to manage severe hyperglycemia. What should be done with the Metformin order?"
```

---

### Option C: Septic Shock vs End-Stage Heart Failure (The 30 mL/kg Fluid Challenge)
*Conflict: Surviving Sepsis guidelines requiring immediate 30 mL/kg IV crystalloids (2.5L) vs. severe systolic heart failure (EF 15%) where 2.5L bolus triggers flash pulmonary edema and arrest.*

```bash
!python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 68-year-old female with severe ischemic cardiomyopathy (left ventricular ejection fraction 15%) presents in septic shock from urosepsis (BP 75/40 mmHg, lactate 4.5 mmol/L, fever 39.2C) with baseline bilateral pulmonary crackles. The clinician insists on immediate 30 mL/kg rapid IV crystalloid bolus (2.5 Liters) per Surviving Sepsis campaign guidelines. The cardiology and pharmacology team must evaluate fluid bolus vs early vasopressors."
```

---

## 🛠️ 2. Non-Medical Cross-Domain Prompts

### Option D: Engineering Domain — EV Battery Accelerated Capacity Fade
*Watch Systems Engineers debate battery degradation, manufacturing defects, and anode SEI layer breakdown.*

```bash
!python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain engineering --turns 4
```

---

### Option E: General Domain — Cognitive Economics Bat & Ball Problem
*Watch General Reasoners debate cognitive reflex bias ($0.10 vs $0.05).*

```bash
!python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain general --turns 4
```

---

## 🔬 3. Full Benchmark & Ablation Study
*Runs 4 ablation trials across counterfactual density conditions (0%, 33%, 66%, 100%) and generates research figures.*

```bash
!python -m benchmarks.mscore --live --backend ollama --model qwen2.5:7b-instruct-q4_K_M --trials 4
```

---

## 📊 4. Launch Visualizer (Run in Cell 9)
*After running any of the above commands, run this code in Cell 9 to render the interactive flowchart visualizer:*

```python
import os, glob, json
import IPython.display as display

matches = glob.glob("/content/**/visualizer.html", recursive=True)
vis_path = matches[0] if matches else "visualizer.html"

with open(vis_path, "r", encoding="utf-8") as f:
    html_content = f.read()

trace_file = "outputs/live_trace.json"
if os.path.exists(trace_file):
    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    print(f"✅ Ingested live trace: {trace_file} ({len(trace_data.get('steps', []))} steps)")
    safe_json = json.dumps(trace_data).replace("</script>", "<\\/script>")
    injection = f"<script>window.LIVE_TRACE_DATA = {safe_json};</script>"
    html_content = html_content.replace("<head>", f"<head>\n  {injection}")

display.display(display.HTML(html_content))
```
