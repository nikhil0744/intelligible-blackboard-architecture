#!/bin/bash
# colab_commands.sh - Direct execution helper for Colab commands

echo "=========================================================="
echo " INTELLIGIBLE BLACKBOARD - COLAB COMMAND EXECUTION HELPER "
echo "=========================================================="
echo "Usage: Choose an option by number or run manually."
echo ""
echo "Option 1: Atrial Fibrillation vs Active Bleed (Conflict Trigger)"
echo "Option 2: Severe Kidney Injury (eGFR 22) vs Metformin (Conflict Trigger)"
echo "Option 3: Septic Shock vs Heart Failure EF 15% (Conflict Trigger)"
echo "Option 4: EV Battery Engineering Debate"
echo "Option 5: General Bat & Ball Debate"
echo ""

case "$1" in
  1)
    python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 74-year-old male with acute atrial fibrillation (heart rate 142 bpm, CHA2DS2-VASc = 5) presents simultaneously with massive hematemesis and melena from an active bleeding gastric ulcer (hemoglobin 6.8 g/dL). The clinician proposes immediate therapeutic anticoagulation with full-dose IV Heparin to prevent imminent embolic stroke. The pharmacy and critical care team must evaluate anticoagulation vs hemostasis."
    ;;
  2)
    python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 66-year-old with poorly controlled Type 2 Diabetes (HbA1c 10.2%) is admitted with acute kidney injury; serum creatinine is 3.4 mg/dL and eGFR is 22 mL/min/1.73m2. The primary clinician insists on continuing and escalating Metformin to 1000 mg BID to manage severe hyperglycemia. What should be done with the Metformin order?"
    ;;
  3)
    python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain medical --turns 5 --prompt "A 68-year-old female with severe ischemic cardiomyopathy (left ventricular ejection fraction 15%) presents in septic shock from urosepsis (BP 75/40 mmHg, lactate 4.5 mmol/L, fever 39.2C) with baseline bilateral pulmonary crackles. The clinician insists on immediate 30 mL/kg rapid IV crystalloid bolus (2.5 Liters) per Surviving Sepsis campaign guidelines. The cardiology and pharmacology team must evaluate fluid bolus vs early vasopressors."
    ;;
  4)
    python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain engineering --turns 4
    ;;
  5)
    python -m scripts.s2_smoke --backend ollama --model qwen2.5:7b-instruct-q4_K_M --domain general --turns 4
    ;;
  *)
    echo "Please specify option 1-5, e.g.: bash colab_commands.sh 1"
    ;;
esac
