# BTC Codex 專案指示

## 執行環境

- 本專案預設使用 Conda environment `llm`。
- 執行 Python、測試、訓練、推論或其他需要專案環境的命令時，優先使用 `conda run -n llm ...`；需要互動式 shell 時使用 `conda activate llm`。
- 不要未告知就切換到其他 Python 或 Conda environment。若 `llm` 不存在，先回報阻塞原因。

## 技能路由

- 對本專案的程式碼實作、程式碼審查、測試工作、可重現除錯或根因分析，預設使用 `$subagent-worker`：`C:\\Users\\louis\\.codex\\skills\\subagent-worker\\SKILL.md`。
- 涉及 AI/ML、PyTorch、OpenCV、CUDA、模型訓練、影像增強、工業瑕疵檢測、實驗設計、論文/研究分析、量化驗證或 ONNX/TensorRT/Jetson 部署時，使用 `$study-work`：`C:\\Users\\louis\\.codex\\skills\\study-work\\SKILL.md`。
- 同時符合兩者時，先依 `$study-work` 建立技術診斷與驗證方法，再依 `$subagent-worker` 委派受限且可驗證的工作。

## 工作要求

- 保留與目前任務無關的既有修改。
- 先確認目標、範圍與驗證方式，再修改檔案。
- 完成後提供實際執行的命令、結果與尚未解決的風險；不要宣稱未驗證的成功。
