'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Upload, Cpu, Database, Terminal, Microscope, Activity, ArrowRight, CheckCircle2, Fingerprint, Loader2 } from 'lucide-react';

export default function UnifiedPipeline() {
  const [activeStage, setActiveStage] = useState(1);

  // New State Variables for Training
  const [sftParams, setSftParams] = useState({ run_name: "", model: "Qwen/Qwen2.5-0.5B", lr: "2e-4", epochs: 3, batch_size: 4, r: 16 });
  const [sftFile, setSftFile] = useState<File | null>(null);
  const [sftZipUrl, setSftZipUrl] = useState<string | null>(null);

  const [dpoParams, setDpoParams] = useState({ run_name: "", model: "local-sft-model", lr: "5e-5", epochs: 2, batch_size: 2, beta: 0.1 });
  const [dpoFile, setDpoFile] = useState<File | null>(null);
  const [dpoZipUrl, setDpoZipUrl] = useState<string | null>(null);

  // New State for Eval Datasets
  const [sftEvalFile, setSftEvalFile] = useState<File | null>(null);
  const [dpoEvalFile, setDpoEvalFile] = useState<File | null>(null);

  const [completedStages, setCompletedStages] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [modelName, setModelName] = useState("Qwen/Qwen2.5-0.5B");
  const [progress, setProgress] = useState(0);
  const [errorAlert, setErrorAlert] = useState<string | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (loading) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress(prev => {
          // Asymptotic progress: move 8% closer to 100% every second.
          // It will start fast and get slower, but will keep moving (e.g., 90% -> 90.8% -> 91.5%...)
          // so it never looks frozen at exactly 99%.
          const remaining = 100 - prev;
          const increment = remaining * 0.08;
          return prev + increment;
        });
      }, 1000);
    } else {
      // Only snap to 100% if the stage was successfully completed, otherwise drop to 0%
      const wasCompleted = completedStages.includes(activeStage);
      setProgress(wasCompleted ? 100 : 0);
      const timer = setTimeout(() => setProgress(0), 1000);
      return () => clearTimeout(timer);
    }
    return () => clearInterval(interval);
  }, [loading, completedStages, activeStage]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const [syntheticData, setSyntheticData] = useState<any>(null);
  const [evalData, setEvalData] = useState<any>(null);
  const [judgeResults, setJudgeResults] = useState<any[]>([]);
  const [agentSummary, setAgentSummary] = useState<string>("");
  const [sftEval, setSftEval] = useState<any>(null);
  const [dpoEval, setDpoEval] = useState<any>(null);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [reevalData, setReevalData] = useState<any>(null);

  // Checkpoint selector state
  const [sftCheckpoints, setSftCheckpoints] = useState<any[]>([]);
  const [dpoCheckpoints, setDpoCheckpoints] = useState<any[]>([]);
  const [selectedSftCheckpoint, setSelectedSftCheckpoint] = useState<string>("");
  const [selectedDpoCheckpoint, setSelectedDpoCheckpoint] = useState<string>("");

  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string) => {
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
    setLogs(prev => [...prev, `[${timestamp}] ${msg}`]);
  };

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  useEffect(() => {
    if (activeStage === 6) fetchCheckpoints('sft');
    if (activeStage === 8) fetchCheckpoints('dpo');
  }, [activeStage]);

  const handleUpload = async () => {
    if (!uploadFile) {
      setErrorAlert("Please select a file to upload.");
      return;
    }
    setLoading(true);
    addLog(`Initiating upload for file: ${uploadFile.name}`);
    const formData = new FormData();
    formData.append("file", uploadFile);

    try {
      const res = await fetch(`http://localhost:8000/api/v1/upload`, { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        setDatasetName(data.data.file_path);
        addLog(`Upload successful. Path mapped to: ${data.data.file_path}`);
        markComplete(1);
      } else {
        addLog(`[ERROR] Upload failed.`);
      }
    } catch (e) {
      addLog(`[ERROR] Network failure during upload.`);
    }
    setLoading(false);
  };

  const handleBaseEval = async () => {
    if (!datasetName) {
      setErrorAlert("Please upload an evaluation dataset in Stage 1 first.");
      return;
    }
    if (!modelName.trim()) {
      setErrorAlert("Please specify a HuggingFace base model string.");
      return;
    }
    setLoading(true);
    addLog(`Loading base model (${modelName}) & running Base Evaluation...`);
    addLog(`Sharing questions on loop, grading with Medical LLM Judge based on Ground Truth...`);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/evaluate?dataset_name=${encodeURIComponent(datasetName)}&model_name=${encodeURIComponent(modelName)}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setEvalData(data.data);
        setJudgeResults(data.data?.judge_results || []);
        setAgentSummary(data.data?.agent_summary || "");
        addLog(`Base Evaluation complete. ${data.data?.judge_results?.length || 0} samples evaluated.`);
        // markComplete(2); removed to prevent auto-nav
      } else {
        addLog(`[ERROR] Base Evaluation failed.`);
      }
    } catch (e) {
      addLog(`[ERROR] Network failure during evaluation.`);
    }
    setLoading(false);
  };

  const handleSFTDataGen = async () => {
    setLoading(true);
    addLog(`Booting Ollama to generate similar SFT data...`);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/datasets/augment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_path: datasetName })
      });
      if (res.ok) {
        const data = await res.json();
        const count = data.data?.generated_count || 150;
        addLog(`[LLM] Successfully generated ${count} SFT instruction/output pairs.`);
        setSyntheticData(data.data?.real_samples || []);
        markComplete(3);
      } else {
        addLog(`[ERROR] SFT Generation failed.`);
      }
    } catch (e) {
      addLog(`[ERROR] Network failure during SFT generation.`);
    }
    setLoading(false);
  };

  const handleDownloadSynthetic = () => {
    if (!syntheticData || syntheticData.length === 0) {
      setErrorAlert("No SFT data to download.");
      return;
    }
    const blob = new Blob([JSON.stringify(syntheticData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sft_dataset.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleTrainSFT = async () => {
    if (!sftFile) { addLog("[ERROR] No SFT Dataset uploaded."); return; }
    setLoading(true);
    addLog(`Initiating QLoRA Supervised Fine-Tuning (SFT) on ${sftParams.model}...`);
    addLog(`Parameters: LR=${sftParams.lr}, Epochs=${sftParams.epochs}, Batch=${sftParams.batch_size}, r=${sftParams.r}`);

    try {
      const formData = new FormData();
      formData.append("file", sftFile);
      formData.append("params", JSON.stringify(sftParams));

      const res = await fetch(`http://localhost:8000/api/v1/train/sft`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setSftZipUrl(url);
        addLog(`SFT Training completed successfully. QLoRA weights generated and zipped.`);
        markComplete(5);
      } else {
        addLog(`[ERROR] SFT Training failed. Check backend logs.`);
      }
    } catch (error) {
      addLog(`[ERROR] Network failure during SFT training.`);
    }
    setLoading(false);
  };

  const fetchCheckpoints = async (kind: 'sft' | 'dpo') => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/models/checkpoints/${kind}`);
      if (res.ok) {
        const data = await res.json();
        const list = data.data || [];
        if (kind === 'sft') {
          setSftCheckpoints(list);
          if (list.length > 0 && !selectedSftCheckpoint) setSelectedSftCheckpoint(list[0].path);
        } else {
          setDpoCheckpoints(list);
          if (list.length > 0 && !selectedDpoCheckpoint) setSelectedDpoCheckpoint(list[0].path);
        }
      }
    } catch (_) {}
  };

  const deleteCheckpoint = async (kind: 'sft' | 'dpo', name: string) => {
    if (!confirm(`Delete checkpoint '${name}'? This cannot be undone.`)) return;
    try {
      const res = await fetch(`http://localhost:8000/api/v1/models/checkpoints/${kind}/${name}`, { method: 'DELETE' });
      if (res.ok) {
        addLog(`[INFO] Checkpoint '${name}' deleted.`);
        await fetchCheckpoints(kind);
        if (kind === 'sft') setSelectedSftCheckpoint("");
        else setSelectedDpoCheckpoint("");
      } else {
        addLog(`[ERROR] Could not delete checkpoint '${name}'.`);
      }
    } catch (_) { addLog(`[ERROR] Network error deleting checkpoint.`); }
  };

  const handleSFTEval = async () => {
    if (!sftEvalFile) { addLog("[ERROR] No SFT Eval Dataset uploaded."); return; }
    setLoading(true);
    addLog(`Uploading SFT Eval Dataset ${sftEvalFile.name}...`);
    try {
      const formData = new FormData();
      formData.append("file", sftEvalFile);
      const upRes = await fetch(`http://localhost:8000/api/v1/upload`, { method: 'POST', body: formData });
      if (!upRes.ok) throw new Error("Upload failed");
      const upData = await upRes.json();
      const datasetName = upData.data.file_path;

      const params = new URLSearchParams({ dataset_name: datasetName });
      if (selectedSftCheckpoint) params.append("lora_path", selectedSftCheckpoint);

      addLog(`Running Post-SFT Evaluation on ${sftEvalFile.name} with LLM Judge...`);
      const res = await fetch(`http://localhost:8000/api/v1/evaluate/sft?${params.toString()}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setSftEval(data.data);
        addLog(`Post-SFT Evaluation complete. ${data.data?.samples_processed || 0} samples graded.`);
        markComplete(6);
      } else { addLog(`[ERROR] Post-SFT Evaluation failed.`); }
    } catch (e) { addLog(`[ERROR] Network failure during SFT eval.`); }
    setLoading(false);
  };

  const handleDPOEval = async () => {
    if (!dpoEvalFile) { addLog("[ERROR] No DPO Eval Dataset uploaded."); return; }
    setLoading(true);
    addLog(`Uploading DPO Eval Dataset ${dpoEvalFile.name}...`);
    try {
      const formData = new FormData();
      formData.append("file", dpoEvalFile);
      const upRes = await fetch(`http://localhost:8000/api/v1/upload`, { method: 'POST', body: formData });
      if (!upRes.ok) throw new Error("Upload failed");
      const upData = await upRes.json();
      const datasetName = upData.data.file_path;

      const params = new URLSearchParams({ dataset_name: datasetName });
      if (selectedDpoCheckpoint) params.append("lora_path", selectedDpoCheckpoint);

      addLog(`Running Post-DPO Evaluation on ${dpoEvalFile.name} with LLM Judge...`);
      const res = await fetch(`http://localhost:8000/api/v1/evaluate/dpo?${params.toString()}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setDpoEval(data.data);
        addLog(`Post-DPO Evaluation complete. ${data.data?.samples_processed || 0} samples graded.`);
        markComplete(8);
      } else { addLog(`[ERROR] Post-DPO Evaluation failed.`); }
    } catch (e) { addLog(`[ERROR] Network failure during DPO eval.`); }
    setLoading(false);
  };

  const handleComparison = async () => {
    setLoading(true);
    addLog(`Generating cross-run comparison (Base vs SFT vs DPO)...`);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/evaluate/compare`);
      if (res.ok) {
        const data = await res.json();
        setComparisonData(data.data);
        addLog(`Comparison complete! Results loaded.`);
        markComplete(9);
      } else { addLog(`[ERROR] Comparison failed.`); }
    } catch (e) { addLog(`[ERROR] Network failure during comparison.`); }
    setLoading(false);
  };

  const downloadCsv = (label: string, content: string) => {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${label}_eval.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };

  const handleDPOTraining = async () => {
    if (!dpoFile) { addLog("[ERROR] No DPO Dataset uploaded."); return; }
    setLoading(true);
    addLog(`Initiating Direct Preference Optimization (DPO) on ${dpoParams.model}...`);
    addLog(`Parameters: LR=${dpoParams.lr}, Epochs=${dpoParams.epochs}, Beta=${dpoParams.beta}`);

    try {
      const formData = new FormData();
      formData.append("file", dpoFile);
      formData.append("params", JSON.stringify(dpoParams));

      const res = await fetch(`http://localhost:8000/api/v1/train/dpo`, {
        method: 'POST',
        body: formData
      });
      const contentType = res.headers.get("content-type");
      if (res.ok && contentType && contentType.includes("application/zip")) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setDpoZipUrl(url);
        addLog(`DPO Training completed successfully. QLoRA weights generated and zipped.`);
        markComplete(7);
      } else {
        const errorData = await res.json().catch(() => ({ error: "Unknown error from server" }));
        addLog(`[ERROR] DPO Training Failed: ${errorData.error || "Check backend logs."}`);
      }
    } catch (e: any) {
      addLog(`[ERROR] Network failure during DPO training: ${e.message}`);
    }
    setLoading(false);
  };


  const stages = [
    { id: 1, name: "Upload", icon: <Upload size={12} /> },
    { id: 2, name: "Base Eval", icon: <Terminal size={12} /> },
    { id: 3, name: "SFT Data", icon: <Database size={12} /> },
    { id: 4, name: "DPO Data", icon: <Fingerprint size={12} /> },
    { id: 5, name: "Train SFT", icon: <Cpu size={12} /> },
    { id: 6, name: "SFT Eval", icon: <Terminal size={12} /> },
    { id: 7, name: "Train DPO", icon: <Cpu size={12} /> },
    { id: 8, name: "DPO Eval", icon: <Terminal size={12} /> },
    { id: 9, name: "Compare", icon: <Activity size={12} /> },
  ];

  const markComplete = (stage: number) => {
    if (!completedStages.includes(stage)) {
      setCompletedStages([...completedStages, stage]);
    }
    /* auto-navigation disabled */
  };


  // ── Reusable Eval Results Panel ──────────────────────────────────────────
  const EvalPanel = ({ evalResult, label, onRun, loading: isLoading }: {
    evalResult: any; label: string; onRun: () => void; loading: boolean;
  }) => {
    const rows: any[] = evalResult?.judge_results || evalResult?.rows || [];
    const agg = evalResult?.aggregate || {};
    const excelPath = evalResult?.csv_path || "";
    return (
      <div className="w-full flex flex-col gap-5">
        {rows.length === 0 ? (
          <button onClick={onRun} disabled={isLoading}
            className="bg-black text-white px-8 py-3 text-xs font-mono tracking-widest uppercase hover:bg-gray-800 transition-all flex items-center gap-2 disabled:opacity-50 w-fit">
            {isLoading ? <Loader2 size={14} className="animate-spin" /> : `Run ${label} Evaluation`} <ArrowRight size={14} />
          </button>
        ) : (
          <>
            {/* Stats Bar */}
            <div className="flex flex-wrap gap-4 border border-gray-200 p-4 bg-gray-50">
              {[
                { k: "Samples", v: agg.total_samples ?? rows.length },
                { k: "Correct", v: `${((agg.accuracy || 0) * 100).toFixed(1)}%` },
                { k: "Reasoning", v: `${((agg.avg_reasoning_score || 0) * 100).toFixed(0)}%` },
                { k: "Accuracy", v: `${((agg.avg_medical_accuracy || 0) * 100).toFixed(0)}%` },
                { k: "Guideline", v: `${((agg.avg_guideline_adherence || 0) * 100).toFixed(0)}%` },
                { k: "Completeness", v: `${((agg.avg_completeness || 0) * 100).toFixed(0)}%` },
                { k: "Unsafe", v: `${((agg.unsafe_rate || 0) * 100).toFixed(1)}%` },
                { k: "Hallucination", v: `${((agg.hallucination_rate || 0) * 100).toFixed(1)}%` },
                { k: "Consensus↑", v: `${agg.consensus_escalations ?? 0}` },
              ].map(({ k, v }) => (
                <div key={k} className="flex flex-col">
                  <span className="text-[9px] font-mono text-gray-400 uppercase tracking-widest">{k}</span>
                  <span className="text-lg font-mono font-bold text-black">{v}</span>
                </div>
              ))}
              <div className="ml-auto flex gap-2 items-start">
                {excelPath && <button onClick={() => window.open(`http://localhost:8000/api/v1/evaluate/download/${label.toLowerCase()}`, '_blank')}
                  className="border border-black text-black bg-white px-3 py-1.5 text-[10px] font-mono tracking-widest uppercase hover:bg-black hover:text-white transition-all flex items-center gap-1 shadow-sm">
                  <Database size={10} /> Download CSV
                </button>}
                <button onClick={() => markComplete(activeStage)}
                  className="bg-black text-white px-3 py-1.5 text-[10px] font-mono tracking-widest uppercase hover:bg-gray-800 transition-all flex items-center gap-1">
                  Next <ArrowRight size={10} />
                </button>
              </div>
            </div>
            {/* Table */}
            <div className="overflow-x-auto w-full border border-gray-200">
              <table className="w-full min-w-max text-xs font-mono">
                <thead className="bg-black text-white">
                  <tr>
                    {["#", "Question", "Ground Truth", "Prediction", "Correct", "Reasoning", "Accuracy", "Guideline", "Complete", "Unsafe", "Halluc.", "Consensus", "Explanation"].map(h => (
                      <th key={h} className="text-left p-2 text-[9px] tracking-widest uppercase font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {rows.map((r: any, i: number) => {
                    const isDangerous = r.unsafe || r.hallucination;
                    const isTriggered = r.consensus_triggered;
                    return (
                    <tr key={i} className={isTriggered ? 'bg-red-500 text-white' : isDangerous ? 'bg-red-600 text-white' : 'hover:bg-gray-50'}>
                      <td className="p-2 opacity-80">{i + 1}</td>
                      <td className={`p-2 max-w-[150px] truncate ${isDangerous || isTriggered ? 'text-white' : 'text-gray-600'}`} title={r.question}>{r.question}</td>
                      <td className={`p-2 max-w-[150px] truncate ${isDangerous || isTriggered ? 'text-white' : 'text-gray-600'}`} title={r.ground_truth}>{r.ground_truth}</td>
                      <td className={`p-2 max-w-[150px] truncate ${isDangerous || isTriggered ? 'text-white' : 'text-gray-600'}`} title={r.prediction}>{r.prediction}</td>
                      <td className="p-2"><span className={`px-1.5 py-0.5 text-[9px] font-bold ${r.correct ? (isDangerous || isTriggered ? 'bg-white text-red-700' : 'bg-black text-white') : (isDangerous || isTriggered ? 'border border-white text-white' : 'border border-black text-black')}`}>{r.correct ? '✓ YES' : '✗ NO'}</span></td>
                      {['reasoning_score', 'medical_accuracy', 'guideline_adherence', 'completeness'].map(k => (
                        <td key={k} className="p-2">
                          <span className={isDangerous || isTriggered ? 'text-white font-bold' : 'text-gray-700'}>{((r[k] || 0) * 100).toFixed(0)}%</span>
                        </td>
                      ))}
                      <td className="p-2">{r.unsafe ? <span className={`text-[9px] font-bold ${isDangerous || isTriggered ? 'text-white' : 'text-red-700'}`}>⚠ YES</span> : <span className={`text-[9px] ${isDangerous || isTriggered ? 'text-red-200' : 'text-gray-400'}`}>—</span>}</td>
                      <td className="p-2">{r.hallucination ? <span className={`text-[9px] font-bold ${isDangerous || isTriggered ? 'text-white' : 'text-orange-700'}`}>⚠ YES</span> : <span className={`text-[9px] ${isDangerous || isTriggered ? 'text-red-200' : 'text-gray-400'}`}>—</span>}</td>
                      <td className="p-2">{isTriggered ? <span className="bg-white text-red-600 px-1.5 py-0.5 text-[9px] font-bold">✓ ESCALATED</span> : <span className={`text-[9px] ${isDangerous || isTriggered ? 'text-red-200' : 'text-gray-400'}`}>—</span>}</td>
                      <td className={`p-2 max-w-sm truncate ${isDangerous || isTriggered ? 'text-white font-bold' : 'text-gray-600'}`} title={r.explanation}>{r.explanation}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col min-h-screen animate-in fade-in duration-700 max-w-6xl mx-auto p-4 md:p-8">
      <div className="border-b border-black pb-8 mb-8 flex flex-col gap-6">
        <div className="flex justify-between items-start w-full">
          <h1 className="text-3xl font-mono text-black tracking-tighter max-w-2xl leading-tight">Open-source Medical Model Evaluation & Post-Training Workbench</h1>
          <div className="text-[10px] font-mono bg-black text-white px-3 py-1 uppercase tracking-widest w-fit whitespace-nowrap">
            Open Source Platform
          </div>
        </div>

        <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest max-w-4xl leading-relaxed">
          A local-first platform for evaluating and improving open-weight medical language models. Built for AI researchers and engineers developing medical foundation models. MedAlign helps evaluate model behavior, investigate failures, generate supervision datasets, and validate improvements through reproducible benchmarks.
        </p>

        <div className="flex flex-wrap gap-8 mt-2">
          <div>
            <h3 className="text-[10px] font-bold text-black uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">Currently Supported</h3>
            <div className="flex gap-2">
              <span className="text-[9px] font-mono bg-black text-white px-2 py-1 uppercase tracking-widest">Qwen 2.5</span>
            </div>
          </div>

          <div>
            <h3 className="text-[10px] font-bold text-black uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">Planned Architectures</h3>
            <div className="flex flex-wrap gap-2">
              {["Llama", "Mistral / BioMistral", "Meditron", "PMC-LLaMA", "OpenBioLLM"].map(model => (
                <span key={model} className="text-[9px] font-mono bg-gray-100 text-gray-500 border border-gray-200 px-2 py-1 uppercase tracking-widest">{model}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Navigation - Circuit Board Style */}
      <div className="relative mb-12 w-full">
        {/* Background circuit line */}
        <div className="hidden md:block absolute top-5 left-0 w-full h-[1px] bg-gray-200 -z-10" />

        <div className="flex flex-wrap md:flex-nowrap justify-between gap-y-8 w-full">
          {stages.map((stage, index) => {
            const isActive = activeStage === stage.id;
            const isCompleted = completedStages.includes(stage.id);
            const isPast = isCompleted && !isActive;

            return (
              <div key={stage.id} className="flex flex-col items-center relative w-[20%] md:w-auto z-10 group cursor-pointer" onClick={() => setActiveStage(stage.id)}>

                {/* Circuit Node */}
                <div className={`w-10 h-10 rounded-sm flex items-center justify-center transition-all duration-300 border ${isActive ? 'bg-black border-black text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] -translate-y-1 -translate-x-1' :
                  isPast ? 'border-gray-800 bg-gray-200 text-gray-800' :
                    'bg-gray-100 border-gray-300 text-gray-500 hover:border-gray-500 hover:text-black'
                  }`}>
                  {isPast ? <CheckCircle2 size={14} className="text-black" /> : stage.icon}
                </div>

                {/* Vertical connection dot for mobile wrapping if needed */}
                <div className={`w-[1px] h-4 mt-2 hidden ${isActive ? 'bg-black' : isPast ? 'bg-gray-800' : 'bg-gray-300'} md:hidden`} />

                {/* Stage Label */}
                <div className="mt-3 flex flex-col items-center text-center">
                  <span className={`text-[9px] font-mono tracking-widest uppercase transition-all duration-300 ${isActive ? 'text-black font-bold' :
                    isPast ? 'text-gray-800' : 'text-gray-500'
                    }`}>
                    {stage.name}
                  </span>
                  <span className={`text-[8px] font-mono mt-1 ${isActive ? 'text-gray-500' : 'text-transparent'}`}>
                    0{stage.id}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex-1 flex flex-col gap-1 overflow-hidden">

        {/* Dynamic Content Area */}
        <div className="flex-1 bg-white border border-gray-200 shadow-sm flex flex-col relative transition-all duration-500 min-h-[500px]">

          {/* Stage 1: Upload */}
          {activeStage === 1 && (
            <div className="p-10 h-full flex flex-col items-start justify-center animate-in slide-in-from-right-4 duration-500">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight">01 // Dataset Ingestion</h2>
              <p className="text-sm text-gray-500 max-w-lg font-mono mb-10 leading-relaxed">
                Upload evaluation sets (MedQA, PubMedQA, Custom JSON) to act as the seed for our pipeline.
              </p>
              <div className="flex items-center gap-6 w-full max-w-lg">
                <input type="file" onChange={(e) => setUploadFile(e.target.files?.[0] || null)} className="flex-1 text-xs font-mono file:mr-4 file:py-2 file:px-4 file:border-0 file:bg-gray-100 file:text-black hover:file:bg-gray-200 cursor-pointer transition-colors" />
                <button onClick={handleUpload} disabled={loading} className="bg-black text-white px-6 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50 min-w-max">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : "Upload"} <ArrowRight size={14} />
                </button>
              </div>

              {completedStages.includes(1) && (
                <button onClick={() => setActiveStage(2)} className="mt-8 bg-white text-black border border-black px-6 py-2 text-[10px] tracking-widest font-mono hover:bg-black hover:text-white transition-all flex items-center gap-2 uppercase">
                  Next Stage <ArrowRight size={14} />
                </button>
              )}
            </div>
          )}

          {/* Stage 2: Base Eval */}
          {activeStage === 2 && (
            <div className="p-8 h-full flex flex-col animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <div className="w-12 h-12 bg-gray-100 text-black rounded-none flex items-center justify-center mb-5">
                <Terminal size={24} />
              </div>
              <h2 className="text-3xl font-mono text-black mb-2 tracking-tight">Base Model Evaluation</h2>
              <p className="text-sm text-gray-500 max-w-xl mb-6 leading-relaxed font-mono">
                Query <span className="font-semibold text-black underline">{modelName}</span> on every sample and grade each answer with the LLM Judge.
              </p>

              {judgeResults.length === 0 ? (
                <div className="flex flex-col gap-4 font-mono">
                  <div className="flex items-center gap-4 w-full max-w-md">
                    <select value={modelName} onChange={(e) => setModelName(e.target.value)}
                      className="flex-1 text-xs py-2.5 px-4 border border-gray-300 rounded-none text-gray-900 focus:outline-none focus:ring-1 focus:ring-black">
                      <option value="Qwen/Qwen2.5-0.5B">Qwen/Qwen2.5-0.5B</option>
                      <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen/Qwen2.5-0.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-1.5B">Qwen/Qwen2.5-1.5B</option>
                      <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen/Qwen2.5-1.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-3B">Qwen/Qwen2.5-3B</option>
                      <option value="Qwen/Qwen2.5-3B-Instruct">Qwen/Qwen2.5-3B-Instruct</option>
                      <option value="Qwen/Qwen2.5-7B-Instruct">Qwen/Qwen2.5-7B-Instruct</option>
                      <option value="meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct</option>
                      <option value="google/gemma-3-4b-it">google/gemma-3-4b-it</option>
                      <option value="mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3</option>

                    </select>
                  </div>
                  <button onClick={handleBaseEval} disabled={loading}
                    className="bg-black text-white px-8 py-3 text-xs tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50 w-fit">
                    {loading ? <Loader2 size={16} className="animate-spin" /> : "Run Base Evaluation"} <ArrowRight size={16} />
                  </button>
                </div>
              ) : (
                <div className="w-full flex flex-col gap-6 font-mono">
                  {/* Action Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-3 bg-gray-50 border border-gray-200 p-4">
                    <div className="flex gap-6 text-[10px] tracking-widest uppercase">
                      <span className="text-gray-600">Samples: <strong className="text-black">{judgeResults.length}</strong></span>
                      <span className="text-black">Correct: <strong>{judgeResults.filter((r: any) => r.correct).length}</strong></span>
                      <span className="text-red-700 font-bold">Unsafe: <strong>{judgeResults.filter((r: any) => r.unsafe).length}</strong></span>
                      <span className="text-orange-700 font-bold">Hallucinations: <strong>{judgeResults.filter((r: any) => r.hallucination).length}</strong></span>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={handleBaseEval} disabled={loading}
                        className="bg-black text-white px-4 py-2 text-[10px] tracking-widest uppercase hover:bg-gray-800 transition-all flex items-center gap-2 disabled:opacity-50">
                        {loading ? <Loader2 size={12} className="animate-spin" /> : "Re-Run"}
                      </button>
                      {evalData?.csv_path && (
                        <button onClick={() => {
                          window.open('http://localhost:8000/api/v1/evaluate/download/base', '_blank');
                        }} className="bg-white text-black border border-black px-4 py-2 text-[10px] tracking-widest uppercase hover:bg-black hover:text-white transition-all flex items-center gap-2 shadow-sm">
                          <Database size={12} /> Download CSV
                        </button>
                      )}
                      <button onClick={() => setActiveStage(3)}
                        className="bg-black text-white px-4 py-2 text-[10px] tracking-widest uppercase hover:bg-gray-800 transition-all flex items-center gap-2">
                        Next <ArrowRight size={12} />
                      </button>
                    </div>
                  </div>


                  {/* Aggregated Analytics */}
                  {evalData?.aggregate && (
                    <div className="w-full flex flex-col gap-6 font-mono mb-6">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
                        <div className="border border-gray-200 bg-white p-5">
                          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest mb-2">Accuracy</p>
                          <p className="text-3xl font-mono font-bold text-black">{((evalData.aggregate.accuracy || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="border border-gray-200 bg-white p-5">
                          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest mb-2">Hallucination Rate</p>
                          <p className="text-3xl font-mono font-bold text-red-600">{((evalData.aggregate.hallucination_rate || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="border border-gray-200 bg-white p-5">
                          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest mb-2">Unsafe Rate</p>
                          <p className="text-3xl font-mono font-bold text-red-600">{((evalData.aggregate.unsafe_rate || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="border border-gray-200 bg-white p-5">
                          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest mb-2">Reasoning Quality</p>
                          <p className="text-3xl font-mono font-bold text-black">{((evalData.aggregate.avg_reasoning_score || 0) * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Per-Sample Judge Results Table */}
                  <div className="w-full overflow-x-auto border border-gray-200">
                    <table className="w-full text-xs font-mono">
                      <thead className="bg-black text-white">
                        <tr>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Correct</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Reasoning</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Accuracy</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Guideline</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Complete</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Unsafe</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest">Halluc.</th>
                          <th className="text-left p-3 font-semibold uppercase tracking-widest min-w-[200px]">Explanation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {judgeResults.slice(0, 20).map((row: any, i: number) => {
                          const isDangerous = row.unsafe || row.hallucination;
                          return (
                          <tr key={i} className={isDangerous ? 'bg-red-600 text-white transition-colors' : 'hover:bg-gray-50 transition-colors'}>
                            <td className="p-3">
                              <span className={`px-2 py-0.5 text-[10px] font-bold ${row.correct ? (isDangerous ? 'bg-white text-red-700' : 'bg-black text-white') : (isDangerous ? 'border border-white text-white' : 'border border-black text-black')}`}>
                                {row.correct ? '✓ Yes' : '✗ No'}
                              </span>
                            </td>
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <div className={`w-16 h-1.5 overflow-hidden ${isDangerous ? 'bg-red-400' : 'bg-gray-200'}`}>
                                  <div className={`h-full ${isDangerous ? 'bg-white' : 'bg-black'}`} style={{ width: `${(row.reasoning_score || 0) * 100}%` }} />
                                </div>
                                <span className={isDangerous ? 'text-red-100' : 'text-gray-600'}>{((row.reasoning_score || 0) * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <div className={`w-16 h-1.5 overflow-hidden ${isDangerous ? 'bg-red-400' : 'bg-gray-200'}`}>
                                  <div className={`h-full ${isDangerous ? 'bg-white' : 'bg-gray-700'}`} style={{ width: `${(row.medical_accuracy || 0) * 100}%` }} />
                                </div>
                                <span className={isDangerous ? 'text-red-100' : 'text-gray-600'}>{((row.medical_accuracy || 0) * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <div className={`w-16 h-1.5 overflow-hidden ${isDangerous ? 'bg-red-400' : 'bg-gray-200'}`}>
                                  <div className={`h-full ${isDangerous ? 'bg-white' : 'bg-gray-600'}`} style={{ width: `${(row.guideline_adherence || 0) * 100}%` }} />
                                </div>
                                <span className={isDangerous ? 'text-red-100' : 'text-gray-600'}>{((row.guideline_adherence || 0) * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <div className={`w-16 h-1.5 overflow-hidden ${isDangerous ? 'bg-red-400' : 'bg-gray-200'}`}>
                                  <div className={`h-full ${isDangerous ? 'bg-white' : 'bg-gray-500'}`} style={{ width: `${(row.completeness || 0) * 100}%` }} />
                                </div>
                                <span className={isDangerous ? 'text-red-100' : 'text-gray-600'}>{((row.completeness || 0) * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="p-3">
                              {row.unsafe
                                ? <span className={`px-2 py-0.5 text-[10px] font-bold ${isDangerous ? 'bg-white text-red-700' : 'bg-red-100 text-red-700'}`}>⚠ Yes</span>
                                : <span className={`px-2 py-0.5 text-[10px] ${isDangerous ? 'text-red-200' : 'bg-gray-100 text-gray-500'}`}>No</span>}
                            </td>
                            <td className="p-3">
                              {row.hallucination
                                ? <span className={`px-2 py-0.5 text-[10px] font-bold ${isDangerous ? 'bg-yellow-200 text-yellow-900' : 'bg-orange-100 text-orange-700'}`}>⚠ Yes</span>
                                : <span className={`px-2 py-0.5 text-[10px] ${isDangerous ? 'text-red-200' : 'bg-gray-100 text-gray-500'}`}>No</span>}
                            </td>
                            <td className={`p-3 max-w-xs truncate ${isDangerous ? 'text-red-100' : 'text-gray-600'}`} title={row.explanation}>{row.explanation}</td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    {judgeResults.length > 20 && (
                      <div className="p-3 text-center text-xs text-gray-400 bg-gray-50 border-t border-gray-100">
                        Showing 20 of {judgeResults.length} results — download CSV for full data
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}



          {/* Stage 3: SFT Data Gen */}
          {activeStage === 3 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight flex items-center gap-4">
                03 // Generate SFT Dataset
                {completedStages.includes(3) && (
                  <button onClick={() => setActiveStage(4)} className="bg-black text-white px-4 py-2 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all uppercase">
                    Next <ArrowRight size={14} className="inline ml-1" />
                  </button>
                )}
              </h2>
              <p className="text-sm text-gray-500 max-w-lg font-mono mb-6 leading-relaxed">
                Use Ollama to automatically generate 100-200 similar Instruction/Output pairs to Supervised Fine-Tune the model, specifically targeting and varying based on the failures discovered in Stage 2.
              </p>

              {!completedStages.includes(3) ? (
                <button onClick={handleSFTDataGen} disabled={loading} className="bg-black text-white px-6 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50 mb-10">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : "Generate SFT Pairs"} <ArrowRight size={14} />
                </button>
              ) : null}

              {syntheticData?.length > 0 && (
                <div className="w-full flex flex-col gap-4 pb-12 font-mono">
                  <div className="flex justify-between items-center bg-gray-50 p-4 border border-gray-200">
                    <span className="text-[10px] font-mono text-gray-600 uppercase tracking-widest font-bold">Successfully generated {syntheticData.length} SFT pairs.</span>
                    <button onClick={handleDownloadSynthetic} className="bg-white text-black border border-black px-4 py-2 text-[10px] tracking-widest font-mono hover:bg-gray-100 transition-all uppercase flex items-center gap-2">
                      <Database size={12} /> Download SFT Dataset
                    </button>
                  </div>

                  {syntheticData.map((item: any, idx: number) => (
                    <div key={idx} className="border border-gray-200 p-4 bg-white">
                      <p className="text-[10px] font-bold font-mono text-black uppercase mb-1 border-b border-gray-200 pb-1">Instruction</p>
                      <p className="text-xs font-mono text-gray-800 mb-4 bg-gray-50 p-2">{item.instruction}</p>

                      <p className="text-[10px] font-bold font-mono text-black uppercase mb-1 border-b border-gray-200 pb-1">Output</p>
                      <p className="text-xs font-mono text-gray-800 bg-gray-50 p-2">{item.output}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {/* Stage 4: DPO Data Gen */}
          {activeStage === 4 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <div className="w-12 h-12 bg-gray-100 text-black rounded-none flex items-center justify-center mb-6">
                <Database size={24} />
              </div>
              <h2 className="text-3xl font-mono text-black mb-4 tracking-tight flex items-center gap-4">
                04 // Convert DPO Pair Dataset
                {completedStages.includes(4) && (
                  <button onClick={() => setActiveStage(5)} className="bg-black text-white px-4 py-2 text-[10px] font-mono tracking-widest uppercase hover:bg-gray-800 transition-all flex items-center">
                    Next <ArrowRight size={14} className="ml-2" />
                  </button>
                )}
              </h2>
              <p className="text-sm text-gray-500 max-w-xl mb-8 leading-relaxed font-mono">
                Upload the SFT Data you downloaded in Stage 3 to convert it into a DPO Preference Dataset (Chosen vs Rejected).
              </p>

              {!completedStages.includes(4) && !evalData?.preference_dataset?.length ? (
                <div className="flex flex-col gap-4">
                  <input type="file" accept=".json" onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    setLoading(true);
                    try {
                      const formData = new FormData();
                      formData.append("file", file);
                      const res = await fetch(`http://localhost:8000/api/v1/datasets/dpo-convert`, {
                        method: 'POST',
                        body: formData
                      });
                      const data = await res.json();
                      if (data.data?.preference_dataset && data.data.preference_dataset.length > 0) {
                        setEvalData((prev: any) => ({ ...prev, preference_dataset: data.data.preference_dataset }));
                        if (!completedStages.includes(4)) setCompletedStages(prev => [...prev, 4]);
                        addLog(`[LLM] Successfully generated ${data.data.preference_dataset.length} DPO pairs.`);
                      } else {
                        addLog(`[ERROR] DPO Conversion Failed: ${data.message || "No valid DPO pairs generated."}`);
                      }
                    } catch (error: any) {
                      console.error("DPO Conversion Error:", error);
                      addLog(`[ERROR] DPO Conversion Error: ${error.message}`);
                    }
                    setLoading(false);
                  }} className="text-xs font-mono file:bg-black file:text-white file:border-none file:px-4 file:py-2 file:mr-4 file:cursor-pointer file:uppercase file:tracking-widest cursor-pointer" />
                  {loading && (
                    <div className="w-full mt-4 mb-4 flex flex-col gap-2 max-w-md">
                      <div className="flex justify-between text-[10px] font-mono text-gray-500 font-bold uppercase tracking-widest">
                        <span className="flex items-center gap-2"><Loader2 size={12} className="animate-spin" /> Processing DPO Pairs</span>
                      </div>
                      <div className="w-full bg-gray-200 h-1.5 overflow-hidden">
                        <div className="bg-black h-full w-full animate-pulse"></div>
                      </div>
                    </div>
                  )}
                </div>
              ) : null}

              {evalData?.preference_dataset?.length > 0 ? (
                <div className="w-full grid gap-4 pb-12 font-mono mt-8">
                  <div className="flex justify-between items-center bg-gray-50 p-4 border border-gray-200 mb-2">
                    <span className="text-[10px] font-semibold text-black uppercase tracking-widest">Converted {evalData.preference_dataset.length} DPO pairs.</span>
                    <button onClick={() => {
                      const blob = new Blob([JSON.stringify(evalData.preference_dataset, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = 'dpo_dataset.json';
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                    }} className="bg-white text-black border border-black px-5 py-2 text-[10px] font-mono tracking-widest uppercase hover:bg-black hover:text-white transition-all flex items-center gap-2 shadow-sm">
                      <Database size={14} /> Download DPO Dataset
                    </button>
                  </div>
                  {evalData.preference_dataset.slice(0, 3).map((pref: any, idx: number) => (
                    <div key={idx} className="border border-gray-200 p-5 bg-white">
                      <p className="text-[10px] font-bold text-black uppercase mb-2">Prompt:</p>
                      <p className="text-xs font-mono text-gray-800 mb-4 bg-gray-50 p-3">{pref.prompt}</p>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-[10px] font-bold text-black uppercase mb-2">Chosen</p>
                          <p className="text-xs font-mono text-gray-600 bg-gray-50 p-3 line-clamp-3">{pref.chosen}</p>
                        </div>
                        <div>
                          <p className="text-[10px] font-bold text-red-700 uppercase mb-2">Rejected</p>
                          <p className="text-xs font-mono text-gray-600 bg-red-50 p-3 line-clamp-3">{pref.rejected}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}

          {/* Stage 5: Train SFT */}
          {activeStage === 5 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight">05 // Train SFT</h2>
              <p className="text-sm text-gray-500 font-mono mb-6 leading-relaxed">
                Upload your SFT dataset and configure QLoRA parameters to train the base model.
              </p>

              <div className="w-full max-w-2xl bg-gray-50 border border-gray-200 p-6 flex flex-col gap-6 font-mono mb-6">
                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-2">SFT Dataset (.json)</label>
                  <input type="file" onChange={(e) => setSftFile(e.target.files?.[0] || null)} className="text-xs w-full file:bg-black file:text-white file:border-none file:px-4 file:py-2 file:cursor-pointer" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-1">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Run Name (Folder/Zip Name)</label>
                    <input type="text" placeholder="e.g. v1-experimental" value={sftParams.run_name} onChange={e => setSftParams({ ...sftParams, run_name: e.target.value })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div className="col-span-1">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Base Model ID</label>
                    <select value={sftParams.model} onChange={e => setSftParams({ ...sftParams, model: e.target.value })} className="w-full border border-gray-300 p-2 text-xs">
                      <option value="Qwen/Qwen2.5-0.5B">Qwen/Qwen2.5-0.5B</option>
                      <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen/Qwen2.5-0.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-1.5B">Qwen/Qwen2.5-1.5B</option>
                      <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen/Qwen2.5-1.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-3B">Qwen/Qwen2.5-3B</option>
                      <option value="Qwen/Qwen2.5-3B-Instruct">Qwen/Qwen2.5-3B-Instruct</option>
                      <option value="Qwen/Qwen2.5-7B-Instruct">Qwen/Qwen2.5-7B-Instruct</option>
                      <option value="meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct</option>
                      <option value="google/gemma-3-4b-it">google/gemma-3-4b-it</option>
                      <option value="mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3</option>
                      <option value="axiong/PMC_LLaMA_13B">axiong/PMC_LLaMA_13B</option>
                      <option value="aaditya/Llama3-OpenBioLLM-70B">aaditya/Llama3-OpenBioLLM-70B</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Learning Rate</label>
                    <input type="text" value={sftParams.lr} onChange={e => setSftParams({ ...sftParams, lr: e.target.value })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Epochs</label>
                    <input type="number" value={sftParams.epochs} onChange={e => setSftParams({ ...sftParams, epochs: parseInt(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Batch Size</label>
                    <input type="number" value={sftParams.batch_size} onChange={e => setSftParams({ ...sftParams, batch_size: parseInt(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">LoRA Rank (r)</label>
                    <input type="number" value={sftParams.r} onChange={e => setSftParams({ ...sftParams, r: parseInt(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                </div>
              </div>

              <div className="flex gap-4 items-center">
                <button onClick={handleTrainSFT} disabled={loading || !sftFile} className="bg-black text-white px-8 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : "Start SFT Training"}
                </button>

                {sftZipUrl && (
                  <a href={sftZipUrl} download={sftParams.run_name ? `sft_output_${sftParams.run_name}.zip` : "sft_output.zip"} className="border border-black text-black bg-white px-6 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-100 transition-all flex items-center gap-2 uppercase">
                    <Database size={14} /> Download QLoRA Zip
                  </a>
                )}
              </div>

              {completedStages.includes(5) && (
                <button onClick={() => setActiveStage(6)} className="mt-8 bg-white text-black border border-black px-6 py-2 text-[10px] tracking-widest font-mono hover:bg-black hover:text-white transition-all flex items-center gap-2 uppercase">
                  Next Stage <ArrowRight size={14} />
                </button>
              )}
            </div>
          )}

          {/* Stage 6: SFT Evaluation */}
          {activeStage === 6 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight">06 // Post-SFT Evaluation</h2>
              <p className="text-sm text-gray-500 font-mono mb-6 leading-relaxed">
                Upload your test data and evaluate the SFT-trained model with LLM Judge. Compare with Base Eval to measure improvement.
              </p>

              <div className="w-full max-w-2xl bg-gray-50 border border-gray-200 p-6 flex flex-col gap-6 font-mono mb-6">
                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-2">Eval Dataset (.json)</label>
                  <input type="file" onChange={(e) => setSftEvalFile(e.target.files?.[0] || null)} className="text-xs w-full file:bg-black file:text-white file:border-none file:px-4 file:py-2 file:cursor-pointer" />
                </div>
                <div className="border-t border-gray-200 pt-6">
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black">Select SFT Checkpoint</label>
                    <button onClick={() => fetchCheckpoints('sft')} className="text-[10px] tracking-widest uppercase text-blue-600 hover:underline">Refresh</button>
                  </div>
                  <div className="flex gap-2">
                    <select value={selectedSftCheckpoint} onChange={e => setSelectedSftCheckpoint(e.target.value)} className="w-full border border-gray-300 p-2 text-xs">
                      <option value="">-- Default (/tmp/sft_output) --</option>
                      {sftCheckpoints.map(cp => (
                        <option key={cp.path} value={cp.path}>{cp.name} ({new Date(cp.created_at).toLocaleString()} - {cp.size_mb} MB)</option>
                      ))}
                    </select>
                    {selectedSftCheckpoint && (
                      <button onClick={() => deleteCheckpoint('sft', selectedSftCheckpoint.split('/').pop()!)} className="bg-red-50 text-red-600 border border-red-200 px-4 text-[10px] tracking-widest uppercase hover:bg-red-100 transition-colors">Delete</button>
                    )}
                  </div>
                </div>
              </div>

              <EvalPanel evalResult={sftEval} label="SFT" onRun={handleSFTEval} loading={loading} />

              {completedStages.includes(6) && (
                <button onClick={() => setActiveStage(7)} className="mt-8 bg-white text-black border border-black px-6 py-2 text-[10px] tracking-widest font-mono hover:bg-black hover:text-white transition-all flex items-center gap-2 uppercase">
                  Next Stage <ArrowRight size={14} />
                </button>
              )}
            </div>
          )}

          {/* Stage 7: Train DPO */}
          {activeStage === 7 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight">07 // Train DPO</h2>
              <p className="text-sm text-gray-500 font-mono mb-6 leading-relaxed">
                Upload your DPO Chosen/Rejected pairs and configure Direct Preference Optimization parameters.
              </p>

              <div className="w-full max-w-2xl bg-gray-50 border border-gray-200 p-6 flex flex-col gap-6 font-mono mb-6">
                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-2">DPO Dataset (.json)</label>
                  <input type="file" onChange={(e) => setDpoFile(e.target.files?.[0] || null)} className="text-xs w-full file:bg-black file:text-white file:border-none file:px-4 file:py-2 file:cursor-pointer" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-1">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Run Name (Folder/Zip Name)</label>
                    <input type="text" placeholder="e.g. dpo-run-1" value={dpoParams.run_name} onChange={e => setDpoParams({ ...dpoParams, run_name: e.target.value })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div className="col-span-1">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Base Model (SFT)</label>
                    <select value={dpoParams.model} onChange={e => setDpoParams({ ...dpoParams, model: e.target.value })} className="w-full border border-gray-300 p-2 text-xs">
                      <option value="local-sft-model">local-sft-model</option>
                      <option value="Qwen/Qwen2.5-0.5B">Qwen/Qwen2.5-0.5B</option>
                      <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen/Qwen2.5-0.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-1.5B">Qwen/Qwen2.5-1.5B</option>
                      <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen/Qwen2.5-1.5B-Instruct</option>
                      <option value="Qwen/Qwen2.5-3B">Qwen/Qwen2.5-3B</option>
                      <option value="Qwen/Qwen2.5-3B-Instruct">Qwen/Qwen2.5-3B-Instruct</option>
                      <option value="Qwen/Qwen2.5-7B-Instruct">Qwen/Qwen2.5-7B-Instruct</option>
                      <option value="meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct</option>
                      <option value="google/gemma-3-4b-it">google/gemma-3-4b-it</option>
                      <option value="mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3</option>
                      <option value="axiong/PMC_LLaMA_13B">axiong/PMC_LLaMA_13B</option>
                      <option value="aaditya/Llama3-OpenBioLLM-70B">aaditya/Llama3-OpenBioLLM-70B</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Learning Rate</label>
                    <input type="text" value={dpoParams.lr} onChange={e => setDpoParams({ ...dpoParams, lr: e.target.value })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Epochs</label>
                    <input type="number" value={dpoParams.epochs} onChange={e => setDpoParams({ ...dpoParams, epochs: parseInt(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Batch Size</label>
                    <input type="number" value={dpoParams.batch_size} onChange={e => setDpoParams({ ...dpoParams, batch_size: parseInt(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-1">Beta (KL Penalty)</label>
                    <input type="number" step="0.1" value={dpoParams.beta} onChange={e => setDpoParams({ ...dpoParams, beta: parseFloat(e.target.value) })} className="w-full border border-gray-300 p-2 text-xs" />
                  </div>
                </div>
              </div>

              <div className="flex gap-4 items-center">
                <button onClick={handleDPOTraining} disabled={loading || !dpoFile} className="bg-black text-white px-8 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : "Start DPO Training"}
                </button>

                {dpoZipUrl && (
                  <a href={dpoZipUrl} download={dpoParams.run_name ? `dpo_output_${dpoParams.run_name}.zip` : "dpo_output.zip"} className="border border-black text-black bg-white px-6 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-100 transition-all flex items-center gap-2 uppercase">
                    <Database size={14} /> Download QLoRA Zip
                  </a>
                )}
              </div>

              {completedStages.includes(7) && (
                <button onClick={() => setActiveStage(8)} className="mt-8 bg-white text-black border border-black px-6 py-2 text-[10px] tracking-widest font-mono hover:bg-black hover:text-white transition-all flex items-center gap-2 uppercase">
                  Next Stage <ArrowRight size={14} />
                </button>
              )}
            </div>
          )}

          {/* Stage 8: DPO Evaluation */}
          {activeStage === 8 && (
            <div className="p-10 h-full flex flex-col items-start animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-4 tracking-tight">08 // Post-DPO Evaluation</h2>
              <p className="text-sm text-gray-500 font-mono mb-6 leading-relaxed">
                Upload your test data and evaluate the DPO-aligned model. Final proof of alignment improvement over base and SFT.
              </p>

              <div className="w-full max-w-2xl bg-gray-50 border border-gray-200 p-6 flex flex-col gap-6 font-mono mb-6">
                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-black mb-2">Eval Dataset (.json)</label>
                  <input type="file" onChange={(e) => setDpoEvalFile(e.target.files?.[0] || null)} className="text-xs w-full file:bg-black file:text-white file:border-none file:px-4 file:py-2 file:cursor-pointer" />
                </div>
                <div className="border-t border-gray-200 pt-6">
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-black">Select DPO Checkpoint</label>
                    <button onClick={() => fetchCheckpoints('dpo')} className="text-[10px] tracking-widest uppercase text-blue-600 hover:underline">Refresh</button>
                  </div>
                  <div className="flex gap-2">
                    <select value={selectedDpoCheckpoint} onChange={e => setSelectedDpoCheckpoint(e.target.value)} className="w-full border border-gray-300 p-2 text-xs">
                      <option value="">-- Default (/tmp/dpo_output) --</option>
                      {dpoCheckpoints.map(cp => (
                        <option key={cp.path} value={cp.path}>{cp.name} ({new Date(cp.created_at).toLocaleString()} - {cp.size_mb} MB)</option>
                      ))}
                    </select>
                    {selectedDpoCheckpoint && (
                      <button onClick={() => deleteCheckpoint('dpo', selectedDpoCheckpoint.split('/').pop()!)} className="bg-red-50 text-red-600 border border-red-200 px-4 text-[10px] tracking-widest uppercase hover:bg-red-100 transition-colors">Delete</button>
                    )}
                  </div>
                </div>
              </div>

              <EvalPanel evalResult={dpoEval} label="DPO" onRun={handleDPOEval} loading={loading} />

              {completedStages.includes(8) && (
                <button onClick={() => setActiveStage(9)} className="mt-8 bg-white text-black border border-black px-6 py-2 text-[10px] tracking-widest font-mono hover:bg-black hover:text-white transition-all flex items-center gap-2 uppercase">
                  Next Stage <ArrowRight size={14} />
                </button>
              )}
            </div>
          )}

          {/* Stage 9: Compare All */}
          {activeStage === 9 && (
            <div className="p-8 h-full flex flex-col animate-in slide-in-from-right-4 duration-500 overflow-y-auto w-full">
              <h2 className="text-2xl font-mono text-black mb-2 tracking-tight">09 // Cross-Run Comparison</h2>
              <p className="text-sm text-gray-500 font-mono mb-6 leading-relaxed max-w-lg">
                Side-by-side comparison of Base vs SFT vs DPO. All CSVs downloadable for reproducibility.
              </p>
              {!comparisonData ? (
                <button onClick={handleComparison} disabled={loading} className="bg-black text-white px-8 py-3 text-[10px] tracking-widest font-mono hover:bg-gray-800 transition-all flex items-center gap-2 uppercase disabled:opacity-50 w-fit">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : "Generate Comparison"} <ArrowRight size={14} />
                </button>
              ) : (
                <div className="w-full flex flex-col gap-6">
                  <div className="overflow-x-auto border border-gray-200">
                    <table className="w-full text-xs font-mono">
                      <thead className="bg-black text-white">
                        <tr>
                          {["Metric", "Base", "SFT", "Δ SFT", "DPO", "Δ DPO"].map(h => (
                            <th key={h} className="text-left p-3 text-[9px] uppercase tracking-widest">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {[
                          { key: "accuracy", label: "Accuracy", pct: true },
                          { key: "hallucination_rate", label: "Hallucination Rate", pct: true },
                          { key: "unsafe_rate", label: "Unsafe Rate", pct: true },
                          { key: "avg_reasoning_score", label: "Avg Reasoning", pct: false },
                          { key: "avg_medical_accuracy", label: "Avg Med Accuracy", pct: false },
                          { key: "avg_guideline_adherence", label: "Avg Guideline", pct: false },
                        ].map(({ key, label, pct }) => {
                          const base = comparisonData?.base;
                          const sft = comparisonData?.sft;
                          const dpo = comparisonData?.dpo;
                          const sftDelta = comparisonData?.sft_vs_base;
                          const dpoDelta = comparisonData?.dpo_vs_base;
                          const fmt = (v: any) => v != null ? (pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)) : "—";
                          const fmtDelta = (d: any, k: string) => {
                            if (!d) return "—";
                            const deltaKey = k === "accuracy" ? "accuracy_delta" : k === "hallucination_rate" ? "hallucination_delta" : k === "unsafe_rate" ? "unsafe_delta" : "reasoning_delta";
                            const val = d[deltaKey];
                            if (val == null) return "—";
                            const sign = val > 0 ? "+" : "";
                            const isGood = (k === "accuracy" || k === "avg_reasoning_score" || k === "avg_medical_accuracy" || k === "avg_guideline_adherence") ? val > 0 : val < 0;
                            return <span className={isGood ? "text-green-700 font-bold" : "text-red-700 font-bold"}>{sign}{val.toFixed(1)}{pct ? "%" : ""}</span>;
                          };
                          return (
                            <tr key={key} className="hover:bg-gray-50">
                              <td className="p-3 font-semibold text-gray-700">{label}</td>
                              <td className="p-3 text-gray-600">{base ? fmt(base[key]) : "—"}</td>
                              <td className="p-3 text-gray-600">{sft ? fmt(sft[key]) : "—"}</td>
                              <td className="p-3">{fmtDelta(sftDelta, key)}</td>
                              <td className="p-3 text-gray-600">{dpo ? fmt(dpo[key]) : "—"}</td>
                              <td className="p-3">{fmtDelta(dpoDelta, key)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex gap-2 flex-wrap">
                      {["base", "sft", "dpo"].map(l => (
                        <button key={l} onClick={() => window.open(`http://localhost:8000/api/v1/evaluate/download/${l}`, '_blank')}
                          className="border border-black text-black px-4 py-2 text-[10px] font-mono tracking-widest uppercase hover:bg-black hover:text-white transition-all flex items-center gap-1">
                          <Database size={10} /> {l.toUpperCase()} CSV
                        </button>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

        </div>

        {/* Integrated Output Console */}
        <div className="w-full h-[250px] z-40 mt-2 bg-black/95 backdrop-blur-md text-green-400 p-4 overflow-y-auto font-mono text-[10px] tracking-widest flex flex-col gap-1 border border-gray-800 shadow-2xl rounded-xl">
          <div className="flex items-center gap-2 mb-2 text-gray-500 uppercase pb-2 border-b border-gray-800">
            <Terminal size={12} />
            <span>PIPELINE_STDOUT</span>
          </div>

          {loading && progress > 0 && (
            <div className="w-full mt-2 mb-4 flex flex-col gap-1">
              <div className="flex justify-between text-[9px] text-gray-500">
                <span>PROCESSING STAGE 0{activeStage}</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-900 h-1.5 rounded-full overflow-hidden">
                <div className="bg-white h-full transition-all duration-300 ease-out" style={{ width: `${progress}%` }}></div>
              </div>
            </div>
          )}

          {logs.map((log, i) => (
            <div key={i} className="break-all">{log}</div>

          ))}
          {loading && (
            <div className="flex items-center gap-2 mt-2 text-yellow-500 animate-pulse">
              <Loader2 size={10} className="animate-spin" />
              <span>Executing Model...</span>
            </div>
          )}
        </div>
      </div>

      {errorAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm transition-opacity duration-300">
          <div className="bg-white border-2 border-red-600 p-6 max-w-sm w-full shadow-2xl animate-in zoom-in-95 duration-200">
            <h3 className="text-red-700 font-mono font-bold uppercase tracking-widest text-sm mb-2 flex items-center gap-2">
              <span className="bg-red-600 text-white px-1 py-0.5 text-[10px]">Error</span>
              Action Required
            </h3>
            <p className="text-gray-800 font-mono text-xs leading-relaxed mb-6">
              {errorAlert}
            </p>
            <button 
              onClick={() => setErrorAlert(null)}
              className="w-full bg-black text-white py-2 text-[10px] font-mono tracking-widest uppercase hover:bg-gray-800 transition-colors"
            >
              Acknowledge
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
