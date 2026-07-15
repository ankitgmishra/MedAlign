import React from 'react';
import { Loader2, ArrowRight, Database } from 'lucide-react';

interface EvalPanelProps {
  evalResult: any;
  label: string;
  onRun: () => void;
  loading: boolean;
  onNext: () => void;
}

export const EvalPanel = ({ evalResult, label, onRun, loading: isLoading, onNext }: EvalPanelProps) => {
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
              <button onClick={onNext}
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
