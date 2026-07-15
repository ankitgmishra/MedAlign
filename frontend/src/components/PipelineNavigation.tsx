import React from 'react';
import { Upload, Cpu, Database, Terminal, Activity, CheckCircle2, Fingerprint } from 'lucide-react';

const STAGES = [
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

interface PipelineNavigationProps {
  activeStage: number;
  completedStages: number[];
  setActiveStage: (id: number) => void;
}

export const PipelineNavigation = ({ activeStage, completedStages, setActiveStage }: PipelineNavigationProps) => {
  return (
    <div className="relative mb-12 w-full">
      {/* Background circuit line */}
      <div className="hidden md:block absolute top-5 left-0 w-full h-[1px] bg-gray-200 -z-10" />

      <div className="flex flex-wrap md:flex-nowrap justify-between gap-y-8 w-full">
        {STAGES.map((stage) => {
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
  );
};
