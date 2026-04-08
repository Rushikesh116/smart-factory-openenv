/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { State, MachineStatus } from "./models.ts";

export class SmartFactoryGrader {
  public static grade(state: State): number {
    if (!state || state.time_step === 0) return 0.0;

    // 1. Throughput Score (0.0 to 1.0)
    // Normalized by max possible jobs (estimated)
    const totalJobs = state.jobs_completed + state.job_queue.length;
    const throughputScore = totalJobs > 0 ? state.jobs_completed / totalJobs : 1.0;

    // 2. Machine Health Score (0.0 to 1.0)
    const avgHealth = state.machines.length > 0 
      ? state.machines.reduce((acc, m) => acc + m.health, 0) / state.machines.length 
      : 1.0;

    // 3. Energy Efficiency Score (0.0 to 1.0)
    // Penalty scales with how much the budget was exceeded
    let energyScore = 1.0;
    if (state.current_energy_usage > state.energy_budget) {
      const overageRatio = (state.current_energy_usage - state.energy_budget) / state.energy_budget;
      energyScore = Math.max(0, 1.0 - overageRatio);
    }

    // 4. Uptime Score (0.0 to 1.0)
    const brokenCount = state.machines.filter(m => m.status === MachineStatus.BROKEN).length;
    const uptimeScore = state.machines.length > 0 
      ? 1.0 - (brokenCount / state.machines.length) 
      : 1.0;

    // Weighted final score
    const finalScore = (throughputScore * 0.4) + (avgHealth * 0.2) + (energyScore * 0.2) + (uptimeScore * 0.2);
    
    return Math.min(1.0, Math.max(0.0, finalScore));
  }

  public static getDetailedMetrics(state: State) {
    return {
      throughput: state.jobs_completed / (state.time_step || 1),
      completion_rate: state.jobs_completed / (state.jobs_completed + state.job_queue.length || 1),
      avg_machine_health: state.machines.reduce((acc, m) => acc + m.health, 0) / (state.machines.length || 1),
      energy_efficiency: state.current_energy_usage <= state.energy_budget ? 1.0 : 0.0,
      broken_machines: state.machines.filter(m => m.status === MachineStatus.BROKEN).length,
      total_reward: state.total_reward
    };
  }
}
