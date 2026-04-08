/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { State, Action, ActionType, MachineStatus } from "./models.ts";

export class BaselineAgent {
  public static getAction(state: State): Action {
    // 1. Prioritize maintenance for broken machines
    const brokenMachine = state.machines.find(m => m.status === MachineStatus.BROKEN);
    if (brokenMachine) {
      return { type: ActionType.MAINTENANCE, machine_id: brokenMachine.id };
    }

    // 2. Proactive maintenance for low health machines (only if idle)
    const lowHealthMachine = state.machines.find(
      m => m.health < 0.4 && m.status === MachineStatus.IDLE
    );
    if (lowHealthMachine) {
      return { type: ActionType.MAINTENANCE, machine_id: lowHealthMachine.id };
    }

    // 3. Assign jobs to idle machines
    const idleMachine = state.machines.find(m => m.status === MachineStatus.IDLE);
    if (idleMachine && state.job_queue.length > 0) {
      // Priority-aware: pick highest priority job, then longest processing time
      const sortedJobs = [...state.job_queue].sort((a, b) => {
        if (b.priority !== a.priority) return b.priority - a.priority;
        return b.processing_time - a.processing_time;
      });
      
      const bestJob = sortedJobs[0];
      
      // Check energy constraint before assigning
      const estimatedNewEnergy = state.current_energy_usage + (10.0 + bestJob.priority * 2.0);
      if (estimatedNewEnergy <= state.energy_budget * 1.1) { // Allow slight overage for high priority
        return { 
          type: ActionType.ASSIGN, 
          job_id: bestJob.id, 
          machine_id: idleMachine.id 
        };
      }
    }

    return { type: ActionType.DO_NOTHING };
  }
}
