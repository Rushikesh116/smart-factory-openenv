/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export enum MachineStatus {
  IDLE = "idle",
  BUSY = "busy",
  BROKEN = "broken",
  MAINTENANCE = "maintenance",
  INTERMITTENT_FAILURE = "intermittent_failure",
}

export interface Machine {
  id: string;
  status: MachineStatus;
  health: number; // 0.0 to 1.0
  energy_usage: number; // current energy consumption
  current_job_id: string | null;
  current_job_priority?: number;
  remaining_time: number;
}

export interface Job {
  id: string;
  processing_time: number;
  priority: number; // 1 (low) to 3 (high)
  arrival_time: number;
  waiting_time: number;
}

export interface State {
  machines: Machine[];
  job_queue: Job[];
  energy_budget: number;
  current_energy_usage: number;
  throughput?: number;
  delay?: number;
  queue_length?: number;
  breakdown_risk?: number;
  time_step: number;
  total_reward: number;
  jobs_completed: number;
  max_steps: number;
  done: boolean;
}

export enum ActionType {
  ASSIGN = "assign",
  DELAY = "delay",
  MAINTENANCE = "maintenance",
  DO_NOTHING = "do_nothing",
}

export interface Action {
  type: ActionType;
  job_id?: string;
  machine_id?: string;
}

export interface StepResult {
  state: State;
  reward: number;
  done: boolean;
  info: any;
}

export interface Task {
  id: string;
  name: string;
  description: string;
  objective?: string;
  grader: string;
  config: {
    num_machines: number;
    initial_jobs: number;
    energy_budget: number;
    failure_rate: number;
    max_steps: number;
    intermittent_failure_rate?: number;
    degradation_rate?: number;
    high_intensity_multiplier?: number;
  };
}
