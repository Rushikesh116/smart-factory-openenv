/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Machine, MachineStatus, Job, State, Action, ActionType, StepResult } from "./models.ts";

export class SmartFactorySimulator {
  private state: State;
  private next_job_id: number = 0;
  private config: any;

  constructor(config: any) {
    this.config = config;
    this.state = this.reset();
  }

  public reset(): State {
    const machines: Machine[] = [];
    for (let i = 0; i < this.config.num_machines; i++) {
      machines.push({
        id: `machine_${i}`,
        status: MachineStatus.IDLE,
        health: 1.0,
        energy_usage: 0.0,
        current_job_id: null,
        remaining_time: 0,
      });
    }

    const job_queue: Job[] = [];
    this.next_job_id = 0;
    for (let i = 0; i < this.config.initial_jobs; i++) {
      job_queue.push(this.generateJob(0));
    }

    this.state = {
      machines,
      job_queue,
      energy_budget: this.config.energy_budget,
      current_energy_usage: 0.0,
      time_step: 0,
      total_reward: 0.0,
      jobs_completed: 0,
      max_steps: this.config.max_steps,
      done: false,
    };

    return this.state;
  }

  private generateJob(time_step: number): Job {
    const id = `job_${this.next_job_id++}`;
    const processing_time = Math.floor(Math.random() * 5) + 2;
    const priority = Math.floor(Math.random() * 3) + 1;
    return {
      id,
      processing_time,
      priority,
      arrival_time: time_step,
      waiting_time: 0,
    };
  }

  public step(action: Action): StepResult {
    if (this.state.done) {
      return { state: this.state, reward: 0, done: true, info: {} };
    }

    let step_reward = 0;

    // Configuration defaults for new failure modes
    const intermittent_failure_rate = this.config.intermittent_failure_rate || 0.02;
    const base_degradation_rate = this.config.degradation_rate || 0.005;
    const high_intensity_multiplier = this.config.high_intensity_multiplier || 2.0;

    // 1. Process Action
    if (action.type === ActionType.ASSIGN && action.job_id && action.machine_id) {
      const machine = this.state.machines.find((m) => m.id === action.machine_id);
      const jobIndex = this.state.job_queue.findIndex((j) => j.id === action.job_id);

      if (machine && machine.status === MachineStatus.IDLE && jobIndex !== -1) {
        const job = this.state.job_queue[jobIndex];
        machine.status = MachineStatus.BUSY;
        machine.current_job_id = job.id;
        machine.current_job_priority = job.priority;
        machine.remaining_time = job.processing_time;
        machine.energy_usage = 10.0 + job.priority * 2.0;
        this.state.job_queue.splice(jobIndex, 1);
        step_reward += 1.0; // Small reward for assigning
      } else {
        step_reward -= 2.0; // Penalty for invalid assignment
      }
    } else if (action.type === ActionType.MAINTENANCE && action.machine_id) {
      const machine = this.state.machines.find((m) => m.id === action.machine_id);
      if (machine && machine.status !== MachineStatus.BUSY) {
        machine.status = MachineStatus.MAINTENANCE;
        machine.remaining_time = 3;
        machine.energy_usage = 5.0;
        step_reward += 0.5; // Small reward for maintenance
      } else {
        step_reward -= 1.0;
      }
    }

    // 2. Advance Time
    this.state.time_step++;
    this.state.current_energy_usage = 0.0;

    // 3. Update Machines and Jobs
    for (const machine of this.state.machines) {
      this.state.current_energy_usage += machine.energy_usage;

      if (machine.status === MachineStatus.BUSY) {
        const intensity = machine.current_job_priority === 3 ? high_intensity_multiplier : 1.0;
        const degradation = base_degradation_rate * intensity;
        machine.health = Math.max(0, machine.health - degradation);
        
        machine.remaining_time--;

        // Random failure based on health and intensity
        const failure_prob = (1.0 - machine.health) * 0.05 * intensity;
        if (Math.random() < failure_prob) {
          machine.status = MachineStatus.BROKEN;
          machine.current_job_id = null;
          machine.remaining_time = 0;
          machine.energy_usage = 0.0;
          step_reward -= 20.0; // Failure penalty
        }

        if (machine.remaining_time <= 0 && machine.status === MachineStatus.BUSY) {
          machine.status = MachineStatus.IDLE;
          machine.current_job_id = null;
          machine.energy_usage = 0.0;
          this.state.jobs_completed++;
          step_reward += 10.0; // Job completed reward
        }
      } else if (machine.status === MachineStatus.IDLE) {
        // Slight aging even when idle
        machine.health = Math.max(0, machine.health - base_degradation_rate * 0.1);
        
        // Intermittent failures (temporary glitch)
        if (Math.random() < intermittent_failure_rate) {
          machine.status = MachineStatus.INTERMITTENT_FAILURE;
          machine.remaining_time = Math.floor(Math.random() * 3) + 1;
          machine.energy_usage = 1.0;
          step_reward -= 2.0;
        }
      } else if (machine.status === MachineStatus.MAINTENANCE) {
        machine.remaining_time--;
        if (machine.remaining_time <= 0) {
          machine.status = MachineStatus.IDLE;
          machine.health = 1.0; // Fully repaired
          machine.energy_usage = 0.0;
          step_reward += 5.0; // Maintenance completed bonus
        }
      } else if (machine.status === MachineStatus.INTERMITTENT_FAILURE) {
        machine.remaining_time--;
        if (machine.remaining_time <= 0) {
          machine.status = MachineStatus.IDLE;
          machine.energy_usage = 0.0;
        }
      } else if (machine.status === MachineStatus.BROKEN) {
        step_reward -= 0.5; // Idle broken machine penalty
      }
    }

    // 4. Update Job Queue
    for (const job of this.state.job_queue) {
      job.waiting_time++;
      step_reward -= 0.1 * job.priority; // Waiting penalty
    }

    // 5. Random Job Arrivals
    if (Math.random() < 0.3) {
      this.state.job_queue.push(this.generateJob(this.state.time_step));
    }

    // 6. Energy Constraints
    if (this.state.current_energy_usage > this.state.energy_budget) {
      step_reward -= (this.state.current_energy_usage - this.state.energy_budget) * 2.0;
    }

    // 7. Check Done
    if (this.state.time_step >= this.state.max_steps) {
      this.state.done = true;
    }

    this.state.total_reward += step_reward;

    return {
      state: JSON.parse(JSON.stringify(this.state)),
      reward: step_reward,
      done: this.state.done,
      info: {
        jobs_completed: this.state.jobs_completed,
        avg_health: this.state.machines.reduce((acc, m) => acc + m.health, 0) / this.state.machines.length,
      },
    };
  }

  public getState(): State {
    return JSON.parse(JSON.stringify(this.state));
  }
}
