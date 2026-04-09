/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { State } from "./models.ts";

function clampScore(score: number): number {
  return Math.max(0.01, Math.min(0.99, score));
}

function queueLength(state: State): number {
  return state.queue_length ?? state.job_queue.length;
}

export function energyEfficiencyGrader(state: State): number {
  const energyBudget = Math.max(state.energy_budget || 1, 1);
  const energyUsage = Math.max(state.current_energy_usage || 0, 0);
  const score = 1 - (energyUsage / Math.max(energyBudget * 1.25, 1));
  return clampScore(score);
}

export function throughputGrader(state: State): number {
  const totalJobs = Math.max(state.jobs_completed + queueLength(state), 1);
  const score = state.jobs_completed / totalJobs;
  return clampScore(score);
}

export function delayGrader(state: State): number {
  const queue = state.job_queue || [];
  const delay = state.delay ?? (
    queue.length > 0
      ? queue.reduce((sum, job) => sum + (job.waiting_time || 0), 0) / queue.length
      : 0
  );
  const realisticBound = Math.max((state.max_steps || 1) * 0.5, 5);
  const score = 1 - (delay / realisticBound);
  return clampScore(score);
}

export class SmartFactoryGrader {
  public static grade(state: State): number {
    const energyScore = energyEfficiencyGrader(state);
    const throughputScore = throughputGrader(state);
    const delayScore = delayGrader(state);
    const score = (energyScore * 0.35) + (throughputScore * 0.4) + (delayScore * 0.25);
    return clampScore(score);
  }

  public static getDetailedMetrics(state: State) {
    return {
      energy_efficiency: energyEfficiencyGrader(state),
      throughput_score: throughputGrader(state),
      delay_score: delayGrader(state),
      completion_rate: state.jobs_completed / Math.max(state.jobs_completed + queueLength(state), 1),
      throughput: state.throughput ?? (state.jobs_completed / Math.max(state.time_step || 1, 1)),
      delay: state.delay ?? 0,
      breakdown_risk: clampScore(state.breakdown_risk ?? 0.01),
      total_reward: state.total_reward,
    };
  }
}
