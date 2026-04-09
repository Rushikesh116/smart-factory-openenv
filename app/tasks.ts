/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Task } from "./models.ts";

export const TASKS: Task[] = [
  {
    id: "easy",
    name: "Energy Efficiency",
    description: "Optimize energy use under a tight budget while keeping the line productive.",
    objective: "energy_efficiency",
    grader: "energyEfficiencyGrader",
    config: {
      num_machines: 2,
      initial_jobs: 10,
      energy_budget: 50.0,
      failure_rate: 0.0,
      max_steps: 50,
    },
  },
  {
    id: "medium",
    name: "Throughput Optimization",
    description: "Increase completed jobs per episode while preserving machine reliability.",
    objective: "throughput",
    grader: "throughputGrader",
    config: {
      num_machines: 4,
      initial_jobs: 20,
      energy_budget: 80.0,
      failure_rate: 0.05,
      max_steps: 100,
    },
  },
  {
    id: "hard",
    name: "Low Latency Scheduling",
    description: "Minimize queueing delay in a failure-prone, high-load factory configuration.",
    objective: "low_latency",
    grader: "delayGrader",
    config: {
      num_machines: 8,
      initial_jobs: 30,
      energy_budget: 150.0,
      failure_rate: 0.15,
      max_steps: 200,
      intermittent_failure_rate: 0.05,
      degradation_rate: 0.01,
      high_intensity_multiplier: 3.0,
    },
  },
];
